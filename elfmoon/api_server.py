"""ElfMoon OpenAI 互換 API サーバ（generation-thread 方式）。

POST /v1/chat/completions   (stream/non-stream, OpenAI 互換)
GET  /v1/models

これにより Claude Code / VS Code Continue / Cursor / Zed / Open Interpreter 等の
OpenAI 互換 API をサポートする全ツールから ElfMoon を使える。

使い方:
    python3 api_server.py [port] [resident_capacity] [--model NAME] [--no-think]
    python3 api_server.py --list                      # 利用可能なモデル一覧

    デフォルト: port=11434, capacity=6144, バインド先=127.0.0.1, model=ELFMOON_MODEL(既定qwen3.6-35b-mlx)
    （LAN に公開する場合のみ ELFMOON_HOST=0.0.0.0 を指定。認証は無いので注意）
    モデル置き場は ELFMOON_MODELS_ROOT で指定（既定 ../models）。各モデルは
    <ELFMOON_MODELS_ROOT>/<name>/ に元重み一式 + integrate.py が作る store/ を持つ。

    curl http://localhost:11434/v1/chat/completions \\
      -d '{"model":"qwen3.6-35b","messages":[{"role":"user","content":"SwiftでFizzBuzzを書いて"}],"stream":true}'

Claude Code から使う場合 (~/.clauderc.json):
    {
      "models": [{
        "name": "elfmoon",
        "provider": "openai",
        "model": "qwen3.6-35b",
        "apiKey": "sk-not-needed",
        "baseUrl": "http://localhost:11434/v1"
      }]
    }
"""

import json

import logging
import os
from pathlib import Path
import re
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Queue
from socketserver import ThreadingMixIn
from threading import Thread, Event as ThreadEvent
from urllib.parse import urlparse

logging.disable(logging.WARNING)

import mlx.core as mx
from kv_manager import kv_manager
from mcp_client import mcp_manager, MCPError
from mlx_lm import load as _mlx_load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from stream_model import MODELS_ROOT, list_models, resolve_model, wire_streaming

HOST = os.environ.get("ELFMOON_HOST", "127.0.0.1")
DEFAULT_PORT = 11434
DEFAULT_CAPACITY = 6144
MODEL_ID = "elfmoon"
MAX_TOKENS = 16384
TEMP = 0.6
# プレフィルのチャンク幅。gather_qmm 経路では融合テンソル読込(~18GB/チャンク巡回)が
# チャンク数に比例する固定費のため、大きいほど長プロンプトで有利。8192 は活性化で
# ピーク ~21.7GB に達し 24GB 機では危険なため既定 4096（ピーク ~5GB）。
PREFILL_STEP = int(os.environ.get("ELFMOON_PREFILL_STEP", "4096"))
NO_THINK = "--no-think" in sys.argv


class ThinkStripper:
    """<think> ブロックをストリームから除去する（リクエスト毎に生成すること）。"""

    _PEEK = len("<think>")

    def __init__(self):
        self._buf = ""
        self._skip = True
        self._peeking = True

    def feed(self, piece):
        if not self._skip:
            return piece
        self._buf += piece
        if self._peeking:
            if len(self._buf) < self._PEEK and "<think>".startswith(self._buf):
                return None
            self._peeking = False
            if not self._buf.lstrip().startswith("<think"):
                self._skip = False
                out, self._buf = self._buf, ""
                return out if out else None
        idx = self._buf.find("</think>")
        if idx >= 0:
            self._skip = False
            after = self._buf[idx + 8 :]
            self._buf = ""
            return after if after else None
        return None

    @property
    def pending(self):
        return self._buf if self._skip else ""


# ---- generation engine（専用スレッドでモデルを動かす） ----


TOOL_CALL_START = "<|tool_call|>"
TOOL_CALL_END = "<tool_call|>"

_TC_START = re.escape(TOOL_CALL_START)
_TC_END = re.escape(TOOL_CALL_END)


def _match_brace(text: str, pos: int) -> int:
    """text[pos] が '{' の場合、対応する '}' の位置+1 を返す。"""
    assert text[pos] == "{"
    depth = 1
    i = pos + 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return i


def _parse_gemma4_args(text: str) -> dict:
    """Gemma4 の call:func{key:val,key2:"val2"} 形式の引数を dict に変換する。"""
    import ast

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    # 簡易パース: {key: "val", key2: 123} 形式
    # str.strip("{}") はネストした {} を壊すため使わない
    result = {}
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    stripped = s.strip()
    if not stripped:
        return result
    buf = stripped
    while buf:
        buf = buf.lstrip().lstrip(",").lstrip()
        if not buf:
            break
        # キー（引用符なし識別子 or 引用符あり文字列）
        if buf[0] in ('"', "'"):
            end = buf.find(buf[0], 1)
            key = buf[1:end]
            buf = buf[end + 1 :]
        else:
            m = re.match(r"(\w+)", buf)
            if not m:
                break
            key = m.group(1)
            buf = buf[m.end() :]
        buf = buf.lstrip()
        if buf and buf[0] == ":":
            buf = buf[1:]
        buf = buf.lstrip()
        # 値
        val, buf = _parse_gemma4_value(buf)
        result[key] = val
    return result


def _parse_gemma4_value(buf: str) -> tuple:
    """Gemma4 形式の値を1つパースして (value, rest) を返す。"""
    buf = buf.lstrip()
    if not buf:
        return None, ""
    if buf[0] in ('"', "'"):
        end = buf.find(buf[0], 1)
        val = buf[1:end]
        rest = buf[end + 1 :]
        # 引用符内のエスケープ
        val = val.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
        return val, rest
    if buf[0].isdigit() or buf[0] == "-":
        m = re.match(r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", buf)
        if m:
            raw = m.group(1)
            rest = buf[m.end() :]
            if "." in raw or "e" in raw.lower():
                return float(raw), rest
            return int(raw), rest
    if buf.startswith("true"):
        return True, buf[4:]
    if buf.startswith("false"):
        return False, buf[5:]
    if buf.startswith("none"):
        return None, buf[4:]
    if buf.startswith("null"):
        return None, buf[4:]
    if buf[0] == "{":
        depth = 1
        i = 1
        while i < len(buf) and depth > 0:
            if buf[i] == "{":
                depth += 1
            elif buf[i] == "}":
                depth -= 1
            i += 1
        inner = buf[1 : i - 1]
        rest = buf[i:]
        return _parse_gemma4_args(f"{{{inner}}}"), rest
    if buf[0] == "[":
        depth = 1
        i = 1
        items = []
        while i < len(buf) and depth > 0:
            if buf[i] == "[":
                depth += 1
            elif buf[i] == "]":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            elif buf[i] == "," and depth == 1:
                i += 1
                continue
            elif buf[i] == ",":
                i += 1
                continue
            i += 1
        rest = buf[i:]
        return items, rest
    # 引用符なしの識別子（リテラル文字列）
    m = re.match(r"([^,}\]]+)", buf)
    if m:
        val = m.group(1).strip().rstrip(",").rstrip("}").rstrip("]")
        rest = buf[m.end() :]
        return val, rest
    return None, buf[1:]


# Gemma4 tokenizer の特殊トークン置換マップ
_TOKEN_ARTIFACTS = {
    re.escape('<|"|>'): '"',
    re.escape("<|'|>"): "'",
    re.escape("<|\n|>"): "\n",
    re.escape("<|\r|>"): "\r",
    re.escape("<|\t|>"): "\t",
}


def _clean_token_artifacts(text: str) -> str:
    for pattern, replacement in _TOKEN_ARTIFACTS.items():
        text = re.sub(pattern, replacement, text)
    return text


def _extract_tool_calls(text: str) -> tuple[str, list[dict]]:
    """テキストから tool_call ブロックを抽出し、(マーカー除去済みテキスト, tool_call リスト) を返す。"""
    calls = []
    cleaned_parts = []
    i = 0

    while i < len(text):
        # 次の <|tool_call|> を探す
        start = text.find(TOOL_CALL_START, i)
        if start == -1:
            cleaned_parts.append(text[i:])
            break

        # 開始マーカー以前のテキストを保存
        cleaned_parts.append(text[i:start])
        content_start = start + len(TOOL_CALL_START)

        # <tool_call|> 終了マーカーを探す
        end = text.find(TOOL_CALL_END, content_start)
        if end == -1:
            cleaned_parts.append(text[i:])
            break

        body = text[content_start:end].strip()
        call_end = end + len(TOOL_CALL_END)

        parsed = None

        # Gemma4 形式: call:func_name{args}
        gemma4_m = re.match(r"call:(\w+)\s*(\{)", body)
        if gemma4_m:
            name = gemma4_m.group(1)
            brace_pos = content_start + gemma4_m.start(2)
            close = _match_brace(text, brace_pos)
            if close > 0:
                raw_text = text[brace_pos:close]
                args = _parse_gemma4_args(raw_text)
                parsed = {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
                call_end = close

        # OpenAI JSON 形式: {"name":..., "arguments":...}
        if parsed is None:
            json_m = re.match(r"\s*(\{)", body)
            if json_m:
                brace_pos = content_start + json_m.start(1)
                close = _match_brace(text, brace_pos)
                if close > 0:
                    raw = text[brace_pos:close]
                    try:
                        data = json.loads(raw)
                        name = data.get("name") or data.get("function", {}).get(
                            "name", ""
                        )
                        args = data.get("arguments") or data.get("function", {}).get(
                            "arguments", {}
                        )
                        if isinstance(args, str):
                            args = json.loads(args)
                        parsed = {
                            "id": f"call_{uuid.uuid4().hex[:12]}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                        call_end = close
                    except (json.JSONDecodeError, TypeError):
                        pass

        if parsed is not None:
            calls.append(parsed)
            i = call_end
        else:
            cleaned_parts.append(text[start:call_end])
            i = call_end

    cleaned = "".join(cleaned_parts).strip()
    return cleaned, calls


class GenerationEngine:
    """モデルを専用スレッドで保持し、リクエストを直列化して generation する。"""

    def __init__(self, model_path: str, store_dir: str, cap: int, perf: bool):
        self._queue = Queue()
        self._ready = ThreadEvent()
        self._thread = Thread(target=self._run, daemon=True)
        self._model_path = model_path
        self._store_dir = store_dir
        self._cap = cap
        self._perf = perf
        self._model = None
        self._tokenizer = None
        self._moe_cache = None

        self._thread.start()
        self._ready.wait()

    def generate(
        self,
        messages: list,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMP,
        no_think: bool = False,
        tools: list | None = None,
    ):
        cancel = ThreadEvent()
        q: Queue = Queue()
        self._queue.put(
            ("messages", q, cancel, messages, max_tokens, temperature, no_think, tools)
        )
        prompt_tokens = None
        try:
            while True:
                msg = q.get()
                if msg is None:
                    break
                if isinstance(msg, Exception):
                    raise msg
                if isinstance(msg, int):
                    prompt_tokens = msg
                    continue
                yield msg
                if cancel.is_set():
                    break
        except GeneratorExit:
            cancel.set()
            raise

    def generate_prompt(
        self,
        prompt: str,
        prompt_nogen: str,
        max_tokens: int,
        temperature: float,
        no_think: bool,
    ):
        cancel = ThreadEvent()
        q: Queue = Queue()
        self._queue.put(
            (
                "prompt",
                q,
                cancel,
                prompt,
                prompt_nogen,
                max_tokens,
                temperature,
                no_think,
            )
        )
        prompt_tokens = None
        try:
            while True:
                msg = q.get()
                if msg is None:
                    break
                if isinstance(msg, Exception):
                    raise msg
                if isinstance(msg, int):
                    prompt_tokens = msg
                    continue
                yield msg
                if cancel.is_set():
                    break
        except GeneratorExit:
            cancel.set()
            raise

    # ---- 以下、generation スレッド ---- #

    def _run(self):
        mx.eval(mx.array(0))
        mx.new_thread_local_stream(mx.default_device())
        self._load_model()
        mcp_manager.load()
        self._ready.set()
        err_count = 0
        while True:
            item = self._queue.get()
            req_type = item[0]
            if req_type == "messages":
                (
                    _dummy,
                    q,
                    cancel,
                    messages,
                    max_tokens,
                    temperature,
                    no_think,
                    tools,
                ) = item
                gen = self._generate_impl(
                    messages, max_tokens, temperature, no_think, tools
                )
            elif req_type == "prompt":
                (
                    _dummy,
                    q,
                    cancel,
                    prompt,
                    prompt_nogen,
                    max_tokens,
                    temperature,
                    no_think,
                ) = item
                gen = self._generate_legacy(
                    prompt, prompt_nogen, max_tokens, temperature, no_think
                )
            else:
                continue
            try:
                for msg in gen:
                    if cancel.is_set():
                        gen.close()
                        break
                    q.put(msg)
                err_count = 0
            except Exception as e:
                err_count += 1
                import traceback

                traceback.print_exc()
                q.put(Exception(str(e)))
            finally:
                q.put(None)

    def _load_model(self):
        mp = Path(self._model_path)
        with open(mp / "config.json") as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type", "")

        if model_type == "deepseek_v4":
            from model_v4 import DeepseekV4Model
            from stream_model import _wire_deepseek_v4

            self._model = DeepseekV4Model(str(mp), fused_quant=True)
            self._moe_cache, _ = _wire_deepseek_v4(
                self._model,
                self._cap,
                top_k=6,
                store_dir=self._store_dir,
                model_path=str(mp),
            )
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(mp))
        else:
            from mlx_lm.utils import load_model as _lm_load

            self._model, _ = _lm_load(mp, lazy=True)
            try:
                _tok_cfg = {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "add_prefix_space": False,
                }
                _, self._tokenizer = _mlx_load(
                    str(mp),
                    tokenizer_config=_tok_cfg,
                    lazy=True,
                )
            except Exception:
                from transformers import PreTrainedTokenizerFast
                from tokenizers import Tokenizer

                tk = Tokenizer.from_file(str(mp / "tokenizer.json"))
                self._tokenizer = PreTrainedTokenizerFast(tokenizer_object=tk)
                ct_path = mp / "chat_template.jinja"
                if ct_path.exists():
                    self._tokenizer.chat_template = ct_path.read_text()
                with open(mp / "config.json") as f:
                    _eos_cfg = json.load(f)
                eos_ids = _eos_cfg.get("eos_token_id", [])
                if isinstance(eos_ids, list) and eos_ids:
                    self._tokenizer.eos_token_id = eos_ids[0]
            if model_type != "gemma4" and os.path.isdir(os.path.join(str(mp), "store")):
                self._moe_cache, _ = wire_streaming(
                    self._model,
                    self._cap,
                    perf=self._perf,
                    store_dir=self._store_dir,
                    model_path=str(mp),
                )
            else:
                pass  # mx.compile は現在の環境で遅くなるためスキップ

    def _generate_legacy(self, prompt, prompt_nogen, max_tokens, temperature, no_think):
        """従来の高速パス: KV Cache 永続化 + 境界スナップショット対応。"""
        tokenizer = self._tokenizer
        model = self._model

        prompt_ids = tokenizer.encode(prompt)
        prompt_tokens = len(prompt_ids)
        yield prompt_tokens

        print(
            f"[ENGINE] prompt={prompt_tokens}tok max_tokens={max_tokens} temp={temperature}",
            file=sys.stderr,
            flush=True,
        )

        nogen_ids = tokenizer.encode(prompt_nogen)
        boundary = 0
        for i in range(min(len(nogen_ids), len(prompt_ids))):
            if prompt_ids[i] != nogen_ids[i]:
                break
            boundary = i + 1

        sampler = make_sampler(temp=temperature)
        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        eos_ids = getattr(tokenizer, "eos_token_ids", None) or {tokenizer.eos_token_id}
        stripper = ThinkStripper() if no_think else None

        cached_cache, cached_len = kv_manager.lookup(prompt_ids, model)

        if cached_cache is not None and cached_len < len(prompt_ids):
            prompt_cache = cached_cache
            print(
                f"[ENGINE] KVC hit offset={cached_len} new={len(prompt_ids) - cached_len}",
                file=sys.stderr,
                flush=True,
            )
        else:
            prompt_cache = make_prompt_cache(model)
            print(
                f"[ENGINE] KVC fresh (prompt={prompt_tokens})",
                file=sys.stderr,
                flush=True,
            )
            cached_len = 0

        save_key_ids = None
        snap = None
        prefill_t = time.time()
        if cached_len < boundary:
            remaining = prompt_ids[cached_len:boundary]
            step = PREFILL_STEP
            for i in range(0, len(remaining), step):
                chunk = remaining[i : i + step]
                model(mx.array([chunk]), cache=prompt_cache)
            snap = kv_manager.snapshot(prompt_cache)
            save_key_ids = prompt_ids[:boundary]
            print(
                f"[ENGINE] KVC history prefilled: {boundary - cached_len}tok in {time.time() - prefill_t:.1f}s",
                file=sys.stderr,
                flush=True,
            )

        remaining_ids = prompt_ids[boundary:]
        if not remaining_ids:
            remaining_ids = [tokenizer.eos_token_id]

        generate_t = time.time()
        generator = generate_step(
            mx.array(remaining_ids),
            model,
            max_tokens=max_tokens,
            sampler=sampler,
            prompt_cache=prompt_cache,
            prefill_step_size=PREFILL_STEP,
        )
        n = 0
        try:
            for token, _logprob in generator:
                if token in eos_ids:
                    break
                detokenizer.add_token(token)
                piece = detokenizer.last_segment
                if not piece:
                    continue
                n += 1
                if stripper is not None:
                    piece = stripper.feed(piece)
                    if piece is None:
                        continue
                yield (piece, n)
            if stripper is not None and stripper.pending:
                yield (stripper.pending, n)
        except Exception as e:
            print(f"[ENGINE] error at token {n}: {e}", file=sys.stderr, flush=True)
            raise
        finally:
            print(
                f"[ENGINE] done: {n} tokens in {time.time() - generate_t:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            if save_key_ids is not None:
                kv_manager.save(save_key_ids, snap)

    def _generate_impl(self, messages, max_tokens, temperature, no_think, tools):
        tokenizer = self._tokenizer
        model = self._model
        eos_ids = getattr(tokenizer, "eos_token_ids", None) or {tokenizer.eos_token_id}

        # MCP ツールは API ハンドラ側で注入済み。ここでは何もしない

        MAX_ROUNDS = 10
        round_idx = 0
        yielded_prompt_tokens = False

        while round_idx < MAX_ROUNDS:
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=tools if round_idx == 0 else None,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=not no_think,
            )
            prompt_ids = tokenizer.encode(prompt)

            if not yielded_prompt_tokens:
                yield len(prompt_ids)
                yielded_prompt_tokens = True
                print(
                    f"[ENGINE] prompt={len(prompt_ids)}tok max_tokens={max_tokens} temp={temperature} tools={bool(tools)}",
                    file=sys.stderr,
                    flush=True,
                )

            prompt_cache = make_prompt_cache(model)
            for i in range(0, len(prompt_ids), PREFILL_STEP):
                model(mx.array([prompt_ids[i : i + PREFILL_STEP]]), cache=prompt_cache)

            sampler = make_sampler(temp=temperature)
            remaining = prompt_ids[-1:] if prompt_ids else [tokenizer.eos_token_id]

            generate_t = time.time()
            generator = generate_step(
                mx.array(remaining),
                model,
                max_tokens=max_tokens,
                sampler=sampler,
                prompt_cache=prompt_cache,
            )
            output_ids = []
            for token, _logprob in generator:
                if token in eos_ids:
                    break
                output_ids.append(token)
            elapsed = time.time() - generate_t
            print(
                f"[ENGINE] round {round_idx}: {len(output_ids)} tokens in {elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )

            if not output_ids:
                break

            output_text = _clean_token_artifacts(
                tokenizer.decode(output_ids, skip_special_tokens=False)
            )

            if tools:
                clean_text, tool_calls = _extract_tool_calls(output_text)
            else:
                clean_text = output_text
                tool_calls = []

            if not tool_calls:
                detokenizer = tokenizer.detokenizer
                detokenizer.reset()
                stripper = ThinkStripper() if no_think else None
                n = 0
                for token in output_ids:
                    detokenizer.add_token(token)
                    piece = detokenizer.last_segment
                    if not piece:
                        continue
                    n += 1
                    if stripper is not None:
                        piece = stripper.feed(piece)
                        if piece is None:
                            continue
                    yield (piece, n)
                if stripper is not None and stripper.pending:
                    yield (stripper.pending, n)
                return

            messages.append(
                {
                    "role": "assistant",
                    "content": clean_text or None,
                    "tool_calls": tool_calls,
                }
            )

            for tc in tool_calls:
                try:
                    name = tc["function"]["name"]
                    args = json.loads(tc["function"]["arguments"])
                    result = mcp_manager.call_tool(name, args)
                except Exception as call_err:
                    result = f"Error: {call_err}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(result),
                    }
                )

            round_idx += 1


# ---- HTTP ----


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


engine: GenerationEngine = None


def _get_engine():
    global engine
    return engine


class APIHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/v1/models":
            return self._handle_models()
        self._send_json(404, {"error": "not_found", "message": f"Not found: {path}"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/v1/chat/completions":
            return self._handle_chat_completions()
        self._send_json(404, {"error": "not_found", "message": f"Not found: {path}"})

    # ---- handlers ----

    def _handle_models(self):
        data = {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "elfmoon",
                }
            ],
        }
        self._send_json(200, data)

    def _handle_chat_completions(self):
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len)) if content_len else {}
        except (json.JSONDecodeError, ValueError) as e:
            return self._send_json(400, {"error": "invalid_request", "message": str(e)})

        messages = body.get("messages", [])
        req_id = body.get("model", "?")
        stream = body.get("stream", False)
        print(
            f"[API] chat req model={req_id} stream={stream} msgs={len(messages)} t0={time.time():.3f}",
            file=sys.stderr,
            flush=True,
        )
        if not messages:
            return self._send_json(
                400, {"error": "invalid_request", "message": "messages is required"}
            )
        if req_id != "?" and req_id != MODEL_ID:
            return self._send_json(
                400,
                {
                    "error": "model_not_loaded",
                    "message": (
                        f"model='{req_id}' はロードされていません。"
                        f"現在ロード中: {MODEL_ID}。"
                        f" クライアント設定で model を {MODEL_ID} に修正してください。"
                    ),
                },
            )

        max_tokens = min(body.get("max_tokens", MAX_TOKENS), MAX_TOKENS)
        temperature = body.get("temperature", TEMP)
        tools = body.get("tools", None)
        tool_choice = body.get("tool_choice", "auto")

        if tools is None:
            mcp_tools = mcp_manager.get_openai_tools()
            if mcp_tools:
                tools = mcp_tools
                tool_choice = "auto"
                print(
                    f"[API] クライアントからツール未指定 → MCP {len(mcp_tools)} ツールを注入",
                    file=sys.stderr,
                    flush=True,
                )

        # ツールなし → 従来通り API ハンドラ側で prompt レンダリング（高速パス）
        # ツールあり → エンジンに messages + tools を渡してループ処理
        if not tools:
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=not NO_THINK,
                )
                prompt_nogen = self._tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=False,
                    enable_thinking=not NO_THINK,
                )
            except Exception as e:
                return self._send_json(
                    400,
                    {
                        "error": "invalid_request",
                        "message": f"chat_template error: {e}",
                    },
                )

            if stream:
                self._handle_stream_legacy(
                    prompt, prompt_nogen, max_tokens, temperature
                )
            else:
                self._handle_nonstream_legacy(
                    prompt, prompt_nogen, max_tokens, temperature
                )
        else:
            if stream:
                self._handle_stream_tools(messages, max_tokens, temperature, tools)
            else:
                self._handle_nonstream_tools(messages, max_tokens, temperature, tools)

    @property
    def _tokenizer(self):
        return _get_engine()._tokenizer

    # ---- 従来の高速パス（ツールなし） ---- #

    def _handle_stream_legacy(self, prompt, prompt_nogen, max_tokens, temperature):
        t0 = time.time()
        self.send_response(200)
        self._cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        completion_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())
        total = 0
        prompt_tokens = 0
        error = False

        gen = _get_engine().generate_prompt(
            prompt, prompt_nogen, max_tokens, temperature, NO_THINK
        )
        try:
            for msg in gen:
                if isinstance(msg, int):
                    prompt_tokens = msg
                    continue
                piece, n = msg
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [
                        {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                    ],
                }
                self._sse(json.dumps(chunk, ensure_ascii=False))
                total = n
        except Exception as e:
            error = True
            print(
                f"[API] stream error at token {total}: {e}", file=sys.stderr, flush=True
            )
            err_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            }
            try:
                self._sse(json.dumps(err_chunk, ensure_ascii=False))
            except OSError:
                pass

    def _handle_nonstream_legacy(self, prompt, prompt_nogen, max_tokens, temperature):
        t0 = time.time()
        pieces = []
        total = 0
        prompt_tokens = 0
        gen = _get_engine().generate_prompt(
            prompt, prompt_nogen, max_tokens, temperature, NO_THINK
        )
        try:
            for msg in gen:
                if isinstance(msg, int):
                    prompt_tokens = msg
                    continue
                piece, n = msg
                pieces.append(piece)
                total = n
        except Exception as e:
            print(f"[API] generate error: {e}", file=sys.stderr, flush=True)
            return self._send_json(
                500, {"error": "generation_error", "message": str(e)}
            )

        text = "".join(pieces)
        print(
            f"[API] generate done in {time.time() - t0:.3f}s",
            file=sys.stderr,
            flush=True,
        )
        resp = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total,
                "total_tokens": prompt_tokens + total,
            },
        }
        self._send_json(200, resp)

    # ---- ツール対応パス ---- #

    def _handle_stream_tools(self, messages, max_tokens, temperature, tools):
        t0 = time.time()
        self.send_response(200)
        self._cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        completion_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())
        total = 0
        prompt_tokens = 0
        error = False

        gen = _get_engine().generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            no_think=NO_THINK,
            tools=tools,
        )
        try:
            for msg in gen:
                if isinstance(msg, int):
                    prompt_tokens = msg
                    continue
                piece, n = msg
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [
                        {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                    ],
                }
                self._sse(json.dumps(chunk, ensure_ascii=False))
                total = n
        except Exception as e:
            error = True
            print(
                f"[API] stream error at token {total}: {e}",
                file=sys.stderr,
                flush=True,
            )
            err_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            }
            try:
                self._sse(json.dumps(err_chunk, ensure_ascii=False))
            except OSError:
                pass

        dt = time.time() - t0
        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total,
                "total_tokens": prompt_tokens + total,
            },
        }
        try:
            self._sse(json.dumps(final, ensure_ascii=False))
            self._sse("[DONE]")
        except OSError:
            pass
        print(
            f"[API] stream tools done: {total} tokens in {dt:.1f}s ({total / dt:.1f} t/s) error={error}",
            file=sys.stderr,
            flush=True,
        )

    def _handle_nonstream_tools(self, messages, max_tokens, temperature, tools):
        t0 = time.time()
        pieces = []
        total = 0
        prompt_tokens = 0
        gen = _get_engine().generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            no_think=NO_THINK,
            tools=tools,
        )
        try:
            for msg in gen:
                if isinstance(msg, int):
                    prompt_tokens = msg
                    continue
                piece, n = msg
                pieces.append(piece)
                total = n
        except Exception as e:
            print(f"[API] generate error: {e}", file=sys.stderr, flush=True)
            return self._send_json(
                500, {"error": "generation_error", "message": str(e)}
            )

        text = "".join(pieces)
        print(
            f"[API] generate tools done in {time.time() - t0:.3f}s",
            file=sys.stderr,
            flush=True,
        )
        resp = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total,
                "total_tokens": prompt_tokens + total,
            },
        }
        self._send_json(200, resp)

    # ---- helpers ----

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, data):
        self.wfile.write(f"data: {data}\n\n".encode())
        self.wfile.flush()

    def log_message(self, fmt, *args):
        print(f"[API] {fmt % args}", file=sys.stderr, flush=True)


def main():
    argv = sys.argv[1:]

    if "--list" in argv:
        models = list_models()
        print(f"利用可能なモデル（ELFMOON_MODELS_ROOT={MODELS_ROOT}）:")
        for name, has_store, is_native in models:
            if is_native:
                print(f"  {name}  ✅ オンメモリ動作")
            elif has_store:
                print(f"  {name}")
            else:
                print(f"  {name}  ⚠️ store/ 未生成（integrate.py split_all が必要）")
        if not models:
            print("  (見つかりません)")
        return

    perf = "--perf" in argv or os.environ.get("ELFMOON_PERF") == "1"
    model_name = None
    if "--model" in argv:
        idx = argv.index("--model")
        model_name = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2 :]
    args = [a for a in argv if a not in ("--no-think", "--perf")]
    port = int(args[0]) if len(args) > 0 else DEFAULT_PORT
    cap = int(args[1]) if len(args) > 1 else DEFAULT_CAPACITY

    model_path, store_dir = resolve_model(model_name)

    global MODEL_ID, engine
    MODEL_ID = model_name or os.path.basename(model_path)

    mode = "性能" if perf else "省メモリ"
    print(f"モデル: {model_path}", flush=True)
    print(f"モデルをロード中...（{mode}モード, capacity={cap}）", flush=True)
    t0 = time.perf_counter()

    engine = GenerationEngine(model_path, store_dir, cap, perf)

    print(f"準備完了（{time.perf_counter() - t0:.0f}秒）", flush=True)
    print("", flush=True)
    print(f"  ElfMoon API サーバ起動: http://{HOST}:{port}", flush=True)
    if HOST == "127.0.0.1":
        print(
            "  （LAN公開する場合: ELFMOON_HOST=0.0.0.0 で起動。認証なし注意）",
            flush=True,
        )
    print("  POST /v1/chat/completions  (OpenAI 互換, stream/non-stream)", flush=True)
    print("  GET  /v1/models", flush=True)
    print("", flush=True)
    print("  Claude Code 設定例 (~/.clauderc.json または claude.json):", flush=True)
    print('    {"models":[{"name":"elfmoon","provider":"openai",', flush=True)
    print(f'      "model":"{MODEL_ID}","apiKey":"sk-not-needed",', flush=True)
    print(f'      "baseUrl":"http://localhost:{port}/v1"}}]}}', flush=True)
    print("", flush=True)
    print("  VS Code Continue 設定例 (~/.continue/config.json):", flush=True)
    print('    {"models":[{"title":"ElfMoon","provider":"openai",', flush=True)
    print(
        f'      "model":"{MODEL_ID}","apiBase":"http://localhost:{port}/v1"}}]}}',
        flush=True,
    )
    print("  Ctrl-C で終了", flush=True)

    server = ThreadingHTTPServer((HOST, port), APIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nシャットダウン中...")
        server.shutdown()


if __name__ == "__main__":
    main()

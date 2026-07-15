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
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Queue
from socketserver import ThreadingMixIn
from threading import Thread, Event as ThreadEvent
from urllib.parse import urlparse

logging.disable(logging.WARNING)

import mlx.core as mx
from kv_manager import kv_manager
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


class GenerationEngine:
    """モデルを専用スレッドで保持し、リクエストを直列化して generation する。

    HTTP スレッドは GPU に一切触れず、Engine にリクエストを渡すだけ。
    Engine は内部で generate_step を使い、chat.py と同一の性能を発揮する。
    """

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
        self._ready.wait()  # モデルロード完了を待つ

    def generate(
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
            (q, cancel, prompt, prompt_nogen, max_tokens, temperature, no_think)
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
        self._ready.set()
        err_count = 0
        while True:
            item = self._queue.get()
            q, cancel, prompt, prompt_nogen, max_tokens, temperature, no_think = item
            try:
                gen = self._generate_impl(
                    prompt, prompt_nogen, max_tokens, temperature, no_think
                )
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

    def _generate_impl(self, prompt, prompt_nogen, max_tokens, temperature, no_think):
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

        # 手動プリフィル: 境界（履歴終端）まで → snapshot
        save_key_ids = None
        snap = None
        prefill_t = time.time()
        if cached_len < boundary:
            remaining = prompt_ids[cached_len:boundary]
            step = 2048
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

        # generate_step に残り全部を渡す（生成プロンプトのプリフィル + 生成を一括）
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
                {"error": "invalid_request", "message": f"chat_template error: {e}"},
            )

        if stream:
            self._handle_stream(prompt, prompt_nogen, max_tokens, temperature)
        else:
            self._handle_nonstream(prompt, prompt_nogen, max_tokens, temperature)

    @property
    def _tokenizer(self):
        return _get_engine()._tokenizer

    def _handle_stream(self, prompt, prompt_nogen, max_tokens, temperature):
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
            f"[API] stream done: {total} tokens in {dt:.1f}s ({total / dt:.1f} t/s) error={error}",
            file=sys.stderr,
            flush=True,
        )

    def _handle_nonstream(self, prompt, prompt_nogen, max_tokens, temperature):
        t0 = time.time()
        pieces = []
        total = 0
        prompt_tokens = 0
        gen = _get_engine().generate(
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

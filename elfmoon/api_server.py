"""ElfMoon OpenAI 互換 API サーバ（KV Cache 永続化対応）。

POST /v1/chat/completions   (stream/non-stream, OpenAI 互換)
GET  /v1/models

これにより Claude Code / VS Code Continue / Cursor / Zed / Open Interpreter 等の
OpenAI 互換 API をサポートする全ツールから ElfMoon を使える。

使い方:
    python3 api_server.py [port] [resident_capacity] [--no-think]

    デフォルト: port=11434, capacity=6144, バインド先=127.0.0.1
    （LAN に公開する場合のみ ELFMOON_HOST=0.0.0.0 を指定。認証は無いので注意）

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
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse
from threading import Lock
import mlx.core as mx
from mlx_lm import load as _mlx_load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from stream_model import wire_streaming, MODEL_PATH
from kv_manager import kv_manager
from mlx_lm.models.cache import make_prompt_cache


HOST = os.environ.get("ELFMOON_HOST", "127.0.0.1")
DEFAULT_PORT = 11434
DEFAULT_CAPACITY = 6144
MODEL_ID = "qwen3.6-35b"
MAX_TOKENS = 4096
TEMP = 0.6
NO_THINK = "--no-think" in sys.argv


class ThinkStripper:
    """<think> ブロックをストリームから除去する（リクエスト毎に生成すること）。"""

    def __init__(self):
        self._buf = ""
        self._skip = True

    def feed(self, piece):
        """テキスト断片を処理。出力すべきテキストか None（保留中）を返す。"""
        if not self._skip:
            return piece
        self._buf += piece
        idx = self._buf.find("</think>")
        if idx >= 0:
            self._skip = False
            after = self._buf[idx + 8 :]
            self._buf = ""
            return after if after else None
        return None

    @property
    def pending(self):
        """ストリーム終了時に </think> 未出現なら溜めた分を返す（応答消失防止）。"""
        return self._buf if self._skip else ""


model = None
tokenizer = None
cache = None
# 生成リクエスト全体を直列化するロック。
# 共有 detokenizer と MoE 常駐キャッシュを並行リクエストの混線から守る。
model_lock = Lock()


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


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

        max_tokens = min(body.get("max_tokens", MAX_TOKENS), MAX_TOKENS)
        temperature = body.get("temperature", TEMP)

        try:
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            # 生成プロンプト（<|im_start|>assistant<think> 等）を除いた安定境界。
            # マルチターンの延長プロンプトはこの境界までが共通prefixになる。
            prompt_nogen = tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
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

    def _generate_cached(self, prompt, prompt_nogen, max_tokens, temperature):
        """KV Cache 永続化 generation。yields (piece: str, n: int)。

        model_lock をリクエスト全体で保持して直列化する（シングルユーザー前提）。
        スナップショットは生成プロンプト末尾を除いた「安定境界」で取得する
        （マルチターンの延長プロンプトがこの境界で prefix ヒットするため）。
        """
        global model, tokenizer

        prompt_ids = tokenizer.encode(prompt)
        self._prompt_tokens = len(prompt_ids)
        print(
            f"[API] generate prompt={len(prompt_ids)}tok max_tokens={max_tokens} temp={temperature}",
            file=sys.stderr,
            flush=True,
        )

        # 安定境界 B（token単位）: 生成プロンプト末尾（<|im_start|>assistant<think> 等）を
        # 除いた位置。トークン化が prefix 性を満たさない場合は末尾-1 に退避。
        nogen_ids = tokenizer.encode(prompt_nogen)
        boundary = len(nogen_ids)
        if not (0 < boundary < len(prompt_ids) and prompt_ids[:boundary] == nogen_ids):
            boundary = len(prompt_ids) - 1

        with model_lock:
            cached_cache, cached_len = kv_manager.lookup(prompt_ids, model)

            if cached_cache is not None and cached_len < len(prompt_ids):
                prompt_cache = cached_cache
                print(
                    f"[KVC] hit offset={cached_len} new_ids={len(prompt_ids) - cached_len}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                prompt_cache = make_prompt_cache(model)
                if cached_cache is not None:
                    print(
                        f"[KVC] miss (cached_len={cached_len} vs prompt={len(prompt_ids)})",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(
                        f"[KVC] fresh (prompt={len(prompt_ids)})",
                        file=sys.stderr,
                        flush=True,
                    )
                cached_len = 0

            def _prefill(ids):
                if ids:
                    model(mx.array([ids]), cache=prompt_cache)
                    mx.eval([c.state for c in prompt_cache])

            prefill_t = time.time()
            snap = None
            save_key_ids = None
            if cached_len <= boundary:
                # 境界まで prefill → 整合スナップショット → 生成プロンプト部を prefill
                _prefill(prompt_ids[cached_len:boundary])
                snap = kv_manager.snapshot(prompt_cache)
                save_key_ids = prompt_ids[:boundary]
                _prefill(prompt_ids[boundary : len(prompt_ids) - 1])
            else:
                # 復元キャッシュが境界より長い＝同等以上の保存済みエントリあり → 保存不要
                _prefill(prompt_ids[cached_len : len(prompt_ids) - 1])
            if len(prompt_ids) - 1 > cached_len:
                print(
                    f"[KVC] prefill done in {time.time() - prefill_t:.1f}s"
                    f" (boundary={boundary})",
                    file=sys.stderr,
                    flush=True,
                )

            start_prompt = mx.array([prompt_ids[-1]])
            sampler = make_sampler(temp=temperature)
            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            eos_ids = getattr(tokenizer, "eos_token_ids", None) or {
                tokenizer.eos_token_id
            }
            stripper = ThinkStripper() if NO_THINK else None
            n = 0

            generate_t = time.time()
            generator = generate_step(
                start_prompt,
                model,
                max_tokens=max_tokens,
                sampler=sampler,
                prompt_cache=prompt_cache,
            )
            try:
                while True:
                    try:
                        token, _ = next(generator)
                    except StopIteration:
                        break
                    if token in eos_ids:
                        print(
                            f"[API] EOS at token {n} (elapsed {time.time() - generate_t:.1f}s)",
                            file=sys.stderr,
                            flush=True,
                        )
                        break
                    try:
                        detokenizer.add_token(token)
                        piece = detokenizer.last_segment
                    except Exception as detok_err:
                        print(
                            f"[API] detokenizer error at token {n}: {detok_err}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    if not piece:
                        continue
                    n += 1
                    if stripper is not None:
                        piece = stripper.feed(piece)
                        if piece is None:
                            continue
                    yield piece, n
                # </think> が最後まで現れなかった場合、溜めた分を破棄せず出力する
                if stripper is not None and stripper.pending:
                    yield stripper.pending, n
            except Exception as e:
                print(
                    f"[API] generate error at token {n}: {e}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                print(
                    f"[API] generate yield: {n} tokens in {time.time() - generate_t:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                # 安定境界時点の整合状態を保存（キー＝先頭 boundary トークン）
                if save_key_ids is not None:
                    kv_manager.save(save_key_ids, snap)

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
        error = False

        try:
            for piece, n in self._generate_cached(
                prompt, prompt_nogen, max_tokens, temperature
            ):
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": piece},
                            "finish_reason": None,
                        }
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
            self._sse(json.dumps(err_chunk, ensure_ascii=False))

        dt = time.time() - t0
        prompt_tokens = getattr(self, "_prompt_tokens", 0)
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
        self._sse(json.dumps(final, ensure_ascii=False))
        self._sse("[DONE]")
        print(
            f"[API] stream done: {total} tokens in {dt:.1f}s ({total / dt:.1f} t/s)"
            f" error={error}",
            file=sys.stderr,
            flush=True,
        )

    def _handle_nonstream(self, prompt, prompt_nogen, max_tokens, temperature):
        t0 = time.time()
        pieces = []
        total = 0
        try:
            for piece, n in self._generate_cached(
                prompt, prompt_nogen, max_tokens, temperature
            ):
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

        prompt_tokens = getattr(self, "_prompt_tokens", 0)
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
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def log_message(self, fmt, *args):
        print(f"[API] {fmt % args}", file=sys.stderr, flush=True)


def main():
    import os

    perf = "--perf" in sys.argv or os.environ.get("ELFMOON_PERF") == "1"
    args = [a for a in sys.argv[1:] if a not in ("--no-think", "--perf")]
    port = int(args[0]) if len(args) > 0 else DEFAULT_PORT
    cap = int(args[1]) if len(args) > 1 else DEFAULT_CAPACITY

    mode = "性能" if perf else "省メモリ"
    global model, tokenizer, cache
    print(
        f"モデルをロード中...（{mode}モード, capacity={cap}）",
        flush=True,
    )
    t0 = time.perf_counter()
    # Load model with tokenizer using PreTrainedTokenizerFast for Qwen3.6 compat
    _tok_cfg = {"tokenizer_class": "PreTrainedTokenizerFast", "add_prefix_space": False}
    model, tokenizer = _mlx_load(MODEL_PATH, tokenizer_config=_tok_cfg, lazy=True)
    cache, _ = wire_streaming(model, cap, perf=perf)
    print(f"準備完了（{time.perf_counter() - t0:.0f}秒）", flush=True)

    print(f"", flush=True)
    print(f"  ElfMoon API サーバ起動: http://{HOST}:{port}", flush=True)
    if HOST == "127.0.0.1":
        print(
            f"  （LAN公開する場合: ELFMOON_HOST=0.0.0.0 で起動。認証なし注意）",
            flush=True,
        )
    print(f"  POST /v1/chat/completions  (OpenAI 互換, stream/non-stream)", flush=True)
    print(f"  GET  /v1/models", flush=True)
    print(f"", flush=True)
    print(f"  Claude Code 設定例 (~/.clauderc.json または claude.json):", flush=True)
    print(f'    {{"models":[{{"name":"elfmoon","provider":"openai",', flush=True)
    print(f'      "model":"{MODEL_ID}","apiKey":"sk-not-needed",', flush=True)
    print(f'      "baseUrl":"http://localhost:{port}/v1"}}]}}', flush=True)
    print(f"", flush=True)
    print(f"  VS Code Continue 設定例 (~/.continue/config.json):", flush=True)
    print(f'    {{"models":[{{"title":"ElfMoon","provider":"openai",', flush=True)
    print(
        f'      "model":"{MODEL_ID}","apiBase":"http://localhost:{port}/v1"}}]}}',
        flush=True,
    )
    print(f"  Ctrl-C で終了", flush=True)

    server = ThreadingHTTPServer((HOST, port), APIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nシャットダウン中...")
        server.shutdown()


if __name__ == "__main__":
    main()

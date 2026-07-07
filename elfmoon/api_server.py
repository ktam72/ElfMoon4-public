"""ElfMoon OpenAI 互換 API サーバ（KV Cache 永続化対応）。

POST /v1/chat/completions   (stream/non-stream, OpenAI 互換)
GET  /v1/models

これにより Claude Code / VS Code Continue / Cursor / Zed / Open Interpreter 等の
OpenAI 互換 API をサポートする全ツールから ElfMoon を使える。

使い方:
    python3 api_server.py [port] [resident_capacity]

    デフォルト: port=11434, capacity=2800
    curl http://localhost:11434/v1/chat/completions \\
      -d '{"model":"qwen3-30b-instruct","messages":[{"role":"user","content":"SwiftでFizzBuzzを書いて"}],"stream":true}'

Claude Code から使う場合 (~/.clauderc.json):
    {
      "models": [{
        "name": "elfmoon",
        "provider": "openai",
        "model": "qwen3-30b-instruct",
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
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from tokenizers import Tokenizer as TkTokenizer
from stream_model import wire_streaming, MODEL_PATH
from kv_manager import kv_manager
from mlx_lm.models.cache import make_prompt_cache


DEFAULT_PORT = 11434
DEFAULT_CAPACITY = 6144
MAX_TOKENS = 4096
MAX_PROMPT_TOKENS = 4096
TEMP = 0.6
NO_THINK = "--no-think" in sys.argv


_think_buf = ""
_think_skip = True


def _strip_think_text(piece):
    global _think_buf, _think_skip
    if not _think_skip:
        return piece
    _think_buf += piece
    idx = _think_buf.find("</think>")
    if idx >= 0:
        _think_skip = False
        after = _think_buf[idx + 8 :]
        _think_buf = ""
        return after if after else None
    return None


model = None
tokenizer = None
cache = None
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
                    "id": "qwen3-30b-instruct",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "elfmoon",
                }
            ],
        }
        self._send_json(200, data)

    def _handle_chat_completions(self):
        import sys

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
        except Exception as e:
            return self._send_json(
                400,
                {"error": "invalid_request", "message": f"chat_template error: {e}"},
            )

        if stream:
            self._handle_stream(prompt, max_tokens, temperature)
        else:
            self._handle_nonstream(prompt, max_tokens, temperature)

    def _generate_cached(self, prompt, max_tokens, temperature):
        """KV Cache 永続化 generation。yields (piece: str, n: int)。"""
        global model, tokenizer

        prompt_ids = tokenizer.encode(prompt)
        print(
            f"[API] generate prompt={len(prompt_ids)}tok max_tokens={max_tokens} temp={temperature}",
            file=sys.stderr,
            flush=True,
        )

        cached_cache, cached_len = kv_manager.lookup(prompt_ids, model)

        if cached_cache is not None and cached_len < len(prompt_ids):
            prompt_cache = cached_cache
            new_ids = prompt_ids[cached_len:]
            print(
                f"[KVC] hit offset={cached_len} new_ids={len(new_ids)}",
                file=sys.stderr,
                flush=True,
            )
        else:
            prompt_cache = make_prompt_cache(model)
            new_ids = prompt_ids
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

        prefill_t = time.time()
        if len(new_ids) > 1:
            chunk = mx.array([new_ids[:-1]])
            with model_lock:
                model(chunk, cache=prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            print(
                f"[KVC] prefill done in {time.time() - prefill_t:.1f}s",
                file=sys.stderr,
                flush=True,
            )

        start_prompt = (
            mx.array([new_ids[-1]]) if len(new_ids) > 0 else mx.array([prompt_ids[-1]])
        )
        sampler = make_sampler(temp=temperature)
        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
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
                    with model_lock:
                        token, _ = next(generator)
                except StopIteration:
                    break
                if token == tokenizer.eos_token_id:
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
                if NO_THINK:
                    piece = _strip_think_text(piece)
                    if piece is None:
                        continue
                yield piece, n
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
            kv_manager.save(prompt_ids, prompt_cache)

    def _handle_stream(self, prompt, max_tokens, temperature):
        import sys

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
            for piece, n in self._generate_cached(prompt, max_tokens, temperature):
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "qwen3-30b-instruct",
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
                "model": "qwen3-30b-instruct",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            }
            self._sse(json.dumps(err_chunk, ensure_ascii=False))

        dt = time.time() - t0
        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "qwen3-30b-instruct",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": total,
                "total_tokens": 0,
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

    def _handle_nonstream(self, prompt, max_tokens, temperature):
        t0 = time.time()
        pieces = []
        try:
            for piece, _ in self._generate_cached(prompt, max_tokens, temperature):
                pieces.append(piece)
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
            "model": "qwen3-30b-instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
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
        import sys

        print(f"[API] {fmt % args}", file=sys.stderr, flush=True)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CAPACITY

    global model, tokenizer, cache
    print(
        f"モデルをロード中...（常駐 {cap} experts ≈ {cap * 1.69 / 1000:.1f}GB）",
        flush=True,
    )
    t0 = time.perf_counter()
    # Load model with tokenizer using PreTrainedTokenizerFast for Qwen3.6 compat
    _tok_cfg = {"tokenizer_class": "PreTrainedTokenizerFast", "add_prefix_space": False}
    model, tokenizer = _mlx_load(MODEL_PATH, tokenizer_config=_tok_cfg)
    cache, _ = wire_streaming(model, cap)
    print(f"準備完了（{time.perf_counter() - t0:.0f}秒）", flush=True)

    print(f"", flush=True)
    print(f"  ElfMoon API サーバ起動: http://localhost:{port}", flush=True)
    print(f"  POST /v1/chat/completions  (OpenAI 互換, stream/non-stream)", flush=True)
    print(f"  GET  /v1/models", flush=True)
    print(f"", flush=True)
    print(f"  Claude Code 設定例 (~/.clauderc.json または claude.json):", flush=True)
    print(f'    {{"models":[{{"name":"elfmoon","provider":"openai",', flush=True)
    print(f'      "model":"qwen3-30b-instruct","apiKey":"sk-not-needed",', flush=True)
    print(f'      "baseUrl":"http://localhost:{port}/v1"}}]}}', flush=True)
    print(f"", flush=True)
    print(f"  VS Code Continue 設定例 (~/.continue/config.json):", flush=True)
    print(f'    {{"models":[{{"title":"ElfMoon","provider":"openai",', flush=True)
    print(
        f'      "model":"qwen3-30b-instruct","apiBase":"http://localhost:{port}/v1"}}]}}',
        flush=True,
    )
    print(f"  Ctrl-C で終了", flush=True)

    server = ThreadingHTTPServer(("0.0.0.0", port), APIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nシャットダウン中...")
        server.shutdown()


if __name__ == "__main__":
    main()

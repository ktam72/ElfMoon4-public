"""ElfMoon 対話型チャットCLI。

モデルを一度だけロード＆ストリーミング化し、以降は対話ループで
何度でもプロンプトを投げられる。会話履歴を保持する。

使い方:
    cd elfmoon
    python3 chat.py            # 常駐 6144 (既定)
    python3 chat.py 1200       # 省メモリ
"""

import sys
import time
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from stream_model import wire_streaming, MODEL_PATH

SYSTEM = "You are an expert coding assistant. Write clean, correct, concise code."
MAX_TOKENS = 4096
MAX_HISTORY = 8
TEMP = 0.2


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 6144
    print(f"モデルをロード中...（常駐 {cap} experts ≈ {cap * 1.69 / 1000:.1f}GB）")
    t0 = time.perf_counter()
    model, tok = load(MODEL_PATH)
    cache, _ = wire_streaming(model, cap)
    print(
        f"準備完了（{time.perf_counter() - t0:.0f}秒）。"
        f"コーディングの依頼をどうぞ。'exit' か Ctrl-D で終了。\n"
    )

    messages = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            user = input("\n\033[1;36mあなた>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break
        if user.lower() in ("exit", "quit", ""):
            print("終了します。")
            break

        messages.append({"role": "user", "content": user})
        if len(messages) > 1 + MAX_HISTORY * 2:
            messages = [messages[0]] + messages[-MAX_HISTORY * 2 :]

        prompt = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        print("\033[1;32mElfMoon>\033[0m ", end="", flush=True)
        resp, t = "", time.perf_counter()
        n = 0
        _sampler = make_sampler(temp=TEMP)
        generator = stream_generate(
            model,
            tok,
            prompt,
            max_tokens=MAX_TOKENS,
            sampler=_sampler,
        )
        try:
            for out in generator:
                piece = out.text
                print(piece, end="", flush=True)
                resp += piece
                n += 1
        except Exception:
            pass

        dt = time.perf_counter() - t
        print(
            f"\n\033[2m（{n} tokens, {n / dt:.1f} tok/s, 命中率{cache.hit_rate * 100:.0f}%）\033[0m"
        )
        messages.append({"role": "assistant", "content": resp})


if __name__ == "__main__":
    main()

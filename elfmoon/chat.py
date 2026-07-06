"""ElfMoon 対話型チャットCLI（実運用の入口）。

モデルを一度だけロード＆ストリーミング化し、以降は対話ループで
何度でもプロンプトを投げられる。会話履歴を保持するのでコーディングの
やり取り（「さっきの関数にエラー処理を足して」等）も続けられる。

使い方:
    cd elfmoon
    python3 chat.py            # 常駐2800(既定)
    python3 chat.py 1200       # 省メモリ
"""
import sys
import time
from mlx_lm import load, stream_generate
from stream_model import wire_streaming, MODEL_PATH

SYSTEM = "You are an expert coding assistant. Prefer concise, correct code. Answer in the user's language."
MAX_TOKENS = 2048
MAX_HISTORY = 8          # 直近何往復を文脈に残すか（長すぎるとプレフィルが重い）


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 2800
    print(f"モデルをロード中...（常駐 {cap} experts ≈ {cap*2.65/1000:.1f}GB）")
    t0 = time.perf_counter()
    model, tok = load(MODEL_PATH)
    cache, _ = wire_streaming(model, cap)
    print(f"準備完了（{time.perf_counter()-t0:.0f}秒）。"
          f"コーディングの依頼をどうぞ。'exit' か Ctrl-D で終了。\n")

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
        # 履歴が長くなりすぎたら古い往復を落とす（systemは残す）
        if len(messages) > 1 + MAX_HISTORY * 2:
            messages = [messages[0]] + messages[-MAX_HISTORY * 2:]

        prompt = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)

        print("\033[1;32mElfMoon>\033[0m ", end="", flush=True)
        resp, t = "", time.perf_counter()
        n = 0
        for out in stream_generate(model, tok, prompt, max_tokens=MAX_TOKENS):
            piece = getattr(out, "text", str(out))
            print(piece, end="", flush=True)
            resp += piece
            n += 1
        dt = time.perf_counter() - t
        print(f"\n\033[2m（{n} tokens, {n/dt:.1f} tok/s, 命中率{cache.hit_rate*100:.0f}%）\033[0m")
        messages.append({"role": "assistant", "content": resp})


if __name__ == "__main__":
    main()

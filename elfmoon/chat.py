"""ElfMoon 対話型チャットCLI。

モデルを一度だけロード＆ストリーミング化し、以降は対話ループで
何度でもプロンプトを投げられる。会話履歴を保持する。

使い方:
    cd elfmoon
    python3 chat.py                  # 常駐 6144 (既定)
    python3 chat.py 1200             # 省メモリ
    python3 chat.py --no-think       # 思考プロセスを非表示
    python3 chat.py 1200 --no-think  # 組合せ
"""

import sys
import time
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from stream_model import wire_streaming, MODEL_PATH

SYSTEM = "You are an expert coding assistant. Write clean, correct, concise code."
MAX_TOKENS = 4096
MAX_HISTORY = 8
TEMP = 0.4


def _strip_think(text_iter, no_think):
    """Strip <think> block from stream if no_think is set."""
    if not no_think:
        yield from text_iter
        return
    skip = True
    buf = ""
    dots = 0
    for piece in text_iter:
        if skip:
            buf += piece
            idx = buf.find("</think>")
            if idx >= 0:
                skip = False
                if dots:
                    print("\b" * dots + " " * dots + "\b" * dots, end="", flush=True)
                    dots = 0
                after = buf[idx + 8 :]
                if after:
                    yield after
                buf = ""
            else:
                while len(buf) // 120 > dots:
                    dots += 1
                    print(".", end="", flush=True)
        else:
            yield piece
    # </think> が最後まで現れなかった場合、溜めた分を破棄せず出力する
    if skip and buf:
        if dots:
            print("\b" * dots + " " * dots + "\b" * dots, end="", flush=True)
        yield buf


def main():
    argv = sys.argv[1:]
    no_think = "--no-think" in argv
    cap_strs = [a for a in argv if a != "--no-think"]
    cap = int(cap_strs[0]) if cap_strs else 6144

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
        if user.lower() in ("exit", "quit"):
            print("終了します。")
            break
        if not user:
            continue

        messages.append({"role": "user", "content": user})
        if len(messages) > 1 + MAX_HISTORY * 2:
            messages = [messages[0]] + messages[-MAX_HISTORY * 2 :]

        prompt = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        print("\033[1;32mElfMoon>\033[0m ", end="", flush=True)
        resp, t, answer_t = "", time.perf_counter(), 0.0
        n = 0
        _sampler = make_sampler(temp=TEMP)
        generator = stream_generate(
            model,
            tok,
            prompt,
            max_tokens=MAX_TOKENS,
            sampler=_sampler,
        )

        def _texts():
            for out in generator:
                yield out.text

        try:
            for piece in _strip_think(_texts(), no_think):
                if answer_t == 0.0:
                    answer_t = time.perf_counter()
                print(piece, end="", flush=True)
                resp += piece
                n += 1
        except Exception as e:
            # 生成失敗を黙殺しない（途中で切れた応答を正常完了に見せない）
            print(f"\n\033[1;31m[エラー] 生成が中断されました: {e}\033[0m")

        elapsed = (
            (time.perf_counter() - answer_t) if answer_t else (time.perf_counter() - t)
        )
        print(
            f"\n\033[2m（{n} tokens, {n / elapsed:.1f} tok/s, 命中率{cache.hit_rate * 100:.0f}%）\033[0m"
        )
        messages.append({"role": "assistant", "content": resp})


if __name__ == "__main__":
    main()

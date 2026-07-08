"""§8.4 Gate-1: オーバーラップ検証

バックグラウンドスレッドで expert の load＋eval を回し続けながらデコードを実行し、
以下を計測:
(a) デコード t/s の劣化率 (バックグラウンド無しとの比較)
(b) 並行ロードのスループット (experts/sec)

教訓: timer 範囲に mx.eval（実体化）を含めること。
generate_step の yield は内部で mx.eval 済み＝タイマー内で直接使ってよい。
バックグラウンドスレッドの load＋eval は明示的に mx.eval をタイマー内で実行する。
"""

import argparse
import os
import sys
import time
import threading
import random

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from stream_model import MODEL_PATH, STORE_DIR, wire_streaming
from expert_store import ExpertStore
from resident_cache import ResidentCache

LONG_PROMPT = (
    "\n".join(
        f"func f{i}(_ x: Int) -> Int {{ return x * {i} + {i * i} }}" for i in range(40)
    )
    + "\n// Swiftで最大公約数gcd(_:_:)を書いて。コードのみ。"
)

N_LAYERS = 40
N_EXPERTS = 256
MAX_TOKENS = 80

# バックグラウンドスレッド用
_background_store = None
_stop_event = threading.Event()
_load_count = 0
_load_mx_dt = 0.0  # タイマー内で累積


def _load_random():
    """ランダムな expert を load＋eval し、mx.eval の所要時間を返す。

    注意: mx.load (lazy) + mx.eval (実体化) の両方をタイマー内に含める。
    """
    global _background_store
    layer = random.randrange(N_LAYERS)
    expert = random.randrange(N_EXPERTS)
    # ★ mx.load は lazy。mx.eval で実体化。両方をタイマー内に。
    t0 = time.perf_counter()
    data = _background_store.load(layer, expert)
    # data は dict of mx.array。各 array を eval する
    vals = list(data.values())
    mx.eval(vals)
    dt = time.perf_counter() - t0
    return dt


def _background_worker():
    """バックグラウンドで load＋eval を繰り返す。"""
    global _stop_event, _load_count, _load_mx_dt
    while not _stop_event.is_set():
        dt = _load_random()
        _load_count += 1
        _load_mx_dt += dt


def run_decode(background=False):
    """デコード実行。

    background=True の場合、バックグラウンドスレッドで load＋eval を回す。
    """
    global _background_store, _stop_event, _load_count, _load_mx_dt

    # リセット
    _load_count = 0
    _load_mx_dt = 0.0
    _stop_event.clear()

    # バックグラウンドスレッド開始
    bg_thread = None
    if background:
        _background_store = ExpertStore(STORE_DIR)
        bg_thread = threading.Thread(target=_background_worker, daemon=True)
        bg_thread.start()

    # モデルロード＋配線
    loaded = load(MODEL_PATH, lazy=True)
    model = loaded[0]
    tok = loaded[1]
    cache, store = wire_streaming(model, 6144)
    mx.clear_cache()

    # プロンプト
    prompt_tokens = tok.encode(LONG_PROMPT)
    prompt_arr = mx.array(prompt_tokens)

    # デコード
    gen_tokens = []
    t0 = time.perf_counter()
    for i, (token, _) in enumerate(
        generate_step(prompt_arr, model, max_tokens=MAX_TOKENS)
    ):
        if i == 0:
            prompt_time = time.perf_counter() - t0
            t_decode = time.perf_counter()
        gen_tokens.append(int(token))
    total_time = time.perf_counter() - t0
    decode_time = time.perf_counter() - t_decode

    n_gen = len(gen_tokens)
    decode_tps = n_gen / decode_time if decode_time > 0 else 0.0

    # バックグラウンド停止
    bg_load_count = 0
    bg_load_total_dt = 0.0
    if bg_thread:
        _stop_event.set()
        bg_thread.join(timeout=5)
        bg_load_count = _load_count
        bg_load_total_dt = _load_mx_dt

    # 出力を先にデコード（後片付けより前）
    output_preview = tok.decode(gen_tokens)[:200]

    # 後片付け
    del model, tok
    mx.clear_cache()

    return {
        "decode_tps": decode_tps,
        "decode_time_s": decode_time,
        "gen_tokens": n_gen,
        "prompt_tps": len(prompt_tokens) / prompt_time if prompt_time > 0 else 0,
        "hit_rate": cache.hit_rate,
        "peak_gb": mx.get_peak_memory() / 1e9,
        "bg_load_count": bg_load_count,
        "bg_load_total_dt": bg_load_total_dt,
        "bg_load_per_sec": bg_load_count / total_time if total_time > 0 else 0,
        "output_preview": output_preview,
    }


def main():
    parser = argparse.ArgumentParser(description="Gate-1: オーバーラップ検証")
    parser.add_argument(
        "--background", action="store_true", help="バックグラウンドロード有効"
    )
    args = parser.parse_args()

    label = "with BG" if args.background else "no BG"
    print(f"\n=== Gate-1: オーバーラップ検証 ({label}) ===\n")

    result = run_decode(background=args.background)

    print(f"\n--- 結果 ({label}) ---")
    print(f"  デコード:      {result['decode_tps']:.3f} t/s")
    print(f"  デコード時間:  {result['decode_time_s'] * 1000:.0f}ms")
    print(f"  生成トークン:  {result['gen_tokens']}")
    print(f"  ヒット率:      {result['hit_rate'] * 100:.1f}%")
    print(f"  ピークメモリ:  {result['peak_gb']:.2f}GB")
    if args.background:
        bg_rate = result["bg_load_per_sec"]
        print(f"  BGロード数:    {result['bg_load_count']}")
        print(f"  BGロード時間:  {result['bg_load_total_dt'] * 1000:.0f}ms")
        print(f"  BGスループット: {bg_rate:.0f} experts/sec")
        avg_expert_ms = (
            result["bg_load_total_dt"] / result["bg_load_count"] * 1000
            if result["bg_load_count"] > 0
            else 0
        )
        print(f"  BG平均/回:     {avg_expert_ms:.2f}ms")
    print(f"\n  出力先頭200字: {result['output_preview'][:80]}...")


if __name__ == "__main__":
    main()

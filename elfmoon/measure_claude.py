"""実 stream_generate 経路での warm A/B 計測（指示#09 に従い再現）"""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
from mlx_lm import generate, load
from stream_model import MODEL_PATH, wire_streaming

PROMPT = "Write Swift function gcd(_ a: Int, _ b: Int) -> Int. Code only."


def measure(label, ssc, cap=6144, n_tokens=80):
    os.environ["SSC"] = str(ssc)
    mx.clear_cache()
    model, tok = load(MODEL_PATH, lazy=True)
    cache, store = wire_streaming(model, cap)
    mx.eval()

    # Warm
    _ = generate(model, tok, prompt=PROMPT, max_tokens=8, verbose=False)
    mx.eval()
    mx.clear_cache()

    # Measure
    t0 = time.perf_counter()
    out = generate(model, tok, prompt=PROMPT, max_tokens=n_tokens, verbose=False)
    mx.eval()
    dt = time.perf_counter() - t0

    s = cache.stats()
    mem = mx.get_active_memory() / 1e9
    gen_tps = n_tokens / dt
    print(
        f"  [{label}] gen={gen_tps:.1f} t/s wall={dt:.2f}s hit={s['hit_rate'] * 100:.1f}% mem={mem:.2f}GB"
    )
    return gen_tps, s["hit_rate"], dt


if __name__ == "__main__":
    cap = 6144

    print("=== Warm A/B (stream_generate, 80 tokens) ===")
    c_tps, c_hr, c_dt = measure("C Baseline", 0)
    print()
    g_tps, g_hr, g_dt = measure("G gather+M2", 2000)
    print()
    print(f"  Baseline:      {c_tps:.1f} t/s  hit={c_hr * 100:.1f}%")
    print(f"  gather+M2:     {g_tps:.1f} t/s  hit={g_hr * 100:.1f}%")
    print(f"  G/C ratio:     {g_tps / c_tps:.2f}x")
    if g_tps < c_tps * 0.92:
        print(f"  >> STOP条件: 不合格 (0.28x vs 8%閾値)")
    else:
        print(f"  >> STOP条件: 合格")

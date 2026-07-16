"""Phase 2 end-to-end 計測: stream_model CLI 経由で3系統比較。

各系統で同一プロンプト、同一 max_tokens で generate 時間を計測。
prefill+decode 混在だが同一条件なので相対比較は有効。

実行: python3 elfmoon/measure_phase2.py [capacity=6144]
"""

import os
import re
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(_HERE, "stream_model.py")

PROMPT = "Write Swift function gcd(_ a: Int, _ b: Int) -> Int. Code only."
MAX_TOKENS = 80


def run_cli(cap, ssc=0):
    env = os.environ.copy()
    env["SSC"] = str(ssc)
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, CLI, str(cap)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    dt = time.perf_counter() - t0
    out = r.stdout + r.stderr

    # Parse generation speed
    m = re.search(r"Generation:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec", out)
    gen_tps = float(m.group(2)) if m else 0.0

    m = re.search(r"命中率=\s*([\d.]+)%", out)
    hr = float(m.group(1)) if m else 0.0

    m = re.search(r"時間=\s*([\d.]+)s", out)
    wall = float(m.group(1)) if m else 0.0

    # GSC mention
    has_gsc = "GlobalSlotCache" in out

    # Peak memory
    m = re.search(r"Peak memory:\s*([\d.]+)\s*GB", out)
    mem = float(m.group(1)) if m else 0.0

    return gen_tps, hr, wall, has_gsc, mem


def measure(label, cap, ssc):
    print(f"  [{label}] SSC={ssc}...", end=" ", flush=True)
    tps, hr, wall, has_gsc, mem = run_cli(cap, ssc)
    s = f"gen={tps:.1f}t/s hit={hr:.1f}% wall={wall:.1f}s mem={mem:.2f}GB"
    if has_gsc:
        s += " GSC=on"
    print(s)
    return {"tps": tps, "hit": hr, "wall": wall, "mem": mem}


if __name__ == "__main__":
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 6144

    print(f"Capacity={cap}, max_tokens={MAX_TOKENS}")
    print()

    # (C) Baseline: SSC=0
    c = measure("C", cap, 0)

    print()
    # (B) Real routing: SSC=2000
    b = measure("B", cap, 2000)

    # (A) All-resident: SSC = enough to cover routed experts
    # First, find how many unique (layer,expert) pairs are routed
    # For 40 layers × top-8 = 320, but with repetition it's ~400-500
    a_ssc = 4000  # conservative
    a = measure("A", cap, a_ssc)

    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print(
        f"  (C) Baseline (SSC=0):       {c['tps']:.1f} t/s  hit={c['hit']:.1f}%  wall={c['wall']:.1f}s"
    )
    print(
        f"  (B) Real routing (SSC=2000): {b['tps']:.1f} t/s  hit={b['hit']:.1f}%  wall={b['wall']:.1f}s"
    )
    print(
        f"  (A) All-resident (SSC={a_ssc}): {a['tps']:.1f} t/s  hit={a['hit']:.1f}%  wall={a['wall']:.1f}s"
    )
    print()
    print(f"  A/C speedup: {a['tps'] / c['tps']:.2f}x")
    print(f"  B/C speedup: {b['tps'] / c['tps']:.2f}x")
    print()
    print(f"  Memory: C={c['mem']:.2f}GB  B={b['mem']:.2f}GB  A={a['mem']:.2f}GB")

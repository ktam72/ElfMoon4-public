"""Phase 2 v2 計測: Claude #08 に従い gather_qmm + M2 を 35B で検証。

3系統:
  (C) Baseline: SSC=0, 既存 stack+_decode_moe
  (G) gather+M2: SSC=2000, gather_qmm + miss充填, コールドスタート
  (P) Pre-primed: SSC=2000, 事前に routed expert を全 priming → miss 0

さらに (C) と (G) の比較で eval 順序のみの効果を切り分け。

実行: python3 elfmoon/measure_phase2_v2.py
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import mlx.core as mx
from mlx_lm import generate, load
from stream_model import MODEL_PATH, wire_streaming
from slot_cache import GlobalSlotCache

PROMPT = "Write Swift function gcd(_ a: Int, _ b: Int) -> Int. Code only."
WARM_TOKENS = 40
MEASURE_TOKENS = 120


def _routed_set(cap):
    os.environ["SSC"] = "0"
    model, tok = load(MODEL_PATH, lazy=True)
    cache, store = wire_streaming(model, cap)
    mx.eval()

    routed = set()
    ids = tok.encode(PROMPT)
    all_ids = list(ids)
    for step in range(WARM_TOKENS):
        logits = model(mx.array([all_ids[-1]])[None])
        mx.eval(logits)
        nxt = int(mx.argmax(logits[:, -1, :], axis=-1))
        all_ids.append(nxt)
        layers = (
            getattr(model, "layers", None)
            or getattr(model.model, "layers", None)
            or getattr(model.language_model, "layers", None)
        )
        if layers:
            for l_idx, layer in enumerate(layers):
                mlp = getattr(layer, "mlp", None)
                if mlp and hasattr(mlp, "_cache") and hasattr(mlp, "_cache"):
                    rc = mlp._cache
                    if hasattr(rc, "_d"):
                        for key in list(rc._d.keys()):
                            routed.add(key)
        mx.clear_cache()
    mx.clear_cache()
    return routed, store, tok


def _measure_generate(model, tok, n_tokens):
    ids = tok.encode(PROMPT)
    all_ids = list(ids)
    t0 = time.perf_counter()
    for step in range(n_tokens):
        logits = model(mx.array([all_ids[-1]])[None])
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
        all_ids.append(int(nxt))
    mx.eval()
    dt = time.perf_counter() - t0
    return dt


def run_condition(cap, ssc, label, prime_set=None):
    print(f"\n[{label}] SSC={ssc} cap={cap}", end="", flush=True)

    os.environ["SSC"] = str(ssc)
    mx.clear_cache()

    n_layers = 40
    n_experts = 256

    model, tok = load(MODEL_PATH, lazy=True)
    cache, store = wire_streaming(model, cap)
    mx.eval()

    if prime_set is not None:
        # Override GSC with primed one
        gsc = GlobalSlotCache(ssc, store, n_layers=n_layers, n_experts=n_experts)
        for layer, eid in prime_set:
            gsc.get_slots(layer, [eid])
        layers = (
            getattr(model, "layers", None)
            or getattr(model.model, "layers", None)
            or getattr(model.language_model, "layers", None)
        )
        if layers:
            for l_idx, layer in enumerate(layers):
                mlp = getattr(layer, "mlp", None)
                if mlp and hasattr(mlp, "_gsc"):
                    mlp._gsc = gsc
        print(f" primed={len(gsc._lru)}", end="", flush=True)

    # Warm
    _measure_generate(model, tok, 8)
    mx.clear_cache()

    # Measure
    dt = _measure_generate(model, tok, MEASURE_TOKENS)
    tps = MEASURE_TOKENS / dt

    s = cache.stats()
    mem = mx.get_active_memory() / 1e9
    print(
        f" tps={tps:.1f} dt={dt:.2f}s hit={s['hit_rate'] * 100:.1f}% mem={mem:.2f}GB",
        flush=True,
    )
    return tps, s


if __name__ == "__main__":
    cap = 6144

    # First, collect routing profile
    print(f"Collecting routing profile ({WARM_TOKENS} tokens)...", flush=True)
    routed, store, tok = _routed_set(cap)
    print(f"  Unique (layer,expert) pairs: {len(routed)}")

    # (C) Baseline
    ctps, cs = run_condition(cap, 0, "C Baseline")

    # (G) gather+M2 cold start
    gtps, gs = run_condition(cap, 2000, "G gather+M2")

    # (P) Pre-primed + gather_qmm (miss = 0)
    ptps, ps = run_condition(cap, len(routed) * 2, "P Pre-primed", prime_set=routed)

    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"  (C) Baseline:          {ctps:.1f} t/s  {cs['hit_rate'] * 100:.1f}% hit")
    print(f"  (G) gather+M2 cold:    {gtps:.1f} t/s  {gs['hit_rate'] * 100:.1f}% hit")
    print(f"  (P) Pre-primed gather: {ptps:.1f} t/s  {ps['hit_rate'] * 100:.1f}% hit")
    print()
    print(f"  G/C: {gtps / ctps:.2f}x")
    print(f"  P/C: {ptps / ctps:.2f}x")
    print()
    if gtps <= ctps * 1.08:
        print("  STOP条件: gather+M2 cold が副次8%を超えず → 中止検討")
    else:
        print(f"  gather+M2 cold: {gtps / ctps:.2f}x (8%超) → 継続検討")
    if ptps > ctps:
        print(f"  Pre-primed gather: 最大可能 {ptps / ctps:.2f}x")

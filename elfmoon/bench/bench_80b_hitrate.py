"""80B: 容量別命中率＋デコード速度 実測（S2 項2）

プロトコル:
  - 容量を変えて別プロセス実行（OOM回避、ページキャッシュ共有）
  - warmup(1回目) → cache stats リセット → measured(2回目)
  - 報告: hit_rate, decode t/s, peak GB, 出力品質確認
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from expert_store import ExpertStore
from mlx_lm import load, stream_generate
from resident_cache import ResidentCache
from stream_model import MODEL_PATH, STORE_DIR

LONG_PROMPT = (
    "\n".join(f"func f{i}(_ x: Int) -> Int {{ return x * {i} + {i * i} }}" for i in range(40))
    + "\n// Swiftで最大公約数gcd(_:_:)を書いて。コードのみ。"
)


def _wire(model, store, cache, top_k):
    from stream_model import StreamingMoE

    layers = getattr(model, "layers", None) or model.model.layers
    for l, layer in enumerate(layers):
        mlp = layer.mlp
        n_exp = mlp.switch_mlp.gate_proj.weight.shape[0]
        gate = mlp.gate
        shared_exp = getattr(mlp, "shared_expert", None)
        shared_gate = getattr(mlp, "shared_expert_gate", None)
        layer.mlp = StreamingMoE(
            l,
            gate,
            n_exp,
            top_k,
            store,
            cache,
            shared_exp=shared_exp,
            shared_gate=shared_gate,
        )
    mx.clear_cache()


def _run_once(model, tok, store, cache, capacity):
    """Run one generate pass, return (out, decode_tok_per_s, stats, peak_gb)."""
    cache.hits = 0
    cache.misses = 0
    mx.reset_peak_memory()

    text_parts = []
    response = None
    for r in stream_generate(model, tok, prompt=LONG_PROMPT, max_tokens=80):
        text_parts.append(r.text)
        response = r

    out = "".join(text_parts)
    s = cache.stats()
    peak = mx.get_peak_memory() / 1e9
    decode_tps = response.generation_tps if response else 0.0
    return out, decode_tps, s, peak


def run_capacity(capacity, label):
    print(f"\n=== capacity={capacity} ({label}) ===")
    store = ExpertStore(STORE_DIR)
    cache = ResidentCache(capacity)
    model, tok = load(MODEL_PATH, lazy=True)
    top_k = 10  # 80B fixed
    _wire(model, store, cache, top_k)

    # warmup
    mx.clear_cache()
    out, warm_tps, warm_s, warm_peak = _run_once(model, tok, store, cache, capacity)
    print(f"  warmup:  {warm_tps:.1f} t/s  hit={warm_s['hit_rate'] * 100:.1f}%  peak={warm_peak:.2f}GB")

    # measured (second run, page cache hot)
    out, tps, s, peak = _run_once(model, tok, store, cache, capacity)
    print(f"  measured: {tps:.1f} t/s  hit={s['hit_rate'] * 100:.1f}%  peak={peak:.2f}GB")
    print(f"  出力先頭: {out[:150]}")

    del model, tok
    mx.clear_cache()

    return {
        "capacity": capacity,
        "decode_tps": tps,
        "hit_rate": s["hit_rate"],
        "hits": s["hits"],
        "misses": s["misses"],
        "peak_gb": peak,
    }


if __name__ == "__main__":
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 6144
    r = run_capacity(cap, "")
    print("\n--- 結果 ---")
    print(f"{'Capacity':>10s} {'decode t/s':>11s} {'hit%':>6s} {'peak GB':>8s} {'hits':>8s} {'misses':>8s}")
    print("-" * 52)
    print(
        f"{r['capacity']:>10d} {r['decode_tps']:>11.1f} {r['hit_rate'] * 100:>5.1f}%"
        f" {r['peak_gb']:>7.2f}GB {r['hits']:>8d} {r['misses']:>8d}"
    )

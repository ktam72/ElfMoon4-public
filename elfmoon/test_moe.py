"""モジュール①②③ の結合検証。
 (1) 正しさ: ハイブリッドMoEブロック == 全常駐の素朴実装
 (2) 命中率→tok/s: 容量を変えてキャッシュ挙動と速度を実測
"""
import time
import mlx.core as mx
from expert_store import ExpertStore, DEFAULT_DIM
from resident_cache import ResidentCache, plan_cache_experts
from moe_block import MoEBlock, reference_moe

DIM = DEFAULT_DIM
N_LAYERS = 4
N_EXPERTS = 64
TOP_K = 8
STORE_DIR = "spike/store"


def build():
    store = ExpertStore(STORE_DIR)
    store.generate_synthetic(N_LAYERS, N_EXPERTS, seed=1)
    mx.random.seed(2)
    gates = [mx.random.normal((N_EXPERTS, DIM)) for _ in range(N_LAYERS)]
    return store, gates


def test_correctness(store, gates):
    # 全expert常駐の参照 dict（layer 0）
    ref_experts = {e: store.load(0, e) for e in range(N_EXPERTS)}
    cache = ResidentCache(capacity=N_EXPERTS)
    blk = MoEBlock(0, gates[0], N_EXPERTS, TOP_K, store, cache)
    max_err = 0.0
    for _ in range(20):
        x = mx.random.normal((DIM,))
        y1 = blk(x)
        y2 = reference_moe(x, gates[0], ref_experts, TOP_K)
        max_err = max(max_err, float(mx.max(mx.abs(y1 - y2))))
    print(f"[正しさ] MoEBlock vs 参照  最大誤差={max_err:.2e}  "
          f"{'OK' if max_err < 1e-4 else 'NG'}")


def test_throughput(store, gates, capacity, n_tokens=200, skew=0.5):
    """capacity個常駐で n_tokens 生成し、命中率と tok/s を測る。
    skew: 入力に共通成分を混ぜてルーティングを偏らせる（現実のホット偏在を模擬）。"""
    cache = ResidentCache(capacity=capacity)
    blocks = [MoEBlock(l, gates[l], N_EXPERTS, TOP_K, store, cache)
              for l in range(N_LAYERS)]
    base = mx.random.normal((DIM,))
    # ウォームアップ（キャッシュを現実的な状態へ）
    for _ in range(20):
        h = skew * base + (1 - skew) * mx.random.normal((DIM,))
        for blk in blocks:
            h = h + blk(h) * 0.0 + mx.random.normal((DIM,)) * 0  # 経路のみ
        mx.eval(h)
    cache.hits = cache.misses = 0
    t = time.perf_counter()
    for _ in range(n_tokens):
        h = skew * base + (1 - skew) * mx.random.normal((DIM,))
        for blk in blocks:
            h = blk(h)
        mx.eval(h)
    dt = time.perf_counter() - t
    tps = n_tokens / dt
    s = cache.stats()
    print(f"[容量{capacity:3d}/{N_EXPERTS}] 命中率={s['hit_rate']*100:5.1f}%  "
          f"{tps:6.1f} tok/s  ({dt/n_tokens*1000:.1f} ms/token, {N_LAYERS}層)")


if __name__ == "__main__":
    store, gates = build()
    print(f"expert 1個 = {store.per_expert_bytes()/1e6:.2f} MB, "
          f"{N_LAYERS}層 x {N_EXPERTS}experts, top-{TOP_K}")
    test_correctness(store, gates)
    print("--- 命中率→tok/s（容量を変えて）---")
    for cap in (N_EXPERTS, 48, 32, 16, 8):
        test_throughput(store, gates, cap)

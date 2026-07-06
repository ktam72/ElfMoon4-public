"""モジュール④検証: ホットリスト・プライムで命中率→tok/sが上がるか。
現実のexpert偏在を「トークンをクラスタから生成」して模擬する。
"""
import time
import mlx.core as mx
from expert_store import ExpertStore, DEFAULT_DIM
from resident_cache import ResidentCache
from moe_block import MoEBlock
from hotlist import profile, build_hotlist, prime_cache

DIM = DEFAULT_DIM
N_LAYERS, N_EXPERTS, TOP_K = 8, 64, 8
N_CLUSTERS = 3          # 少数クラスタ＝強い偏在
NOISE = 0.1             # 小ノイズ＝ルーティング安定
WORKING_SET = N_LAYERS * TOP_K   # 1トークンが同時に要するexpert数(=64)
STORE_DIR = "spike/store2"


def clustered_stream(centers, n):
    for _ in range(n):
        c = centers[int(mx.random.randint(0, len(centers)))]
        yield c + mx.random.normal((DIM,)) * NOISE


def run(cache, store, gates, centers, n_tokens=200):
    blocks = [MoEBlock(l, gates[l], N_EXPERTS, TOP_K, store, cache)
              for l in range(N_LAYERS)]
    cache.hits = cache.misses = 0
    t = time.perf_counter()
    for x in clustered_stream(centers, n_tokens):
        h = x
        for blk in blocks:
            h = blk(h)
        mx.eval(h)
    dt = time.perf_counter() - t
    return cache.stats(), n_tokens / dt


def main():
    store = ExpertStore(STORE_DIR)
    store.generate_synthetic(N_LAYERS, N_EXPERTS, seed=3)
    mx.random.seed(4)
    gates = [mx.random.normal((N_EXPERTS, DIM)) for _ in range(N_LAYERS)]
    centers = [mx.random.normal((DIM,)) for _ in range(N_CLUSTERS)]

    print(f"expert 1個={store.per_expert_bytes()/1e6:.2f}MB, "
          f"{N_LAYERS}層x{N_EXPERTS}exp, top{TOP_K}, クラスタ{N_CLUSTERS}, ノイズ{NOISE}")

    # キャリブレーションでホットリスト作成
    calib = list(clustered_stream(centers, 100))
    hot = build_hotlist(profile(gates, TOP_K, calib))
    print(f"作業集合(同時必要)={WORKING_SET}, 実際に使われた distinct expert={len(hot)}/{N_LAYERS*N_EXPERTS}")

    # 容量スイープ（命中率が予算でどう動くか）
    print("--- 容量→命中率（プライム無）---")
    for cap in (WORKING_SET, 96, 128, 192, len(hot) + 8):
        c = ResidentCache(cap)
        s, t = run(c, store, gates, centers)
        print(f"  常駐{cap:3d}: 命中率={s['hit_rate']*100:5.1f}%  {t:6.1f} tok/s")

    # タイト予算でプライム有無を比較
    cap = 96
    print(f"--- 常駐{cap}でプライム有無 ---")
    ca = ResidentCache(cap); sa, ta = run(ca, store, gates, centers)
    cb = ResidentCache(cap); prime_cache(cb, store, hot, cap); sb, tb = run(cb, store, gates, centers)
    print(f"  プライム無: 命中率={sa['hit_rate']*100:5.1f}%  {ta:6.1f} tok/s")
    print(f"  プライム有: 命中率={sb['hit_rate']*100:5.1f}%  {tb:6.1f} tok/s")


if __name__ == "__main__":
    main()

"""§7.3-1 マイクロテスト: mx.load の並列性確認

同一層の8ファイルを逐次 vs ThreadPoolExecutor(8) で mx.load＋mx.eval し、
所要時間を比較する。並列が逐次より明確に速ければ C-1a に進む。
"""

import concurrent.futures
import os
import time

import mlx.core as mx

STORE_DIR = os.path.join(os.path.dirname(__file__), "spike/real_store")


def _path(layer, expert):
    return os.path.join(STORE_DIR, f"l{layer}_e{expert}.safetensors")


def load_one(path):
    return mx.load(path)


def bench(parallel, n_warmup=3, n_measured=10):
    """逐次(parallel=False) または 並列(parallel=True) で 8 expert をロード+eval。

    Returns:
        float: 平均所要時間(秒)
    """
    layer = 0
    experts = list(range(8))
    paths = [_path(layer, e) for e in experts]

    label = "並列(8)" if parallel else "逐次"

    for epoch in range(n_warmup + n_measured):
        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                t0 = time.perf_counter()
                results = list(pool.map(load_one, paths))
                dt = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            results = [load_one(p) for p in paths]
            dt = time.perf_counter() - t0

        mx.eval(results)

        if epoch < n_warmup:
            print(f"  [{label}] warmup {epoch + 1}: {dt * 1000:.2f}ms")
        else:
            print(f"  [{label}] measured {epoch - n_warmup + 1}: {dt * 1000:.2f}ms")

    return dt  # last epoch time


def main():
    print("=== §7.3-1 mx.load 並列性マイクロテスト ===\n")

    # warmup page cache
    print("[warmup] ページキャッシュ準備...")
    for e in range(256):
        mx.load(_path(0, e))
    print(f"  done ({256} files)\n")

    print("--- 逐次ロード (8 files) ---")
    seq_times = []
    for i in range(5):
        t0 = time.perf_counter()
        res = [load_one(_path(0, e)) for e in range(8)]
        mx.eval(res)
        dt = time.perf_counter() - t0
        seq_times.append(dt)
        print(f"  run {i + 1}: {dt * 1000:.2f}ms")
    avg_seq = sum(seq_times) / len(seq_times)
    print(f"  平均: {avg_seq * 1000:.2f}ms\n")

    print("--- 並列ロード (ThreadPoolExecutor 8, 8 files) ---")
    par_times = []
    for i in range(5):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            t0 = time.perf_counter()
            res = list(pool.map(lambda e: load_one(_path(0, e)), range(8)))
            dt = time.perf_counter() - t0
        mx.eval(res)
        par_times.append(dt)
        print(f"  run {i + 1}: {dt * 1000:.2f}ms")
    avg_par = sum(par_times) / len(par_times)
    print(f"  平均: {avg_par * 1000:.2f}ms\n")

    print("=== 結果 ===")
    print(f"  逐次: {avg_seq * 1000:.2f}ms")
    print(f"  並列: {avg_par * 1000:.2f}ms")
    print(f"  高速化率: {avg_seq / avg_par:.2f}x")
    if avg_par < avg_seq * 0.8:
        print("  ✅ 並列が明確に有効。C-1a 本実装へ進む。")
    elif avg_par < avg_seq * 0.95:
        print("  ⚠️  軽度の改善。C-1a は実施するが効果を注視。")
    else:
        print("  ❌ 並列が空振り。代替案（層単位連結バイナリ）を検討。")

    # mx.load のコールド状態も確認
    print("\n--- コールド参照 (ページキャッシュ未温の別層) ---")
    cold_times = []
    for run in range(3):
        layer = 20 + run
        t0 = time.perf_counter()
        res = [load_one(_path(layer, e)) for e in range(8)]
        mx.eval(res)
        dt = time.perf_counter() - t0
        cold_times.append(dt)
        print(f"  layer {layer} run {run + 1}: {dt * 1000:.2f}ms")
    avg_cold = sum(cold_times) / len(cold_times)
    print(f"  平均(コールド): {avg_cold * 1000:.2f}ms\n")

    # 並列コールド
    print("--- 並列コールド (別層 8 files) ---")
    cold_par_times = []
    for run in range(3):
        layer = 30 + run
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            t0 = time.perf_counter()
            res = list(pool.map(lambda e: load_one(_path(layer, e)), range(8)))
            dt = time.perf_counter() - t0
        mx.eval(res)
        cold_par_times.append(dt)
        print(f"  layer {layer} run {run + 1}: {dt * 1000:.2f}ms")
    avg_cold_par = sum(cold_par_times) / len(cold_par_times)
    print(f"  平均(並列コールド): {avg_cold_par * 1000:.2f}ms\n")

    print("=== 総評 ===")
    print(f"  ウォーム 逐次/並列: {avg_seq * 1000:.1f}ms / {avg_par * 1000:.1f}ms ({avg_seq / avg_par:.2f}x)")
    print(
        f"  コールド 逐次/並列: {avg_cold * 1000:.1f}ms / {avg_cold_par * 1000:.1f}ms ({avg_cold / avg_cold_par:.2f}x)"
    )


if __name__ == "__main__":
    main()

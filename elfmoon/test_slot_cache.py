"""SlotResidentCache 退避テスト: 退避発生時の重み一致を検証。

小容量（per_layer=6, capacity=12, N_LAYERS=2）で退避を強制発生させ、
キャッシュ経由で読み出した全 expert の重みを ExpertStore 直読みと突き合わせる。

実行: python3 elfmoon/test_slot_cache.py
"""

import os
import sys
import tempfile

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_store import ExpertStore
from slot_cache import SlotResidentCache

N_LAYERS = 2
N_EXPERTS = 16
PER_LAYER = 6  # 退避が頻発する小容量
CAPACITY = N_LAYERS * PER_LAYER  # 12
DIM = 2048
INTER = 512


def test_eviction_weight_integrity():
    """退避後もキャッシュから取得した重みが直読みと一致するか"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ExpertStore(tmpdir, dim=DIM, inter=INTER)
        store.generate_synthetic(N_LAYERS, N_EXPERTS, seed=42)
        cache = SlotResidentCache(CAPACITY, store, n_layers=N_LAYERS, min_per_layer=PER_LAYER)

        # 全 expert をアクセス → 退避を強制（per_layer=6 < N_EXPERTS=16）
        for layer in range(N_LAYERS):
            for expert in range(N_EXPERTS):
                cache.get_slots(layer, [expert])

        # 退避が発生した状態で、全 expert の重み一致を確認
        max_err = 0.0
        for layer in range(N_LAYERS):
            for expert in range(N_EXPERTS):
                ref = store.load(layer, expert)
                slots = cache.get_slots(layer, [expert])
                slot = slots[0]
                cached = {
                    name: getattr(cache, name.replace(".", "_"))[layer][slot]
                    for name in [
                        "gate.wq",
                        "gate.s",
                        "gate.b",
                        "up.wq",
                        "up.s",
                        "up.b",
                        "down.wq",
                        "down.s",
                        "down.b",
                    ]
                }
                for key in ref:
                    err = float(mx.max(mx.abs(ref[key].astype(mx.float32) - cached[key].astype(mx.float32))))
                    # bf16 保存との精度差を許容（合成データは fp32 だがバッファは bf16）
                    if err > 0.02:
                        print(f"  layer={layer} expert={expert} key={key} 誤差={err:.2e}")
                    max_err = max(max_err, err)

        ok = max_err < 0.02  # bf16 精度範囲を許容
        print(f"  最大誤差={max_err:.2e}  {'OK' if ok else 'NG'}")
        assert ok, f"重み誤差超過: {max_err:.2e}"


def test_lru_order():
    """LRU 順序と退避が正しく機能するか"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ExpertStore(tmpdir, dim=DIM, inter=INTER)
        store.generate_synthetic(N_LAYERS, N_EXPERTS, seed=42)
        cache = SlotResidentCache(CAPACITY, store, n_layers=N_LAYERS, min_per_layer=PER_LAYER)

        layer = 0
        # PER_LAYER 個ロード → 全スロット占有
        for e in range(PER_LAYER):
            cache.get_slots(layer, [e])

        assert len(cache._lru) == PER_LAYER

        # 次の expert をロード → 退避発生
        cache.get_slots(layer, [PER_LAYER])

        # LRU 最古 (expert 0) が消えているはず
        assert (layer, 0) not in cache._lru, "expert 0 が退避されていない"

        # expert 0 を再ロード → 成功
        slots = cache.get_slots(layer, [0])
        assert len(slots) == 1

        # 重み一致確認（uint32 は完全一致）
        ref = store.load(layer, 0)
        slot = slots[0]
        err = float(mx.max(mx.abs(ref["gate.wq"].astype(mx.float32) - cache.gate_wq[layer][slot].astype(mx.float32))))
        assert err == 0, f"再ロード後の gate.wq 誤差: {err:.2e}"
        # bf16 のスケール/バイアスも許容誤差内
        for key in ["gate.s", "gate.b"]:
            err = float(
                mx.max(
                    mx.abs(
                        ref[key].astype(mx.float32)
                        - getattr(cache, key.replace(".", "_"))[layer][slot].astype(mx.float32)
                    )
                )
            )
            assert err < 0.02, f"再ロード後の {key} 誤差: {err:.2e}"
        print("  OK")


def test_in_use_guard():
    """in_use な expert が退避されないことを確認"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ExpertStore(tmpdir, dim=DIM, inter=INTER)
        store.generate_synthetic(N_LAYERS, N_EXPERTS, seed=42)
        cache = SlotResidentCache(CAPACITY, store, n_layers=N_LAYERS, min_per_layer=PER_LAYER)

        layer = 0
        # 全スロット占有
        for e in range(PER_LAYER):
            cache.get_slots(layer, [e])

        # PER_LAYER 番目と一緒に expert 0 も in_use として要求
        slots = cache.get_slots(layer, [0, PER_LAYER])

        # expert 0 は in_use なので退避されない
        assert (layer, 0) in cache._lru, "in_use の expert 0 が退避された"

        # 重み一致確認
        ref = store.load(layer, 0)
        slot_of_0_in_result = slots[0]
        err = float(
            mx.max(
                mx.abs(ref["gate.wq"].astype(mx.float32) - cache.gate_wq[layer][slot_of_0_in_result].astype(mx.float32))
            )
        )
        assert err == 0, f"in_use 保護後の gate.wq 誤差: {err:.2e}"
        for key in ["gate.s", "gate.b"]:
            err = float(
                mx.max(
                    mx.abs(
                        ref[key].astype(mx.float32)
                        - getattr(cache, key.replace(".", "_"))[layer][slot_of_0_in_result].astype(mx.float32)
                    )
                )
            )
            assert err < 0.02, f"in_use 保護後の {key} 誤差: {err:.2e}"
        print("  OK")


def test_prime_eviction():
    """prime() が退避を正しく扱うか（タプルクラッシュしないか）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ExpertStore(tmpdir, dim=DIM, inter=INTER)
        store.generate_synthetic(N_LAYERS, N_EXPERTS, seed=42)
        cache = SlotResidentCache(CAPACITY, store, n_layers=N_LAYERS, min_per_layer=PER_LAYER)

        layer = 0
        # 全スロット占有
        for e in range(PER_LAYER):
            cache.get_slots(layer, [e])

        # prime() で退避発生（旧実装ではタプル代入でクラッシュ）
        cache.prime(layer, PER_LAYER)

        key = (layer, PER_LAYER)
        assert key in cache._lru, "prime した expert が LRU にない"

        # 重み一致確認
        ref = store.load(layer, PER_LAYER)
        slot = cache._lru[key]
        err = float(mx.max(mx.abs(ref["gate.wq"].astype(mx.float32) - cache.gate_wq[layer][slot].astype(mx.float32))))
        assert err == 0, f"prime 後の gate.wq 誤差: {err:.2e}"
        for key in ["gate.s", "gate.b"]:
            err = float(
                mx.max(
                    mx.abs(
                        ref[key].astype(mx.float32)
                        - getattr(cache, key.replace(".", "_"))[layer][slot].astype(mx.float32)
                    )
                )
            )
            assert err < 0.02, f"prime 後の {key} 誤差: {err:.2e}"
        print("  OK")


def test_evict_runtime_error():
    """in_use が per_layer を超える場合に RuntimeError が発生するか"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ExpertStore(tmpdir, dim=DIM, inter=INTER)
        store.generate_synthetic(N_LAYERS, N_EXPERTS, seed=42)
        cache = SlotResidentCache(CAPACITY, store, n_layers=N_LAYERS, min_per_layer=PER_LAYER)

        layer = 0
        # 全スロット占有
        for e in range(PER_LAYER):
            cache.get_slots(layer, [e])

        # per_layer=6 に対して 7 つ要求（6 in_use + 1 miss = 全部 in_use）
        try:
            cache.get_slots(layer, list(range(PER_LAYER + 1)))
            assert False, "RuntimeError が発生すべき"
        except RuntimeError as e:
            assert "退避可能スロットなし" in str(e)
            print(f"  OK (caught: {e})")


if __name__ == "__main__":
    print("=== test_eviction_weight_integrity ===")
    test_eviction_weight_integrity()
    print("=== test_lru_order ===")
    test_lru_order()
    print("=== test_in_use_guard ===")
    test_in_use_guard()
    print("=== test_prime_eviction ===")
    test_prime_eviction()
    print("=== test_evict_runtime_error ===")
    test_evict_runtime_error()
    print("All tests passed.")

"""GlobalSlotCache 検証: 命中率一致 + 重み完全性 + slot_map正確性。

- ResidentCache (dict LRU) と同一アクセスパターンで hit_rate 一致
- slot_map から gather した重みが store 直読みと一致
- gather_qmm 用 3D バッファの get_gather_bufs() が全件返る

実行: python3 elfmoon/test_global_slot_cache.py
"""

import os
import sys
import tempfile

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_store import ExpertStore
from resident_cache import ResidentCache
from slot_cache import GlobalSlotCache, _SENTINEL

N_LAYERS = 4
N_EXPERTS = 64
CAPACITY = 20  # ~31% resident → ~70% hit rate expected
DIM = 2048
INTER = 512


def _access_pattern():
    """Simulates decode: top-10 experts across layers with temporal locality."""
    for layer in range(N_LAYERS):
        base = (layer * 3) % N_EXPERTS
        yield from [(layer, (base + i) % N_EXPERTS) for i in range(10)]


def test_global_slot_cache_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ExpertStore(tmpdir, dim=DIM, inter=INTER)
        store.generate_synthetic(N_LAYERS, N_EXPERTS, seed=42)
        gsc = GlobalSlotCache(CAPACITY, store, n_layers=N_LAYERS, n_experts=N_EXPERTS)

        # Fill cache gradually, measure slot_map integrity
        pattern = list(_access_pattern())

        # 4 pass (warm-up), then stats
        gsc.hits = 0
        gsc.misses = 0
        for _ in range(3):
            for layer, eid in pattern:
                gsc.get_slots(layer, [eid])

        hr = gsc.hit_rate
        print(f"  3-pass hit rate: {hr:.1%} ({gsc.hits}/{gsc.hits + gsc.misses})")
        print(f"  resident: {len(gsc._lru)}/{CAPACITY}")

        # slot_map integrity: every resident expert maps to a valid slot
        for (layer, eid), slot in gsc._lru.items():
            gpu_val = int(gsc.slot_map[layer, eid])
            assert gpu_val == slot, (
                f"slot_map[{layer},{eid}] = {gpu_val}, expected {slot}"
            )
            assert 0 <= slot < CAPACITY, f"slot {slot} out of range"
        print(f"  slot_map integrity: OK ({len(gsc._lru)} entries)")

        # Non-resident experts → SENTINEL
        for layer in range(N_LAYERS):
            for eid in range(N_EXPERTS):
                if (layer, eid) not in gsc._lru:
                    gpu_val = int(gsc.slot_map[layer, eid])
                    assert gpu_val == _SENTINEL, (
                        f"slot_map[{layer},{eid}] = {gpu_val}, expected SENTINEL"
                    )

        # Weight integrity: gather from buffer matches store
        max_err = 0.0
        for (layer, eid), slot in gsc._lru.items():
            ref = store.load(layer, eid)
            cached = {
                "gate.wq": gsc.gate_wq[slot],
                "gate.s": gsc.gate_s[slot],
                "gate.b": gsc.gate_b[slot],
                "up.wq": gsc.up_wq[slot],
                "up.s": gsc.up_s[slot],
                "up.b": gsc.up_b[slot],
                "down.wq": gsc.down_wq[slot],
                "down.s": gsc.down_s[slot],
                "down.b": gsc.down_b[slot],
            }
            for key in ref:
                err = float(
                    mx.max(
                        mx.abs(
                            ref[key].astype(mx.float32) - cached[key].astype(mx.float32)
                        )
                    )
                )
                if err > 0.02:
                    print(f"  layer={layer} expert={eid} key={key} err={err:.2e}")
                max_err = max(max_err, err)
        ok = max_err < 0.02
        print(f"  weight max_err={max_err:.2e}  {'OK' if ok else 'NG'}")
        assert ok, f"weight error exceeded: {max_err:.2e}"

        # get_gather_bufs returns all 9 arrays with correct dims
        bufs = gsc.get_gather_bufs()
        assert len(bufs) == 9, f"get_gather_bufs returned {len(bufs)} arrays"
        names = [
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
        for name, buf in zip(names, bufs):
            expected_shape = (
                (CAPACITY, INTER, DIM // 8)
                if "wq" in name and "down" not in name
                else (CAPACITY, DIM, INTER // 8)
                if name == "down.wq"
                else (CAPACITY, INTER, DIM // 64)
                if "gate" in name and "wq" not in name
                else (CAPACITY, INTER, DIM // 64)
                if "up" in name and "wq" not in name
                else (CAPACITY, DIM, INTER // 64)
                if "down" in name and "wq" not in name
                else (CAPACITY,)
            )
            expected_dtype = mx.uint32 if "wq" in name else mx.float32
            assert buf.shape[0] == CAPACITY, (
                f"{name} shape[0]={buf.shape[0]} != {CAPACITY}"
            )
            assert buf.dtype == expected_dtype, (
                f"{name} dtype={buf.dtype} != {expected_dtype}"
            )
        print(f"  get_gather_bufs shapes: OK")


def test_global_vs_dict_hit_rate():
    """GlobalSlotCache hit rate must match ResidentCache under same pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ExpertStore(tmpdir, dim=DIM, inter=INTER)
        store.generate_synthetic(N_LAYERS, N_EXPERTS, seed=42)

        gsc = GlobalSlotCache(CAPACITY, store, n_layers=N_LAYERS, n_experts=N_EXPERTS)
        rc = ResidentCache(CAPACITY)

        pattern = list(_access_pattern())
        for _ in range(5):
            for layer, eid in pattern:
                gsc.get_slots(layer, [eid])
                rc.get((layer, eid), lambda: store.load(layer, eid))

        diff = abs(gsc.hit_rate - rc.hit_rate)
        ok = diff < 0.01
        print(f"  GlobalSlotCache hit_rate={gsc.hit_rate:.4f}")
        print(f"  ResidentCache   hit_rate={rc.hit_rate:.4f}")
        print(f"  diff={diff:.4f}  {'OK' if ok else 'NG'}")
        assert ok, f"hit rate mismatch: {gsc.hit_rate:.4f} vs {rc.hit_rate:.4f}"


def test_weight_via_gather():
    """Verify gather_via_qmm: slot_map→gather→correct weight."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ExpertStore(tmpdir, dim=DIM, inter=INTER)
        store.generate_synthetic(N_LAYERS, N_EXPERTS, seed=42)
        gsc = GlobalSlotCache(CAPACITY, store, n_layers=N_LAYERS, n_experts=N_EXPERTS)

        # Load all experts into cache
        for layer in range(N_LAYERS):
            for eid in range(N_EXPERTS):
                gsc.get_slots(layer, [eid])

        # For each layer+expert, verify gather via slot_map gives correct weights
        layer = 0
        eids = [0, 3, 7, 15, 31]
        slots = gsc.get_slots(layer, eids)
        for eid, slot in zip(eids, slots):
            ref = store.load(layer, eid)
            gathered = gsc.gate_wq[slot]
            err = float(
                mx.max(
                    mx.abs(
                        ref["gate.wq"].astype(mx.float32) - gathered.astype(mx.float32)
                    )
                )
            )
            assert err == 0, f"gather_via_qmm expert {eid} slot {slot} err={err:.2e}"
        print(f"  gather_via_slot_map: OK ({len(eids)} experts)")


if __name__ == "__main__":
    print("=== test_global_slot_cache_basic ===")
    test_global_slot_cache_basic()
    print("=== test_global_vs_dict_hit_rate ===")
    test_global_vs_dict_hit_rate()
    print("=== test_weight_via_gather ===")
    test_weight_via_gather()
    print("All GlobalSlotCache tests passed.")

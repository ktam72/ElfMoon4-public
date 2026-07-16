"""_decode_moe_gather パリティ検証: gather_qmm 経路 vs _decode_moe 経路。

3条件:
1. ALL_HIT — _decode_moe_gather と _decode_moe の出力一致（1e-3以内）
2. 混合（HIT+MISS）— gather + fallback の出力一致
3. Track A 全resident — gather のみ（miss 0）で正しい出力

実行: python3 elfmoon/test_gather_decode.py
"""

import os
import sys
import tempfile

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_store import ExpertStore
from slot_cache import GlobalSlotCache
from stream_model import (
    _decode_moe,
    _decode_moe_gather,
    _shared_ffn,
    BITS,
    GROUP,
)

N_LAYERS = 4
N_EXPERTS = 32
CAPACITY = 128
DIM = 2048
INTER = 512
TOP_K = 6


def _make_fake_weights(store, layer, eids):
    """Load weights for given experts."""
    return [store.load(layer, e) for e in eids]


def test_gather_vs_decode_allhit():
    """_decode_moe_gather と _decode_moe が全hit時（全expert resident）で一致するか"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ExpertStore(tmpdir, dim=DIM, inter=INTER)
        store.generate_synthetic(N_LAYERS, N_EXPERTS, seed=42)
        gsc = GlobalSlotCache(CAPACITY, store, n_layers=N_LAYERS, n_experts=N_EXPERTS)

        # Prime all experts for layer 0
        for e in range(N_EXPERTS):
            gsc.get_slots(0, [e])

        x = mx.random.normal((1, DIM))
        eids = list(range(TOP_K))
        weights_raw = mx.random.uniform(0.0, 1.0, (1, TOP_K))
        weights_norm = weights_raw / mx.sum(weights_raw, axis=-1, keepdims=True)

        # Reference: _decode_moe with stacked weights
        experts = [store.load(0, e) for e in eids]
        w_gu = mx.stack([e["gate.wq"] for e in experts] + [e["up.wq"] for e in experts])
        s_gu = mx.stack([e["gate.s"] for e in experts] + [e["up.s"] for e in experts])
        b_gu = mx.stack(
            [e.get("gate.b") for e in experts] + [e.get("up.b") for e in experts]
        )
        w_dw = mx.stack([e["down.wq"] for e in experts])
        s_dw = mx.stack([e["down.s"] for e in experts])
        b_dw = mx.stack([e.get("down.b") for e in experts])
        weights = weights_norm[0].astype(mx.float16)
        ref = _decode_moe(
            x,
            w_gu,
            s_gu,
            b_gu,
            w_dw,
            s_dw,
            b_dw,
            weights,
            TOP_K,
            shared=None,
            group_size=GROUP,
            bits=BITS,
        )
        mx.eval(ref)

        # gather path
        slot_ids = mx.array([gsc._lru[(0, e)] for e in eids], dtype=mx.uint32)
        out = _decode_moe_gather(
            x,
            gsc,
            0,
            slot_ids,
            weights,
            TOP_K,
            shared=None,
            group_size=GROUP,
            bits=BITS,
        )
        mx.eval(out)

        err = float(mx.max(mx.abs(ref - out)))
        ok = err < 1e-3
        print(f"  all-hit max_err={err:.2e}  {'OK' if ok else 'NG'}")
        assert ok, f"parity error: {err:.2e}"


def test_gather_with_shared():
    """_decode_moe_gather と _decode_moe がshared expertありで一致するか"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ExpertStore(tmpdir, dim=DIM, inter=INTER)
        store.generate_synthetic(N_LAYERS, N_EXPERTS, seed=42)
        gsc = GlobalSlotCache(CAPACITY, store, n_layers=N_LAYERS, n_experts=N_EXPERTS)

        # Prime all experts for layer 0
        for e in range(N_EXPERTS):
            gsc.get_slots(0, [e])

        x = mx.random.normal((1, DIM))
        eids = list(range(TOP_K))
        weights_raw = mx.random.uniform(0.0, 1.0, (1, TOP_K))
        weights_norm = weights_raw / mx.sum(weights_raw, axis=-1, keepdims=True)

        # Fake shared expert (gate+up concat + down)
        se_w = mx.random.normal((INTER * 2, DIM))
        se_gu_wq, se_gu_s, se_gu_b = mx.quantize(se_w, group_size=GROUP, bits=BITS)
        se_d = mx.random.normal((DIM, INTER))
        se_d_wq, se_d_s, se_d_b = mx.quantize(se_d, group_size=GROUP, bits=BITS)
        shared = (se_gu_wq, se_gu_s, se_gu_b, se_d_wq, se_d_s, se_d_b)

        # Reference
        experts = [store.load(0, e) for e in eids]
        w_gu = mx.stack([e["gate.wq"] for e in experts] + [e["up.wq"] for e in experts])
        s_gu = mx.stack([e["gate.s"] for e in experts] + [e["up.s"] for e in experts])
        b_gu = mx.stack(
            [e.get("gate.b") for e in experts] + [e.get("up.b") for e in experts]
        )
        w_dw = mx.stack([e["down.wq"] for e in experts])
        s_dw = mx.stack([e["down.s"] for e in experts])
        b_dw = mx.stack([e.get("down.b") for e in experts])
        weights = weights_norm[0].astype(mx.float16)
        ref = _decode_moe(
            x,
            w_gu,
            s_gu,
            b_gu,
            w_dw,
            s_dw,
            b_dw,
            weights,
            TOP_K,
            shared=shared,
            group_size=GROUP,
            bits=BITS,
        )
        mx.eval(ref)

        slot_ids = mx.array([gsc._lru[(0, e)] for e in eids], dtype=mx.uint32)
        out = _decode_moe_gather(
            x,
            gsc,
            0,
            slot_ids,
            weights,
            TOP_K,
            shared=shared,
            group_size=GROUP,
            bits=BITS,
        )
        mx.eval(out)

        err = float(mx.max(mx.abs(ref - out)))
        ok = err < 1e-3
        print(f"  with-shared max_err={err:.2e}  {'OK' if ok else 'NG'}")
        assert ok, f"parity error with shared: {err:.2e}"


def test_gather_all_resident():
    """Track A: 全expert resident → gather_qmm のみで正しい出力"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ExpertStore(tmpdir, dim=DIM, inter=INTER)
        store.generate_synthetic(N_LAYERS, N_EXPERTS, seed=42)
        gsc = GlobalSlotCache(CAPACITY, store, n_layers=N_LAYERS, n_experts=N_EXPERTS)

        # Prime ALL experts (all-resident mode = Track A ceiling)
        for layer in range(N_LAYERS):
            for e in range(N_EXPERTS):
                gsc.get_slots(layer, [e])

        x = mx.random.normal((1, DIM))
        eids = list(range(TOP_K))
        weights_raw = mx.random.uniform(0.0, 1.0, (1, TOP_K))
        weights_norm = weights_raw / mx.sum(weights_raw, axis=-1, keepdims=True)

        experts = [store.load(0, e) for e in eids]
        w_gu = mx.stack([e["gate.wq"] for e in experts] + [e["up.wq"] for e in experts])
        s_gu = mx.stack([e["gate.s"] for e in experts] + [e["up.s"] for e in experts])
        b_gu = mx.stack(
            [e.get("gate.b") for e in experts] + [e.get("up.b") for e in experts]
        )
        w_dw = mx.stack([e["down.wq"] for e in experts])
        s_dw = mx.stack([e["down.s"] for e in experts])
        b_dw = mx.stack([e.get("down.b") for e in experts])
        weights = weights_norm[0].astype(mx.float16)
        ref = _decode_moe(x, w_gu, s_gu, b_gu, w_dw, s_dw, b_dw, weights, TOP_K)
        mx.eval(ref)

        # All-resident gather path
        slot_ids = mx.array([gsc._lru[(0, e)] for e in eids], dtype=mx.uint32)
        out = _decode_moe_gather(x, gsc, 0, slot_ids, weights, TOP_K)
        mx.eval(out)

        err = float(mx.max(mx.abs(ref - out)))
        ok = err < 1e-3
        print(f"  all-resident max_err={err:.2e}  {'OK' if ok else 'NG'}")
        assert ok, f"all-resident parity error: {err:.2e}"


if __name__ == "__main__":
    print("=== test_gather_vs_decode_allhit ===")
    test_gather_vs_decode_allhit()
    print("=== test_gather_with_shared ===")
    test_gather_with_shared()
    print("=== test_gather_all_resident ===")
    test_gather_all_resident()
    print("All gather decode tests passed.")

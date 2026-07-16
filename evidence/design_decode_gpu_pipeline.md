# Design Document: 80B Decode GPU-only Pipeline

- Date: 2026-07-17 (v3)
- Target: Qwen3-Next-80B-A3B, decode MoE path optimization
- Inputs: `poc_decode_optimization_report.md`, `80b_decode_profile.md` (v4), `directive_deepseek_04.md`-`06.md`

---

## 1. Problem Restatement

Current decode pipeline bottleneck: **per-layer CPU-GPU round-trip (`mx.eval(idx,w)` + `tolist`)** × 48 layers. Each round-trip forces GPU to idle while CPU does routing → dict lookup ×10 → mx.stack ×4 → next layer. Total 118.4ms/step (8.4 t/s).

Pre-buffered PoC (fixed experts, no routing, no load, no stack) achieved 15.5ms (64.7 t/s, 7.6×) — proving ~103ms of per-layer overhead is eliminable.

## 2. Approach Attempted: Stage A (Global LRU + mx.take)

**Result: FAILED (0.66x, worse than baseline).**

Prototype: replaced dict cache with global LRU 3D buffer + `mx.take` for weight gathering. Kept routing sync (48 eval+tolist).

| Configuration | Time/step | t/s | vs baseline |
|-------------|-----------|-----|-------------|
| Baseline | 107.6ms | 9.3 | 1.00× |
| Stage A (slot cache + mx.take) | **163.7ms** | **6.1** | **0.66×** |

**Root cause**: `mx.take` from 3D buffer (21.6ms microbenchmarked) is slower than `mx.stack` (17.3ms) + dict lookup (11.4ms = 28.7ms total). The data copy overhead from the global buffer nullifies any benefit from removing dict individual allocations.

**Decision**: Stage A abandoned. Proceed directly to GPU-hit-path.

## 3. True Bottleneck (Revised)

| Component | ms | % | Notes |
|-----------|-----|-----|-------|
| Pure compute (matmul+attn+norms+embed) | 15.5 | 13% | Floor. Not optimizable. |
| Routing sync (eval+tolist+pipeline stall) | 13.0 | 11% | Tolerable. |
| **Dict lookup ×480 + stack ×192 + CPU dispatch** | **89.9** | **76%** | **Target.** ~61ms is GPU-idle CPU serialization. |

The 89.9ms = 28.7ms pure ops + ~61ms pipeline disruption. `mx.take` replacement doesn't fix this because `mx.take` still requires a CPU round-trip for slot indices.

**Only GPU-side slot resolution eliminates the round-trip.**

## 4. GPU-Hit-Path Design (Primary)

### Core Idea
Route on GPU (already GPU), resolve expert-to-slot on GPU (new), gather weights on GPU (gather_qmm), all in one computation graph. CPU only handles cache misses async.

### Pipeline (hit case — ~82% of requests)

```
[GPU] gate(xf) → softmax → argpartition(top_k)      # already GPU
[GPU] idx                                                    # already GPU
[GPU] slot = slot_map[idx]                                    # new: GPU array lookup
[GPU] weights = gather_qmm(global_3d_buffer, slot, rhs_indices=slots)  # new: fused gather+matmul
[GPU] _decode_moe(weights, ...)                               # same as compiled
→ single mx.eval at end of 48 layers                          # 1 sync/step (not 48)
```

No CPU round-trip for hit experts. 48 syncs reduced to 1.

### slot_map Structure

GPU array: `[n_layers, n_experts] → slot_index` (uint16)

- 48 × 512 = 24,576 entries × 2 bytes = 49KB — negligible.
- `slot_index ∈ [0, capacity)` or sentinel `0xFFFF` for "not resident".
- Updated by CPU only when an expert is loaded/evicted (async).

### Miss Handling

~18% of requests miss (82% hit rate from measurement). For misses:

**Option M1: Deferred fill (1-step stale)**
- Missed expert's slot is `0xFFFF` (sentinel).
- GPU gather gets zeros for that expert → contribution = 0.
- CPU detects miss IDs from sentinel in `slot_map` (need to check after step).
- Loads missed experts to buffer, updates slot_map for next step.
- **Quality effect**: 1 step where ~18% of one layer's expert contribution is zero per step.
  - Average: ~2-3 experts per step get zero weight instead of correct weight.
  - With top-10 routing, the missing expert's weight is redistributed to others (renormalized).
  - Impact: ~18% × ~10% per layer ≈ 1.8% output perturbation per step.
  - Need logits parity measurement to verify acceptability.

**Option M2: Synchronous fill (no stale)**
- CPU reads back miss IDs from slot_map sentinels.
- Loads experts to buffer, updates slot_map, evals buffers.
- Continues with correct gather.
- **Cost**: small sync per step for miss IDs only (~20% of 480 = ~96 IDs vs current 480).
- This is strictly faster than current path (current: 480 tolist + 480 dict get. M2: ~96 readback + 96 buffer writes).

**Recommendation**: Start with M2 (synchronous fill, minimal sync). If the small sync is still measurable (>5% of step time), evaluate M1 (stale).

### Global 3D Buffer

Single buffer `[capacity, ...]` shared across all layers:

```
buf_gate_wq: [6144, 512, 256] uint32  = 3.0 GB
buf_gate_s:  [6144, 512,  32] float16 = 0.2 GB
buf_gate_b:  [6144, 512,  32] float16 = 0.2 GB
buf_up_wq:   [6144, 512, 256] uint32  = 3.0 GB
buf_up_s:    [6144, 512,  32] float16 = 0.2 GB
buf_up_b:    [6144, 512,  32] float16 = 0.2 GB
buf_down_wq: [6144, 2048, 64] uint32  = 3.0 GB
buf_down_s:  [6144, 2048,  8] float16 = 0.2 GB
buf_down_b:  [6144, 2048,  8] float16 = 0.2 GB
Total: ~10.2 GB (vs ~12GB dict cache — saves ~1.8GB)
```

### Phase 0: gather_qmm Microbenchmark (pre-Phase 1 gate)

Per Claude directive #07 §1: verify gather_qmm N=1 is viable before Phase 1.

**Method**: Pre-filled 3D buffer (10 experts), gather_qmm fused gate+up call (single gather_qmm on concatenated [gate_wq, up_wq] buffer → split output) + down call. Compared to `_decode_moe` with pre-stacked weights. Warm, 2000 steps, 1 layer.

| Variant | 1-layer time | 48-layer extrap | vs baseline |
|---------|-------------|----------------|-------------|
| `_decode_moe` (no shared) | 0.2455ms | 11.8ms | 1.00× matmul baseline |
| `mx.stack` alone (pre-work) | 0.3610ms | 17.3ms | — |
| **stack + _decode_moe (total)** | **0.6065ms** | **29.1ms** | **1.00× total** |
| **gather_qmm fused GU** (no stack) | **0.2745ms** | **13.2ms** | **2.21×** |

**Verdict**: gather_qmm is **2.21× faster** than the current stack+_decode_moe path at the op level (including the pre-work it eliminates). The 3D buffer eliminates `mx.stack` entirely — the weight data is already contiguous.

**Critical distinction from Stage A (mx.take FAIL)**: `gather_qmm` fuses the gather into the quantized matmul — there is NO separate data copy. `mx.take` (Stage A) did an explicit copy from the 3D buffer into a new array, which cost more than `mx.stack`. `gather_qmm` eliminates that copy entirely. The "copy trap" that killed Stage A is avoided.

**Gate decision**: ✅ PASS. gather_qmm is ~free at the gather level (fused into matmul), and the pipeline benefit (eliminate stack+dict+tolist+48-sync) is additive and substantial.

### gather_qmm for decode
```python
# slot_map: [n_layers, n_experts] -> slot_index
slots = slot_map[layer_id][idx]           # GPU gather, no CPU round-trip
weights_g = mx.gather_qmm(xf, buf_gate_wq, buf_gate_s, buf_gate_b,
                          rhs_indices=slots, transpose=True, ...)
weights_u = mx.gather_qmm(xf, buf_up_wq, buf_up_s, buf_up_b,
                          rhs_indices=slots, transpose=True, ...)
h = (weights_g * mx.sigmoid(weights_g)) * weights_u
out = mx.gather_qmm(h, buf_down_wq, buf_down_s, buf_down_b,
                    rhs_indices=slots, transpose=True, ...)
```

This replaces: dict lookup ×10 + mx.stack ×4 + mx.quantized_matmul ×2.

### Expected Improvement (Provisional)

- **Hit path**: 15.5ms (compute floor) + gather_qmm overhead (~1-2ms) + miss handling
- **M2 (sync miss)**: ~13ms routing sync (kept per-layer) + ~3-5ms miss readback + ~1-2ms gather overhead
  - Estimate: ~20-25ms → **40-50 t/s (2.4-3.0×)** — **UNVERIFIED**
- **M1 (stale miss)**: 1 sync/step, miss detection deferred
  - Estimate: ~17-20ms → **50-60 t/s (3.0-3.6×)** — **UNVERIFIED, quality risk**

**Important**: 40-50 t/s is provisional. M2 retains per-layer sync for miss readback — the "1 sync/step" claim only applies to M1. Actual performance will be measured in Phase 2 (two tracks: all-resident hit-path ceiling, and real routing+M2).

This is below the 62.9 t/s upper bound (fixed experts, no routing) and above the 8.4 t/s baseline.

## 5. Memory Budget (eco 6144)

| Component | Current (dict) | Future (3D buffer) |
|-----------|---------------|-------------------|
| Expert weights | ~12GB (dict overhead) | ~10.2GB (contiguous) |
| slot_map | 0 | 49KB |
| **Total** | ~12GB | ~10.2GB |

3D buffer eliminates per-expert Python dict overhead (~1.8GB saving).

## 6. Implementation Plan (Stage B)

### Phase 1: slot_map + global 3D buffer (no decode changes yet)
1. Modify `slot_cache.py` `SlotResidentCache` to support global LRU mode.
2. Add slot_map GPU array `[n_layers, n_experts] → uint16`.
3. Validate hit rate matches current dict-based ResidentCache.

### Phase 2: decode MoE using gather_qmm from slot_map
1. Replace `StreamingMoE.__call__` N==1 path:
   - Keep router (gate + softmax + top-k) on GPU.
   - Remove `mx.eval(idx, w)` + `tolist` from hit path.
   - Add GPU slot_map gather → gather_qmm.
2. Add miss handling (M2: synchronous readback of miss IDs, load, buffer write).
3. Measure end-to-end warm t/s + logits parity (1e-3 vs baseline).

### Phase 3 (conditional): M1 stale miss if M2 sync cost measurable
1. If M2 sync adds >5% overhead, implement M1 (1-step stale).
2. Measure logits parity: if >1e-3, reject M1.

## 7. Gates

0. ✅ **Phase 0**: gather_qmm viability confirmed (stack込み2.21×勝ち, mx.take型コピー地雷回避済み).
1. ✅ **Phase 1**: GlobalSlotCache + slot_map 実装完了. slot_map 整合性・重み完全性・ResidentCache 命中率一致確認.
2. ⚡ **Phase 2 (i) all-resident**: 実装完了. パリティ検証済み (0.0). 実モデル未計測.
3. ⏳ **Phase 2 (ii) real routing + M2**: 実装完了. 計測・logits パリティ・メモリ未検証.
   - Gate: end-to-end ≥20 t/s (2.4× baseline) → full implementation.
4. **Phase 3 (conditional)**: M2 sync >2ms/step → implement M1 and measure logits parity.

## 8. Historical Record

| Attempt | Result | Date |
|---------|--------|------|
| Stage A: global LRU 3D buffer + mx.take | **0.66× FAIL** | 2026-07-16 |
| Phase 0: gather_qmm vs _decode_moe op-level | **stack込み2.21× gather勝ち** (閉じた比較ではgatherが無料) | 2026-07-16 |
| Phase 1: GlobalSlotCache + slot_map | **完了・放棄**: グローバル3D buffer + GPU slot_map [48,512]→uint16 | 2026-07-17 |
| Phase 2: _decode_moe_gather + StreamingMoE統合 | **完了・放棄**: gather_qmm 実装。実経路 0.55× で失敗 | 2026-07-17 |

**GSC decode 方向は行き止まり** (directive_deepseek_09.md)。
実 stream_generate 経路で gather+M2 (SSC=2000) = 0.55× (約1.8倍悪化/8.2 vs 14.9 t/s)。
死因: (1) M2 fill で SSD I/O が大量発生、(2) GSC+ResidentCache 二重保持で OOM、(3) per-layer eval は残る。
SSC=0（既定）で無効。コードは履歴として維持、新規開発に使わない。

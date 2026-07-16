# Phase 2 Report v3: gather_qmm + M2 検証結果（Claude #09 訂正）

## 訂正（v3）

**v2 の「1.28× / 継続可」は誤り。正しくは実 stream_generate 経路で 0.55×、行き止まり。**

v2 は self-contained per-step loop で計測したが、このループが実ルーティング/実 miss パターンを再現せず、GSC warm-up を過大評価していた。実 stream_generate (warm A/B) で再計測した結果、GSC 経路は baseline より遅い。

- Date: 2026-07-17
- Model: Qwen3.6-35B (hidden=2048, inter=512, experts=256, top_k=8, 40層)
- Hardware: MacBook Pro (Apple Silicon), 64GB unified memory
- Method: Self-contained measurement, generate 120 tokens, per-step loop

## 修正内容

### 逸脱① gather_qmm の正しい実装
`_decode_moe_gather` を mx.take から gather_qmm に変更。3 回の gather_qmm（gate, up, down）で MoE 計算を完結。x は `expand_dims` で 4D にしてから gather_qmm に渡す。**gather コピーゼロ、matmul に融合。**

### 逸脱② M2 miss handling
1件でも miss があれば即 fallback ではなく、**miss expert を同期充填 → 全 resident 化 → 常に gather_qmm を走らせる**。`gsc.get_slots(layer, miss_ids)` で store から load & GSC バッファ書き込み、その後に GPU slot_map 経由で gather_qmm。

## 計測結果（訂正後: 実 stream_generate warm A/B）

| 系統 | 設定 | decode t/s | baseline比 | hit率 | 備考 |
|------|------|-----------|-----------|-------|------|
| (C) Baseline | SSC=0 | **14.9** | **1.00×** | 83.9% | dict→stack→_decode_moe |
| (G) gather+M2 | SSC=2000 | **8.2** | **0.55×（約1.8倍遅い）** | 50.0% | GSC cold + M2 fill で悪化 |
| (P) Pre-primed | SSC=992 | — | — | — | G/C=0.55× の時点で中止 |

**v2 の「1.28×」は自己完結ループの測定アーティファクト。** 実 stream_generate 経路では GSC が悪化要因。

### 悪化要因

1. **M2 fill で SSD ロードが大量発生**: ResidentCache（83.9% hit）の代わりに GSC（コールドスタート）を使うため、ほとんどの expert を store.load（SSD I/O）から取得。GSC の容量 2000 は 10240 総 expert に対し小さすぎ、thrash 状態に。
2. **GSC バッファのメモリ二重化**: ResidentCache + GSC buffer でメモリが倍増。SSC を ResidentCache 容量 (6144) に合わせると OOM。
3. **per-layer の eval+tolist は残っている**（miss 検出に idx を CPU 側で要する）。

## STOP 条件判定

**不合格 → 中止。** 実経路 0.55× は副次 8% を大きく下回る。GSC decode 方向は行き止まり。

### 死因（構造的）

1. GSC は ResidentCache の「上に載る第2の expert キャッシュ」— 有効サイズで二重保持不可、小さくすれば thrash。
2. gather_qmm の 3 kernel 化（gate/up/down × 40層 = 120 kernel launch）は mx.stack＋2 quantized_matmul（80 kernel）より launch 数が多い。
3. 本質的に decode step のボトルネックは GPU compute ではなく、router sync + expert ロードの帯域。gather を fusion してもバイト数は減らない。

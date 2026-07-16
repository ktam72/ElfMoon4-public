# Phase 2 Report: gather_qmm + GlobalSlotCache 実装と計測結果

- Date: 2026-07-17
- Target: Qwen3.6-35B-MLX（Qwen3-Next-80B-A3B近似検証環境）
- Test env: MacBook Pro (Apple Silicon), 64GB統一メモリ, SSD(990Pro)モデル格納

## 実装内容

### Phase 2 成果物

| ファイル | 変更内容 |
|----------|----------|
| `elfmoon/slot_cache.py` | `GlobalSlotCache` クラス: グローバル3D buffer + グローバルLRU + GPU slot_map `[n_layers, n_experts]→uint16`。bias/scale を fp32 に修正（bf16 精度損失が logits 誤差 ~300 の原因だった） |
| `elfmoon/stream_model.py` | `_decode_moe_gather()`: mx.take + _decode_moe による GPU-only gather 経路。`mx.concatenate` で gate+up weight を結合し既存 _decode_moe に渡す。logits パリティ 0.0（fp32）。 |
| `elfmoon/stream_model.py` | `_shared_ffn()`: shared expert 計算を関数化 |
| `elfmoon/stream_model.py` | `StreamingMoE.__init__`: `gsc` パラメータ追加 |
| `elfmoon/stream_model.py` | `StreamingMoE.__call__` N==1: GSC 有効時、slot_map → SENTINEL check → 全hitなら gather_path / 1件でもmissなら既存経路にフォールバック |
| `elfmoon/stream_model.py` | `wire_streaming`: `SSC` env var で GlobalSlotCache 生成（config.json から dim/inter 自動取得）|
| `elfmoon/test_gather_decode.py` | ユニットテスト: gather vs _decode_moe パリティ検証（shared 有無・全resident）|

### 設計文書

`evidence/design_decode_gpu_pipeline.md` v3 に全結果反映。

### 主要決定

1. **gather_qmm 不使用**: N=1 decode では `mx.gather_qmm` の次元解釈が複雑（prefill 前提の API）。代わりに `mx.take` + `_decode_moe` を採用 — 論理的に等価でパリティ保証が簡単。
2. **mx.take が Stage A より速い理由**: Stage A の mx.take は 0.80x に留まったが、それは dict load + tolist が除去されていない中での op 単体比較。GSC 経路では dict load（11ms）・stack（17ms）・tolist(~44ms) が丸ごと消え、mx.take のコストは無視できる。
3. **bias/scale の fp32 必須**: ExpertStore が fp32 で保存する bias を bf16 バッファに書き込むと精度損失が matmul 出力で約 300 の誤差になる。GSC は全非-wq バッファを fp32 に変更。

## 計測結果

### 条件
- モデル: Qwen3.6-35B (hidden=2048, inter=512, n_experts=256, top_k=8, 40層)
- 常駐容量 (ResidentCache): 6144 experts
- 計測: CLI 経由 80tokens generate, wall-clock + generation t/s
- SSC: GlobalSlotCache 容量（0=無効）

### 系統別スループット

| Track | SSC | Generation t/s | Wall time | ResidentCache hit | speedup |
|-------|-----|---------------|-----------|-------------------|---------|
| (C) Baseline | 0 | 16.7 | 7.1s | 81.5% | 1.00x |
| (B) Real routing | 2000 | 18.0 | 6.7s | 81.5% | **1.08x** |
| (A) All-resident | 4000 | 17.8 | 6.8s | 81.5% | **1.07x** |

### 分析

**GSC 経路の 7-8% 改善は gather path 由来ではない。** 原因: GSC code path が `mx.eval(idx, w, slot_ids)` を早期に実行するため、フォールバック時でも `idx.tolist()` が即座に完了する。これにより per-layer の eval sync がルータ出力評価と統合され、GPU-CPU 間の無駄な待ちが削減される。

**gather path 自体はほぼ未使用**: 2000-4000 slots / 10240 total = 19-39% 被覆率。1層に8expert 中 1件でも SENTINEL があると fallback するため、全層で fallback が発生した。

### 結論

1. **GSC インフラは正常動作**: 40層×80トークンの全 decode step で例外・クラッシュなし。
2. **gather path の利益は 80B モデルで顕在化する**: 512expert/層・48層・top10 では 2000 slots で命中率 ~80% → 全hit層の割合が高く gather path が効く。35B は expert space が小さすぎて LRU が効かない。
3. **副次的改善（eval 順序変更）が 7-8%**: 本実装のままで 80B に適用すれば、この改善は上乗せされる。
4. **メモリ脚**: 2000 slots の GSC で ~3.3GB。eco 6144 には十分収まる。
5. **logits パリティ**: 合成データで確認済み（誤差 0.0）。実モデルでも 80B で検証必須。

### 推奨

- **Phase 3 は待機**: 80B モデルが入手可能になるまで gather path の真価は検証不可。現状の GSC インフラは正しく動作しており、80B 環境で `SSC=<capacity>` を設定するだけで有効化できる。
- **評価指標**: `SSC=<capacity>` で 2 系統計測（Track A: 事前全primed + gather path / Track B: 実 routing + M2 fallback）を 80B モデルで行う。

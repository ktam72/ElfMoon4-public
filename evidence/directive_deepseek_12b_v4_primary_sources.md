# DeepSeek 向け 指示 #12b（V4 一次情報の反映・feasibility を最初の hard gate に）

- 作成日: 2026-07-17
- 前提: `directive_deepseek_12_v4_kickoff.md`、公式ページ `huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark`、ローカル実データ
- 位置づけ: #12 の追補。**Phase 0 の前に「feasibility gate」を追加**する。

---

## 1. 一次情報の所在（推定禁止・ここから取る）

- **技術報告書 arXiv 2606.19348** — アーキテクチャ全体、mHC の式、CSA/HCA、indexer。**最優先の一次情報。**
- **モデル repo の `inference/` フォルダ**（公式参照実装コード）— 実装の写経元。mlx_lm `deepseek_v32.py` は近縁だが V4 そのものではないので、**`inference/` を正とする**。
- DeepSpec repo — DSpark（投機デコード, V4 に外付け）。**初回ポートでは対象外。**

## 2. 用語・構造の確定（旧 doc の訂正）

- **CSA/HCA は実在する公式用語**: Compressed Sparse Attention / Heavily Compressed Attention（長文脈効率化。1M で V3.2 比 27% FLOPs・10% KV）。旧 doc は**略語は正しく、展開（Cross-Sliding/Hybrid Chunk）と次元が誤り**。実次元は実重み＋報告書で確定する。
- **mHC = Manifold-Constrained Hyper-Connections**（`attn_hc/ffn_hc.{base,fn,scale}`）。残差接続の強化。**式は arXiv 2606.19348 から取る。**
- 量子化（ローカル実測）: **experts = mxfp4(group32, bits4) / 非expert = affine int4(group64, bits4)**。公式の「FP8 混在」ではなく MLX 4bit 量子化版。→ **FP8 対応は不要**。`_wire_deepseek_v4` の `mode="mxfp4", group_size=32` 想定と一致。

## 3. 【最重要・新規】Phase −1: feasibility gate（実装より前）

**284B・282GB・expert 7倍サイズという規模が 24GB に収まり実用速度で動くかを、実装前に定量評価する。** ここが NG なら V4 対応は中止/縮小。

### 実データ（Claude 確認済み）
- ディスク **282GB**、43層 / hidden 4096 / 256 experts / top6 / shared1 / **moe_intermediate 2048**。
- **expert 1個 ≈ 12.6MB**（3 × 2048 × 4096 × 4bit）＝ **80B の expert(1.77MB) の約7倍**。
- sparse 層 41 × 256 = 10,496 experts。

### 見積もるべき数値（DeepSeek が算出）
1. **常駐メモリ**: 非expert 常駐（MLA attention・embed・lm_head・mHC・norm を hidden4096×43層で）＋ shared expert 常駐 ＋ 活性化ピーク。残りが expert 常駐キャッシュに使える量。
2. **expert キャッシュ被覆率**: eco 24GB で常駐できる expert 数 ÷ 10,496。expert が 7倍大きいので、80B(6144/24576=25%)より大幅に下がるはず。命中率を試算。
3. **decode 速度**: top6 × 41層 × 12.6MB ≈ **3.1GB/token の expert 読み込み**。命中しない分は SSD(実測 ~4.4GB/s)。純ミスなら ~1.4 t/s、命中込みで **~2〜6 t/s** の見込み。実際の命中率で幅を詰める。
4. **prefill 速度**: gather_qmm 経路でも expert サイズ 7倍で mmap 帯域が効く。試算。

### 判定
- **decode が実用下限（例: 5 t/s）を割る／常駐が 24GB に収まらない場合は、PM に「V4-Flash は 24GB の実用圏外」と上げて中止 or perf モード(より大 RAM)前提に切替**。
- 収まって実用速度が出る見込みが立って初めて Phase 0（設計）に進む。

**この feasibility 見積り（数値＋前提）を最初の成果物として PM/Claude レビューに上げること。実装コードはまだ書かない。**

## 4. 以降（#12 のまま）

feasibility OK なら #12 の Phase 0（差分表・`inference/` 写経元特定）→ Phase 1（attention）→ Phase 2（model_v4 + streaming MoE）→ Phase 3（検証・実測）。規律（一次情報・各 Phase ゲート・実 stream_generate warm A/B）は不変。

## 5. まとめ（最初の一手・改訂）

1. **feasibility gate（§3）を最優先**: 常駐メモリ・被覆率・decode/prefill 速度の定量見積り → PM 判断。
2. OK なら arXiv 2606.19348 と `inference/` フォルダを一次情報に、Phase 0 差分表へ。
3. NG なら中止 or 前提変更（RAM/変種）を PM に上げる。

**規模が ElfMoon の 24GB ニッチの外かもしれない。まず数字で確かめる。**

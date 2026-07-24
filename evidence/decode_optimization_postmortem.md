# decode_optimization_postmortem

このドキュメントは decode 高速化の試行錯誤を記録する。
各エントリは試行・結果・死因を1-3行で要約する。

---

## 2026-07-18: GlobalSlotCache（GSC）

- **方法**: ResidentCache + GlobalSlotCache 二重キャッシュ + gather_qmm
- **結果**: 0.28x（baseline比）、OOM 多発
- **死因**: M2 fill の SSD ロード＋二重キャッシュ OOM。
  （#18 で gather_qmm カーネル自体は隔離 2.5x と確認。GSC の死因は二重化と SSD ロードに限定。）
- **指示**: `directive_deepseek_09_dispatch_batch.md`

## 2026-07-18: expert 低ビット化

- **方法**: 2bit/3bit requantization
- **結果**: SSD前提シナリオで最大 ~2x 確認したが品質低下が顕著
- **死因**: 品質トレードオフが許容範囲外
- **指示**: `directive_deepseek_14_bits_topk.md` Task ①

## 2026-07-18: top_k 削減

- **方法**: top_k を 10→4 に削減
- **結果**: 実効 ~1.6x（warm A/B）、品質トレードオフあり
- **状態**: opt-in つまみとして確定。デフォルト据え置き。
- **指示**: `directive_deepseek_14_bits_topk.md` Task ②、`directive_deepseek_15_topk_verify.md`

## 2026-07-18: dispatch バッチ化（連続 slot 配列 + mx.take）

- **方法**: ResidentCache 内部再レイアウト。連続 slot 配列 + mx.take で mx.stack 削除。
- **結果**: 期待速度向上 ~10%。GO 条件（≥1.3x）未達。
- **死因**: mx.eval による weight materialize（1.24ms/call）が連続配列化でも軽減不可。
- **状態**: NO-GO（仮結論）。
- **指示**: `directive_deepseek_16_dispatch_batch.md`

## 2026-07-18: gather_qmm zero-copy decode（#17 → #18 差し戻し後）

- **方法**: `ELFMOON_GATHER_DECODE=1`、lazy-build contiguous arrays from cache + gather_qmm
- **結果**: 1.7 t/s（11.1 t/s baseline比 0.15x）。gather 通過率 11.3%（169/1488）。
- **死因（推定）**:
  1. **p^10 ゲート**: gather 通過率 11.3% → per-expert カバレッジ p ≈ 80%
     （キャッシュ容量制約で説明可能。p が 95% 超なら連続配列の同期不良が疑われるが未検証）
  2. **二重化**: cache + contiguous arrays で +1.2 GB（指示の常駐不変に違反）
  3. 固定 overhead（routing eval 0.68ms, tolist, cache 操作）が gather_qmm の利得を相殺
- **状態**: STOP（最終）。gather_qmm zero-copy は **この変種（lazy-rebuild コピー、二重化）で不成立**。
  真の in-place slot 同期（cache と contiguous 配列が同一メモリを参照）は未実証のまま費用対効果で打ち切り。
  「構造的に不可能」とは書かない。
- **構造メモ**: top_k=10 の全ヒットゲートは p^10 で効く。
  p=99%→通過率 90%、p=95%→60%、p=80%→11%。
  将来 MLX 側が改善した時の再起票判断に有用。
- **指示**: `directive_deepseek_18_gather_qmm_retry_report.md`

---

## decode 高速化キャンペーン総括（#14〜#18）

| レバー | 結果 |
|---|---|
| **top_k 削減** | ✅ 唯一の確定成果：opt-in ~1.6x（top_k=4）、デフォルト据え置き |
| **expert 低ビット** | ❌ bf16→3bit でも品質不合格 |
| **dispatch バッチ化（mx.take）** | ❌ 効果 ~10% 未満で NO-GO |
| **gather_qmm zero-copy** | ❌ 隔離 2.5x だが実経路 0.15x で打ち切り |

### 再発防止チェックリスト
- [x] 物理上限突合（#15）: top_k 倍率が物理上限内であることを確認
- [x] 床一致確認（#15/#18）: baseline が既知床（11.1 t/s）と一致
- [x] p^10 ゲート（#18）: gather_qmm の通過率が per-selection coverage の冪乗で決まる構造を記録
- [x] 誤帰属クローズ禁止（#18）: 「gather_qmm カーネルが遅い」と閉じず、隔離 2.5x の事実と矛盾しない死因を記載

以後の decode 施策は MLX 本体の改善など外部要因が出た時のみ再起票。

## 2026-07-19: F（in-place slot 再レイアウト）hard gate で STOP・真の死因が最終確定

実装前 hard gate（隔離・実サイズ・parity 0.0 のクリーン測定）で判明:

**`mx.gather_qmm` の単トークン実行時間は重みテーブルの expert 数 E に線形比例する（O(E)）。**

| E | stack+qmm | gather_qmm | 比 |
|---|---|---|---|
| 64 | 0.466ms | 0.434ms | 1.08x |
| 128 | 0.449ms | 0.579ms | 0.78x |
| 256 (35B) | 0.450ms | 0.897ms | 0.50x |
| 512 (80B) | 0.460ms | 1.576ms | 0.29x |

- 少行数（decode）では gather カーネルがテーブル全体に触れるため、top_k=10 だけ読む stack 経路に勝てない。
- prefill で gather_qmm が 3.5x 勝つのは N が大きく O(E) が償却されるから。decode(N=1) では償却不能。
- **#17 隔離 PoC の「2.5x」は E=64 縮小構成の産物**（E=64 なら互角〜勝ち）。実モデル E=256/512 では構造的に負け。
- これで GSC 0.28x（#08）・#18 の 0.15x も統一的に説明がつく（二重キャッシュ/フォールバックに加えカーネルが O(E)）。
- チャンク分割 gather も不成立（top_k が複数チャンクに散り、呼び出し回数×O(64) で悪化）。
- Gate1（slab in-place 書き込み）は 0.2ms と安価で合格だったが、Gate2 不合格により無意味。

**結論: decode の gather_qmm 系 zero-copy は MLX 現行カーネルでは構造的に不可（O(E) が原因）。**
再起票条件を「MLX が O(top_k) の gather_qmm（少行数最適化）を実装した時」に具体化する。
hard gate 方式が機能した（実装 0 行・約1時間で 3 敗目を回避）。

## 2026-07-20: store v2（層単位ファイル）は撤退 — 「隠れ第2キャッシュ」の再演

integrate.py の store 形式変更（expert 単位 → 層単位ファイル）を実装・実測して撤退。

- **mmap 版**（当初 +45% decode / 2.6x prefill @35B）: キャッシュした expert 配列が層ファイル全体の
  GPU バッファ（bytes_no_copy）をピン留め。実質「store 全体を RAM に載せる無制限の第2キャッシュ」で、
  80B は 48×~0.9GB ≈ 43GB で **Metal OOM**。35B の 18GB も 24GB 機では綱渡り。
  **GSC / #18 と同じ「第2キャッシュは OOM まで速い」パターン**。速度の出所を必ずメモリ会計と突き合わせること。
- **pread 版**（ピン留めゼロの安全版）: open+parse 0.4ms/miss は消えるが、コピー＋CPU 配列化コストが上回り
  実経路で **35B -18% / 80B -31%**。micro（≈同等）と実経路（劣化）はまたも乖離。
- **結論**: v1（expert 単位ファイル）は「ピン留め粒度＝キャッシュ粒度」で既に最適。
  mmap ゼロコピーを保ったままピン留めを避ける手段は MLX Python API に無い。
  store 形式は速度レバーではない（DeepSeek 4 提案時の判定が正しかった）。コードは bad803a に完全リセット済み。
- 副産物: mx.save_safetensors はキー順を保存しない（連続配置が要る場合は自前ライタが必要）。

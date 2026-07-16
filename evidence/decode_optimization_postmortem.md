# 80B Decode 高速化キャンペーン 総括

- 作成日: 2026-07-17
- 期間: 2026-07-15 〜 2026-07-17
- 結論: **decode 側の高速化は打ち止め。prefill 側に資源集中。**

## 試行した方向

| 試行 | 結果 | 要因 |
|------|------|------|
| 投機デコード (R≒4.75) | 不可 | 80B A3B の routing パターンが投機に不適合 |
| Stage A: global LRU + mx.take | 0.66× FAIL | mx.take のコピーコストが mx.stack より大 |
| Stage B: gather_qmm GSC + M2 | 0.55× FAIL | M2 fill で SSD I/O 大量発生、GSC 二重保持で OOM |
| gather_qmm prefill | **成功 (3.5×)** | 融合テンソル mmap + gather_qmm で prefill が 3.5倍 |

## 死因分析

### GSC decode の死因（3層）

1. **M2 fill が store.load（SSD I/O）を大量発生させる**: ResidentCache は hit 率 83.9% で大部分の expert を CPU メモリから返す。GSC は容量不足で cold-thrash し、毎 step の M2 fill が SSD I/O をトリガ → 8.2 t/s に低下。
2. **GSC + ResidentCache の二重保持がメモリを圧迫**: ResidentCache 単体で 10.4GB。GSC (2000 slots) で 3.3GB 追加 → eco 6144 に収まらず OOM の危機。容量を減らせば thrash 加速。
3. **per-layer eval+tolist が除去できない**: miss 検出に idx の CPU 値が必要（`mx.eval(idx,w)` + `tolist()`）。GSC パスでもこの同期は残り、狙った「全 GPU 化」は達成できず。

### 本質的制約

**decode の 1 step は 48 層 × top-10 の expert 重みを CPU-GPU 間で往復させる必要がある。** このバイト数（~4MB/step）を減らす機構はなく、gather/cache 再配置では帯域が減らない。帯域不足なら投機（accept 率不足）にも GSC（SSD I/O 増加）にも効かない。

80B decode ~10 t/s は streaming-expert 設計の実質的な床と考える。

## 確定した成果（本番投入可能）

| 成果 | スピード | 備考 |
|------|---------|------|
| gather_qmm prefill (35B) | 3.5× | 融合テンソル mmap + gather_qmm。80B 展開 pending |
| 投機デコード知見 | — | R≈4.75 不足の確認。routing パターン分析完了 |
| GSC 知見 | — | gather_qmm + M2 が実経路で効かない確認。同種の試行を回避可能 |
| 計測手法教訓 | — | 速度主張は実 stream_generate warm A/B 必須。micro/self-contained は誤導する |

## 今後の方向性

Claude #09 §6 の推奨に従い:

1. **prefill 側の 80B 展開**: gather_qmm prefill が 35B で 3.5× 確認済み。80B でも同等の効果が期待できる。優先度: 高。
2. **レバーB（層ごと融合永続化）**: prefill で読んだ融合テンソルをチャンク間で使い回す。既に効くと分かっている方向。
3. **decode 側は静的**: 現行の ResidentCache + stack + _decode_moe 経路を維持。新規最適化の試行は prefill 側に集中。

## 計測手法の誓約

今後、速度主張は以下の条件で行う:
- **実 stream_generate 経路**（mlx_lm.generate を通す）
- **warm A/B**（cold start ではなく warm 状態で比較）
- micro-benchmark / 単層計測 / self-contained loop / 固定 expert は**速度判定に使わない**（3 回連続で誤導した）。

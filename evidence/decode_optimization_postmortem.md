# 80B Decode 高速化キャンペーン 総括

- 作成日: 2026-07-17
- 期間: 2026-07-15 〜 2026-07-17
- 結論: **decode 側の高速化は打ち止め。prefill 側に資源集中。**

## 試行した方向

| 試行 | 結果 | 要因 |
|------|------|------|
| 投機デコード (R≒4.75) | 不可 | compiled 単トークン decode が既に速く、group 検証パスが 4.75× コスト、tok/pass ~1.7 では相殺不能（35B 実測） |
| Stage A: global LRU + mx.take | 0.66× FAIL | mx.take のコピーコストが mx.stack より大 |
| Stage B: gather_qmm GSC + M2 | 0.55× FAIL | M2 fill で SSD I/O 大量発生、GSC 二重保持で OOM |
| gather_qmm prefill | **成功 (3.5×)** | 融合テンソル mmap + gather_qmm で prefill が 3.5倍 |

## 死因分析

### GSC decode の死因（3層）

1. **M2 fill が store.load（SSD I/O）を大量発生させる**: ResidentCache は hit 率 83.9% で大部分の expert を CPU メモリから返す。GSC は容量不足で cold-thrash し、毎 step の M2 fill が SSD I/O をトリガ → 8.2 t/s に低下。
2. **GSC + ResidentCache の二重保持がメモリを圧迫**: ResidentCache 単体で 10.4GB。GSC (2000 slots) で 3.3GB 追加 → eco 6144 に収まらず OOM の危機。容量を減らせば thrash 加速。
3. **per-layer eval+tolist が除去できない**: miss 検出に idx の CPU 値が必要（`mx.eval(idx,w)` + `tolist()`）。GSC パスでもこの同期は残り、狙った「全 GPU 化」は達成できず。

### 本質的制約

**decode の 1 step は active ~3B params（4-bit で ~1.5GB/token）の expert 重みを統一メモリから GPU が読む帯域に律速される。** このバイト数を減らす機構はなく、gather/cache 再配置では帯域が減らない。帯域不足なら投機（accept 率不足）にも GSC（SSD I/O 増加）にも効かない。

80B decode ~10 t/s は streaming-expert 設計の実質的な床と考える。

## 成果サマリ（このキャンペーンの確定物）

| 領域 | 状態 |
|------|------|
| prefill gather_qmm (35B) | ✅ **3.5×**（速度+パリティ確認済み） |
| prefill gather_qmm (80B) | ⏳ 速度確認済み（3.5×相当）。**パリティ未検証** → §2 で取得後に完全確定 |
| MCP / opencode tool_calls | ✅ 実動作確認済み |
| decode | ⛔ **投機・GSC とも行き止まり。~10t/s(80B) は床。打ち止め済み** |
| レバーB（融合永続化） | ⏸ **保留**: gather_qmm mmap で既に達成済み。cold TTFT 問題が顕在化した場合のみ再検討 |
| 計測手法教訓 | ✅ 総括文書に誓約済み |

### 確定知見

- 投機デコード: compiled 単トークン decode が既に速く、group 検証パス 4.75× コスト、tok/pass ~1.7 では相殺不能（35B 実測）
- GSC decode: gather_qmm + M2 が実経路 0.55× で効かない確認。同種の試行を回避可能
- 速度主張は実 stream_generate warm A/B 必須（micro/self-contained は 3 回連続で誤導）

## 今後の方向性

Claude #09/#10 の推奨に従い:

1. **80B prefill logits パリティ取得**（§2）→ 確定後に本番投入
2. ~~**レバーB（層ごと融合永続化）**:~~ → **保留**: cold TTFT が実ユーザーの問題として挙がった場合のみ再検討
3. **decode 側は静的**: 現行経路維持。新規最適化の試行は行わない

## 計測手法の誓約

今後、速度主張は以下の条件で行う:
- **実 stream_generate 経路**（mlx_lm.generate を通す）
- **warm A/B**（cold start ではなく warm 状態で比較）
- micro-benchmark / 単層計測 / self-contained loop / 固定 expert は**速度判定に使わない**（3 回連続で誤導した）。

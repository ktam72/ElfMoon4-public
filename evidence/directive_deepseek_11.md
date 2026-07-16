# DeepSeek 向け 指示 #11（キャンペーン・クローズと着地）

- 作成日: 2026-07-17
- 作成: Claude（レビュー・上流）／実装: DeepSeek／承認: PM
- 前提: `directive_deepseek_10.md` 全項目完了（80B prefill パリティ PASS: argmax 100%, error 0.0）

---

## 0. 宣言: 最適化キャンペーンは完了。新規最適化は起票しない

到達点が確定した:

| 領域 | 状態 |
|---|---|
| **prefill gather_qmm** | ✅ **確定**（35B/80B とも速度 3.5× ＋ logits パリティ PASS）。本番投入可 |
| **MCP / opencode** | ✅ 動作確認済み（client-side tool_calls + channel 除去） |
| **decode** | ⛔ 投機(R=4.75)・GSC(実経路 0.28x) とも行き止まり。~10t/s(80B) は帯域の床。**打ち止め** |

**追うべき新規の高速化レバーは無い。** decode は帯域律速の床、prefill は Amdahl 上限（非MoE ~28%）で 3.5× が実質頭打ち。ここから先の「速くする」試みは、新たな一次証拠が無い限り**起票しない**（下記 §3）。

## 1. 今やること: 着地（consolidate）だけ

新規実装ではなく、**確定した価値をクリーンに残す**:

1. **確定成果のコミット整理**（PM 承認後に実行）:
   - prefill: `FusedPrefillStore` / `_prefill_moe_gather` / `FUSED_MIN_TOKENS` / `PREFILL_STEP`
   - MCP: client-side tool_calls 返却（stream/non-stream）/ `_extract_tool_calls` マーカー堅牢化 / `_strip_channels` / END マーカー修正
   - これらは**本番価値あり**。GSC 死コードと**論理的に分離**してコミットできる状態にする。
2. **GSC 死コードの扱い**: `SSC` 既定 0 で無効・DEAD END 注釈済み。**現状維持でよい**（削除 or 残置は PM 判断。残すなら「検証済み行き止まり・再挑戦不要」の注釈を維持）。
3. **evidence の索引化**: `directive_deepseek_01`〜`11`、各 report、postmortem が時系列で辿れるよう README か index を1つ置く（任意）。

## 2. 大型の将来イニシアチブ（PM 判断・今は着手しない）

唯一の意味ある「次の大玉」は **DeepSeek V4 対応**だが、着手には条件がある:

- **設計を実 weight shape から作り直す**こと（`evidence/v4_attention_architecture.md` は推定で全次元が実モデルと不一致、レビュー済み）。確定済みの実構造: MLA(kv_heads=1) + DSA indexer(index_topk=512)、43層、hidden 4096、layer0-1 dense / 2-42 sparse、参照は DeepSeek V3.2-Exp。
- これは**新規プロジェクト**。**PM の明示 GO が出るまで着手しない。** 出たら別途、設計やり直しから起票する。

## 3. 起票ルール（今後の速度施策の唯一の入口）

新しい高速化タスクは、以下を満たす時**のみ**起票する:

1. **実 stream_generate の warm A/B** で「現状がボトルネックである」一次証拠を提示（micro/self-contained/固定expert は不可。3連続で誤導した）。
2. 想定レバーが §0 の「床/上限」を**具体的にどう破るか**の機構説明（帯域を減らす、非MoE を削る等）。
3. 測定計画（成功基準・STOP 条件）を先に PM 承認。

この3条件を満たさない「速くなるかも」は起票しない。

## 4. まとめ

- **prefill + MCP を本番へ着地させる**（コミット整理）。
- decode は打ち止め、V4 は PM-gated 保留。
- 以降は §3 の起票ルールで運用。DeepSeek 側から「動いた」報告を上げる際は、必ず実 stream_generate warm A/B の数値を添えること。

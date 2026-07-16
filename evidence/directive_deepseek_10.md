# DeepSeek 向け 指示 #10（クローズアウト: 成果の確定と締め）

- 作成日: 2026-07-17
- 作成: Claude（レビュー・上流）／実装: DeepSeek／承認: PM
- 前提: `decode_optimization_postmortem.md`、`directive_deepseek_09.md`
- 方針: **新規の投機的最適化は増やさない。** 確定した成果（prefill 3.5×）を正しく締め、記録の精度を上げる。decode は打ち止め済み。

---

## 1. 総括文書の数値訂正（2点・記録の精度）

`decode_optimization_postmortem.md` を訂正すること:

1. **投機デコード R≈4.75 は「35B で Claude が実測」した値**。表は 80B campaign 下にあるが 80B の spec は R 未測定（保留のまま終了）。死因も「80B routing パターン不適合」でなく **「compiled 単トークン decode が既に速く、group 検証パスが 4.75× コスト、tok/pass ~1.7 では相殺不能」** に修正。
2. **「~4MB/step」は桁違い**。80B A3B は active ~3B params、4-bit で **~1.5GB/token の expert 重みを読む**。機構も「CPU-GPU 往復」でなく **「統一メモリ上の常駐重みを GPU が読む帯域」**。結論（帯域律速＝床）は正しいので、数値と表現だけ直す。

## 2. 唯一の実質残ゲート: 80B prefill の logits パリティ検証

prefill gather_qmm は **速度は 35B/80B とも実測済み（end-to-end 3.5×）**だが、**数値パリティは 35B しか確認していない**（argmax 一致）。80B は速度のみ。本番の信頼性のため 80B でパリティを取る:

- **方法**: 80B 実モデルで、同一プロンプト（~1000tok）を (a) fused 経路（現状）と (b) per-expert 経路（`_fused_store=None` 強制）でプレフィル → 続く decode の logits を比較。**argmax 一致 + 最大誤差 ~1e-3 以内**を確認。
- 不一致なら fused 経路の dtype/量子化を点検（prefill の float32 合算等）。
- これが通れば **prefill 最適化は 35B/80B とも本番投入可で確定**。

## 3. コード衛生と確定

1. **GSC decode 死コード**: `SSC` 既定 0 で無効、DEAD END 注釈済み。**現状維持でよい**（消すか残すかは PM 判断。残すなら「検証済み行き止まり・再挑戦不要」と明記されていること）。
2. **prefill 最適化の確定コミット**: `FusedPrefillStore` / `_prefill_moe_gather` / `FUSED_MIN_TOKENS` / `PREFILL_STEP` が feature ブランチに入っているか確認。GSC 死コードと混在しているなら、prefill 成果だけをクリーンに識別できる状態にする（コミット分離は PM 承認後）。
3. **既定挙動の非回帰**: `SSC=0`（既定）で decode が従来どおり（stack + _decode_moe）であること、prefill が fused 経路であることを 1 度 warm 実測で確認。

## 3.5. レバーB は「条件付き・保留」

`elfmoon-prefill-optimization` memory の「レバーB（層ごと融合永続化）」は、**gather_qmm を mmap 経由で既に達成済みのため前提が消えている**（融合テンソルは元 safetensors の mmap で足りている）。cold TTFT が実ユーザーの明確な問題として挙がった場合のみ、測定計画つきで再検討する。**今は着手しない。**

## 4. 着手しないこと（明示）

- decode 側の新規最適化（GSC・投機・その派生）。打ち止め済み。
- 速度目的の leverB / 新キャッシュ機構。
- self-contained / micro / 固定 expert による速度主張（総括の誓約どおり）。

## 5. 成果サマリ（このキャンペーンの確定物）

| 領域 | 状態 |
|---|---|
| prefill gather_qmm | ✅ 3.5×（35B 速度+パリティ、80B 速度）。§2 で 80B パリティ取得後に完全確定 |
| MCP / opencode | ✅ tool_calls 返却 + channel 除去、実動作確認済み |
| decode | ⛔ 投機・GSC とも行き止まり、~10t/s(80B) は床。打ち止め |

§1（訂正）+ §2（80B パリティ）+ §3（衛生）で本キャンペーンはクローズ。以降の速度施策は、新たな一次証拠（実 stream_generate warm A/B）が出た時のみ起票する。

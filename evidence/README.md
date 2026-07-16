# Evidence Directory Index

最適化キャンペーンの全記録。時系列順。

## 指令（Claude → DeepSeek）

| # | ファイル | 内容 |
|---|---------|------|
| 01 | `optimization_directive_deepseek.md` | 初回設計指示: 80B decode bottleneck 分析・gather_qmm prefill構想 |
| 02 | `directive_deepseek_02.md` | 投機デコード feasibility 検証指示 |
| 03 | `directive_deepseek_03.md` | prefill 高速化 + 数値パリティ検証指示 |
| 04 | `directive_deepseek_04.md` | MoE 4成分分解 + barrier 帰属検証指示 |
| 05 | `directive_deepseek_05.md` | Stage A (global LRU) 試行→失敗受けて方針転換指示 |
| 06 | `directive_deepseek_06.md` | decode GPU pipeline 設計指示 |
| 07 | `directive_deepseek_07.md` | Phase 0 必須化・M1/M2 sync 訂正・設計文書修正指示 |
| 08 | `directive_deepseek_08.md` | Phase 2 v1 却下: gather_qmm + M2 の正しい実装要求 |
| 09 | `directive_deepseek_09.md` | Phase 2 v2 却下・裁定: GSC 行き止まり・STOP条件不合格 |
| 10 | `directive_deepseek_10.md` | クローズアウト: 総括訂正・80B パリティ・コード衛生 |
| 11 | `directive_deepseek_11.md` | キャンペーンクローズ: 着地・起票ルール・V4 保留 |

## 計測レポート

| ファイル | 内容 |
|---------|------|
| `80b_decode_profile.md` | 80B decode 5-step ablation + MoE 成分分解プロファイル (v4) |
| `80b_prefill_gather_qmm_result.md` | gather_qmm prefill 35B/80B 速度 + パリティ結果 |
| `poc_decode_optimization_report.md` | pre-buffered PoC (上限値 7.64x) |
| `phase2_report.md` | Phase 2 v1 (mx.take + fallback) — **撤回済み #08** |
| `phase2_report_v2.md` | Phase 2 v2-v3 (gather_qmm + M2, 訂正後: 実経路 0.55x) |
| `decode_optimization_postmortem.md` | 総括: 全試行の死因・確定成果・教訓 |
| `design_decode_gpu_pipeline.md` | GPU pipeline 設計文書 v3 (DEAD END 注釈済み) |

## アーキテクチャ

| ファイル | 内容 |
|---------|------|
| `v4_attention_architecture.md` | DeepSeek V4 注意機構推定設計 — **推定・全次元不一致のため保留** |

## 凡例

- ✅ 確定（本番投入可）
- ⛔ 行き止まり（検証済み・再挑戦不要）
- ⏸ 保留（条件付き）

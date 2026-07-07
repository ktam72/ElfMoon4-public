# Change Request Record

## CR-001: Qwen3.6-35B-A3B モデル移行 + 各種最適化

### 概要
旧モデル（Qwen3-30B-A3B）から新モデル（Qwen3.6-35B-A3B）への移行に伴う一連の修正。

### 変更理由
- より高性能な Qwen3.6 シリーズへの移行
- 思考モード（`<think>` ブロック）対応による応答品質向上
- SSM（State Space Model）層を含むハイブリッドアーキテクチャ対応

### 変更内容

| # | ファイル | 変更内容 |
|---|---------|---------|
| 1 | `elfmoon/integrate.py` | キープレフィックス自動検出（`language_model.model` 対応） |
| 2 | `elfmoon/stream_model.py` | MODEL_PATH 更新、レイヤーアクセス `model.layers` 対応 |
| 3 | `elfmoon/stream_model.py` | SharedExpert 計算を StreamingMoE に追加 |
| 4 | `elfmoon/stream_model.py` | `wire_streaming` で shared_expert をキャプチャ |
| 5 | `elfmoon/stream_model.py` | デコード `@mx.compile` 最適化（約57%高速化） |
| 6 | `elfmoon/kv_manager.py` | ArraysCache（SSM層）の保存・復元対応 |
| 7 | `elfmoon/kv_manager.py` | タグ付きフォーマットで KV/SSM 混在キャッシュを管理 |
| 8 | `elfmoon/api_server.py` | model_lock を prefill の model() のみに短縮 |
| 9 | `elfmoon/api_server.py` | デフォルト常駐容量 2800→6144 |
| 10 | `elfmoon/chat.py` | デフォルト常駐容量 2800→6144 |
| 11 | `models/qwen3.6-35b-mlx/` | モデル入れ替え（旧: qwen3-30b-instruct-mlx） |
| 12 | `README.md` | 全面刷新（llama.cpp比較、chat.py/api_server.py使い分け） |

### 影響範囲
- 旧モデル（Qwen3-30B-A3B）は削除済み
- `tokenizer_config.json` の tokenizer_class を PreTrainedTokenizerFast に変更
- 新モデルは 256 experts/層 × 40層 = 10240 expert ファイルが必要

### 性能比較

| 指標 | 旧モデル (Qwen3-30B) | 新モデル (Qwen3.6-35B) |
|------|---------------------|----------------------|
| デコード速度 | 25 t/s | 25 t/s |
| 常駐 expert | 2800 / 6144 (45%) | 6144 / 10240 (60%) |
| 応答品質 | 標準 | 思考モード付き |
| モデルサイズ | 16 GB | 19 GB |

### ステータス
- [x] モデル入替完了
- [x] SharedExpert 対応
- [x] KV Cache 永続化（SSM含む）
- [x] 並行リクエスト最適化
- [x] README 更新

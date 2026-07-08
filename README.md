# ElfMoon 🌙

**24GB Apple Silicon で 35B MoE モデルを実用速度で動かす MLX 推論エンジン**

OpenAI 互換 API サーバーと対話 CLI を同梱。Claude Code / opencode 等の既存ツールから直接利用できる。

---

## llama.cpp との比較

| | llama.cpp (expert-offload) | **ElfMoon** |
|---|---|---|
| メモリ使用量 | 16 GB（全 expert 常駐） | **6.9 GB**（ストリーミング） |
| ストレージ | GGUF 1ファイル ~16GB | **~47GB（元モデル+分解済 expert）※** |
| セットアップ | GGUF 変換が必要 | **MLX 4bit を直接ロード** |
| API 互換性 | llama.cpp 独自 | **OpenAI 互換** |
| サーバ再起動 | キャッシュ消失 | **KV Cache ディスク永続化** |
| 速度（35B MoE, 24GB） | メモリ不足で動作不可 | **~53 t/s**（総合、プレフィル込み） |

ElfMoon は全 expert を GPU に載せるのではなく、アクティブな expert だけを SSD からストリーミングロードする。ホットな expert は LRU キャッシュ（6144 スロット）に保持。これにより 24GB の Mac でも大規模 MoE モデルを実用的な速度で使える。

2つの動作モード:
- **省メモリモード（既定）**: 実効 6120 スロット ≈ 10.3GB。Xcode など他アプリと共存可能
- **性能モード（`--perf`）**: 実効 8000 スロット ≈ 13.5GB。単体利用で最高速度

※ 元モデル（~19GB）は expert 分解後に削除可能。expert のみで **~28GB** で運用できる。

---

## パフォーマンス（Qwen3.6-35B-A3B, M4 Pro 24GB, 990tok+80tok 生成）

| モード | 合計 throughput | ピークメモリ | 出力品質 |
|--------|:-:|:-:|:--------:|
| **省メモリ（既定 capacity=6144）** | **84.9 t/s** | **12.7 GB** | ✅ |
| **性能（--perf, capacity=8000）** | **87.0 t/s** | **16.0 GB** | ✅ |

- 合計 throughput は 990tok プレフィル＋80tok 生成を含む（初回コールド状態）
- デコード速度は ~15 t/s（コールド, 命中率 ~50%）〜 ~25 t/s（ウォーム, 命中率 90%+）
- KV Cache ディスク復元 ~1s

---

## 動作要件

| 項目 | 要件 |
|---|---|
| ハードウェア | Apple Silicon Mac（M1 以降）、**RAM 24GB 推奨**（常駐容量を下げれば 16GB でも可） |
| OS | macOS 14 以降 |
| Python | 3.10 以降 |
| ディスク空き | 約 47GB（元モデル 19GB ＋ 分解済み expert 28GB。分解後は元モデル削除可） |
| 依存 | MLX / mlx-lm / **transformers==4.57.6**（5.x は mlx-lm と非互換） |

---

## セットアップ

```bash
# 依存
pip install mlx mlx-lm "transformers==4.57.6" huggingface_hub hf_transfer

# モデルダウンロード（Qwen3.6-35B-A3B, MLX 4bit, ~19GB）
HF_HUB_DISABLE_XET=1 hf download mlx-community/Qwen3.6-35B-A3B-4bit \
  --local-dir ./models/qwen3.6-35b-mlx

# expert 分解（40層 × 256 expert = 10240 ファイル）
cd elfmoon
python3 integrate.py split_all ../models/qwen3.6-35b-mlx
```

---

## 使い方

### 対話CLI（手軽に試す）: chat.py

```bash
cd elfmoon
python3 chat.py                                # 省メモリモード（既定、10.3GB）
python3 chat.py --perf                         # 性能モード（13.5GB、単体利用向け）
python3 chat.py 2048                           # 省メモリ（容量指定）
python3 chat.py --no-think                     # 思考プロセスを非表示
python3 chat.py --perf --no-think              # 組合せ
```

- 起動時にモード・実効容量・GB を表示
- モデルを 1 回ロードするだけで対話ループ
- 会話履歴を保持するので続きの相談も可能
- 日本語・英語どちらでも可
- `--no-think` を指定すると think ブロックを非表示にし、回答のみ表示。思考中は `...` が進捗として表示される
- `exit` で終了

**向いている用途**: ちょっとしたコード生成、質問、調査。

### API サーバー（外部ツールから使う）: api_server.py

```bash
ELFMOON_PERF=1 python3 elfmoon/api_server.py           # 性能モード（環境変数）
python3 elfmoon/api_server.py --perf                   # 性能モード（フラグ）
python3 elfmoon/api_server.py                          # 省メモリモード（既定）
python3 elfmoon/api_server.py --no-think               # 思考プロセス非表示
python3 elfmoon/api_server.py 8080 2048                # ポート・常駐容量を指定
# → http://127.0.0.1:11434 で起動
```

引数: `python3 elfmoon/api_server.py [port] [常駐expert数] [--no-think] [--perf]`
環境変数: `ELFMOON_PERF=1`（`--perf` と同等）

> ⚠️ ポート 11434 は Ollama の既定ポートと同じ。Ollama を併用する場合は別ポートを指定すること。

**バインド先と LAN 公開（ELFMOON_HOST）:**

既定では `127.0.0.1`（ローカルのみ）にバインドする。同一 LAN の別マシン（iPad の Textastic、別 PC のエディタ等）から使いたい場合のみ、環境変数 `ELFMOON_HOST` で公開する:

```bash
ELFMOON_HOST=0.0.0.0 python3 elfmoon/api_server.py   # LAN 内の全マシンからアクセス可
```

> ⚠️ **認証機構はない**。`0.0.0.0` で起動すると同一ネットワークの誰でも API を利用できるため、信頼できるネットワーク（自宅 LAN 等）でのみ公開すること。公衆 Wi-Fi では既定の `127.0.0.1` のまま使う。

OpenAI 互換エンドポイント:

| エンドポイント | 用途 |
|---|---|
| `POST /v1/chat/completions` | チャット（stream / non-stream） |
| `GET /v1/models` | モデル一覧 |

**接続設定例:**

```bash
# curl
curl http://localhost:11434/v1/chat/completions \
  -d '{"model":"qwen3.6-35b","messages":[{"role":"user","content":"SwiftでFizzBuzz"}],"stream":true}'

# opencode (~/.config/opencode/opencode.json)
  "model": "elfmoon/qwen3.6-35b",
  "provider": {
    "elfmoon": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "ElfMoon (local)",
      "options": {
        "apiKey": "sk-not-needed",
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "qwen3.6-35b": {
          "name": "Qwen3.6 35B A3B (ElfMoon)",
          "limit": {
            "context": 131072,
            "output": 4096
          }
        }
      }
    }
  }

```

**対応リクエストパラメータ:**

| パラメータ | 既定値 | 備考 |
|---|---|---|
| `messages` | （必須） | system / user / assistant |
| `stream` | `false` | SSE ストリーミング |
| `max_tokens` | 4096 | 上限 4096（超過分は切り詰め） |
| `temperature` | 0.6 | |

- モデル ID は `qwen3.6-35b`（`GET /v1/models` が返す値）。リクエストの `model` 欄は記録用で、値が異なっても動作する
- API キーは不要（ツール側で必須の場合は `sk-not-needed` 等の任意文字列を設定）
- リクエストは 1 件ずつ直列処理される（シングルユーザー前提。同時リクエストは待たされる）

**向いている用途**: opencode, Claude Code等の AI コーディングツールから利用する場合。

### 使い分け

| シチュエーション | 推奨 |
|---|---|
| ちょっとしたコード生成・質問 | chat.py |
| 常駐させて使い回したい | api_server.py |

### 常駐容量の調整

```bash
python3 elfmoon/api_server.py 11434 6144          # 速度重視（既定、10.4GB）
python3 elfmoon/api_server.py 11434 6144 --perf   # 性能モード（200/層, 13.5GB）
python3 elfmoon/api_server.py 11434 2048          # 省メモリ（3.5GB）
```

常駐 expert 数 × 1.69MB がキャッシュメモリ量の目安。減らすとメモリは下がるが命中率が落ちて遅くなる。
`--perf` を付けると 1層あたり 200 slot（実効 8000/6144 GB）に固定され、命中率が向上する。

### KV Cache の保存先とクリア

プロンプトの KV Cache（SSM 状態含む）は `~/.cache/elfmoon/kv_cache/` にディスク永続化される（最大 4 エントリ、古いものから自動削除）。挙動がおかしい場合や容量を空けたい場合は削除してよい:

```bash
rm -rf ~/.cache/elfmoon/kv_cache
```

### テスト・検証ツール

```bash
cd elfmoon
python3 test_kv_manager.py       # KV Cache 永続化のユニットテスト（pytest 不要）
python3 test_slot_cache.py       # スロットキャッシュ退避のユニットテスト（参照用、SlotResidentCache 対象）
python3 test_moe.py              # MoE ブロックの正確性・性能テスト
python3 test_prefetch.py         # 投機プリフェッチのテスト
python3 verify_stream.py         # StreamingMoE と元モデルの層単位一致検証（要: split_all 済み）
python3 integrate.py verify ../models/qwen3.6-35b-mlx   # expert 分解の往復検証
```

---

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| `mlx_lm` の import エラー | `pip install "transformers==4.57.6"`（5.x 非互換） |
| モデル DL が遅い・SHA 不一致 | `HF_HUB_DISABLE_XET=1` を付けて単一接続でダウンロード（多接続 DL ツールは破損の原因） |
| 起動直後にメモリ逼迫 | 常駐容量を下げる（例: `api_server.py 11434 2048`）または `--perf` を外して省メモリモードにする |
| ポート競合（Ollama 併用時） | 別ポートで起動（例: `api_server.py 8080`） |
| 応答品質が急に劣化した | `rm -rf ~/.cache/elfmoon/kv_cache` でキャッシュをクリアして再起動 |

---

## アーキテクチャ

```
入力
  │
  ├─ Router（gate）→ 使用する expert を 8 個選択
  │
  ├─ ResidentCache（6144 experts）にある？
  │    Yes → 直接使用（ホット）
  │    No  → ExpertStore から SSD ロード → キャッシュ投入
  │
  ├─ SharedExpert + StreamingMoE（8 expert 分の計算）
  │
  └─ 次層へ
```

### モジュール

| ファイル | 役割 |
|---|---|
| `api_server.py` | OpenAI 互換 API サーバー |
| `chat.py` | 対話 CLI |
| `stream_model.py` | StreamingMoE + `@mx.compile` デコード |
| `kv_manager.py` | KV Cache メモリ＋ディスク永続化 |
| `expert_store.py` | expert の SSD 保存・ロード |
| `resident_cache.py` | LRU 常駐キャッシュ（トークン予算制御付き） |
| `slot_cache.py` | スロットバッファ方式キャッシュ（参照用、現在未使用） |
| `integrate.py` | モデル → per-expert 分解 |

### 最適化技術

- **`@mx.compile`**: 3 回の量子化 matmul + 活性化関数を GPU 1 カーネルに融合（57% 高速化）
- **Expert グループバッチ**: プレフィル時、同一 expert を使うトークンをまとめて処理（1.7 倍）
- **KV Cache ディスク永続化**: SHA256 ハッシュベースで保存。サーバー再起動後も継続利用可
- **SSM 状態キャッシュ**: Qwen3.6 の SSM 層状態も保存・復元し、部分プレフィルを正しく動作

---

## 対応モデル

| モデル | サイズ | expert 数 | 速度 | 備考 |
|---|---|---|---|---|
| **Qwen3.6-35B-A3B**（推奨） | 19 GB | 10240 | 25 t/s | 思考モード対応、最新 |

---

## ライセンス

Apache License 2.0（[LICENSE](LICENSE) 参照）。モデル本体（Qwen3.6-35B-A3B）のライセンスは配布元（Hugging Face のモデルカード）に従うこと。

---

## クレジット

- 着想元: [antirez/ds4 (DwarfStar)](https://github.com/antirez/ds4)
- 基盤: [MLX](https://github.com/ml-explore/mlx) / [mlx-lm](https://github.com/ml-explore/mlx-lm)
- モデル: [Qwen3.6-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit)（MLX Community / Alibaba Tongyi Lab）

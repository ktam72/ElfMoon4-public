# ElfMoon 🌙

**24GB Apple Silicon で 35B MoE モデルを実用速度で動かす MLX 推論エンジン**

OpenAI 互換 API サーバーと対話 CLI を同梱。Claude Code / Continue.dev / Cursor 等の既存ツールから直接利用できる。

---

## llama.cpp との比較

| | llama.cpp (expert-offload) | **ElfMoon** |
|---|---|---|
| メモリ使用量 | 16 GB（全 expert 常駐） | **6.9 GB**（ストリーミング） |
| 最低要件 | 48GB 以上の Mac | **24GB でも動作** |
| セットアップ | GGUF 変換が必要 | **MLX 4bit を直接ロード** |
| API 互換性 | llama.cpp 独自 | **OpenAI 互換** |
| サーバ再起動 | キャッシュ消失 | **KV Cache ディスク永続化** |
| 速度（35B MoE, 24GB） | メモリ不足で動作不可 | **25 t/s** |

ElfMoon は全 expert を GPU に載せるのではなく、アクティブな expert だけを SSD からストリーミングロードする。ホットな expert は LRU キャッシュ（6144 スロット）に保持。これにより 24GB の Mac でも大規模 MoE モデルを実用的な速度で使える。

---

## パフォーマンス（Qwen3.6-35B-A3B, M4 Pro 24GB）

| 指標 | 値 |
|---|---|
| デコード速度 | **25 t/s**（40ms/token） |
| プレフィル（7k tokens） | ~7s（初回）、~1s（2回目以降） |
| KV Cache ディスク復元 | ~1s |
| 常駐メモリ | 10.4 GB（6144 experts / 10240 中） |

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
python3 chat.py                # 常駐 6144（既定）
python3 chat.py 2048           # 省メモリ
```

- モデルを 1 回ロードするだけで対話ループ
- 会話履歴を保持するので続きの相談も可能
- 日本語・英語どちらでも可
- `exit` で終了

**向いている用途**: ちょっとしたコード生成、質問、調査。

### API サーバー（外部ツールから使う）: api_server.py

```bash
python3 elfmoon/api_server.py
# → http://localhost:11434 で起動
```

OpenAI 互換エンドポイント:

| エンドポイント | 用途 |
|---|---|
| `POST /v1/chat/completions` | チャット（stream / non-stream） |
| `GET /v1/models` | モデル一覧 |

**接続設定例:**

```bash
# curl
curl http://localhost:11434/v1/chat/completions \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"SwiftでFizzBuzz"}],"stream":true}'

# opencode (~/.config/opencode/opencode.json)
  "model": "elfmoon/qwen3.6-35b-a3b",
  "provider": {
    "elfmoon": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "ElfMoon (local)",
      "options": {
        "apiKey": "sk-not-needed",
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "qwen3.6-35b-a3b": {
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

**向いている用途**: opencode, Claude Code等の AI コーディングツールから利用する場合。

### 使い分け

| シチュエーション | 推奨 |
|---|---|
| ちょっとしたコード生成・質問 | chat.py |
| 常駐させて使い回したい | api_server.py |

### 常駐容量の調整

```bash
python3 elfmoon/api_server.py 11434 6144   # 速度重視（既定）
python3 elfmoon/api_server.py 11434 2048   # 省メモリ（3.5GB）
```

---

## アーキテクチャ

```
入力
  │
  ├─ Router（gate）→ 使用する expert を 8 個選択
  │
  ├─ ResidentCache（6144 スロット）にある？
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
| `resident_cache.py` | LRU 常駐キャッシュ |
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

## クレジット

- 着想元: [antirez/ds4 (DwarfStar)](https://github.com/antirez/ds4)
- 基盤: [MLX](https://github.com/ml-explore/mlx) / [mlx-lm](https://github.com/ml-explore/mlx-lm)
- モデル: [Qwen3.6-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit)（MLX Community / Alibaba Tongyi Lab）

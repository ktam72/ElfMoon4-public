# ElfMoon

**24GB Apple Silicon で 35B/80B MoE モデルを実用速度で動かす MLX 推論エンジン**

OpenAI 互換 API サーバーと対話 CLI を同梱。Claude Code / opencode 等のツールから直接利用できる。

ElfMoon は全 expert を GPU に載せるのではなく、アクティブな expert だけを SSD からストリーミングロードする。ホットな expert は LRU キャッシュ（6144 スロット）に保持。

2つの動作モード:
- **省メモリモード（既定）**: 常駐 6144 experts ≈ 10.4GB。Xcode など他アプリと共存可能
- **性能モード（`--perf`）**: 常駐 8000 experts ≈ 13.5GB。単体利用で最高速度

---

## 他エンジンとの比較

ElfMoon4 の立ち位置を明確にするため、類似エンジンとの比較を示す。

| 観点 | llama.cpp | DwarfStar4 | **ElfMoon4** |
|------|-----------|------------|--------------|
| **対象モデル** | 汎用 GGUF（Llama / Mistral / Qwen / DeepSeek 等） | DeepSeek V4 Flash (284B) / PRO 特化 | **Qwen3.6-35B-A3B / Qwen3-Next-80B-A3B 特化** |
| **メモリ目標** | モデルサイズ次第（GGUF 量子化で調整） | 96GB+（ハイエンド Mac / Linux） | **24GB**（Xcode 同時起動可） |
| **言語 / 基盤** | C/C++ + 各種 GPU バックエンド | C + Metal / CUDA / ROCm | **Python + MLX** |
| **ビルド** | cmake / make（要コンパイル） | make（要コンパイル） | **pip install のみ**（インタプリタ） |
| **モデル形式** | GGUF（汎用フォーマット） | GGUF（DS4 独自レイアウト） | **MLX safetensors**（標準 MLX 形式） |
| **量子化** | 多種（Q2〜Q8、IQ シリーズ等） | 非対称 IQ2_XXS / Q2_K（2bit） | **Q4 group64 統一 + router 8bit** |
| **SSD Streaming** | レイヤー単位（mmap × partial load） | **エキスパート単位**（per-expert cache） | **エキスパート単位**（per-expert LRU cache） |
| **アーキテクチャ** | 汎用 Transformer / MoE | 純粋 Transformer MoE | **ハイブリッド対応**（線形 attention + full attention） |
| **分散推論** | 対応 | **マルチマシン TCP 分割** | シングルマシンのみ |
| **エージェント連携** | 外部 API 経由 | **ネイティブ DSML エージェント内蔵** | OpenAI API 経由の外部連携 |
| **API 互換性** | OpenAI 互換サーバー同梱 | OpenAI / Anthropic / Responses 対応 | **OpenAI 互換**（`/v1/chat/completions`） |
| **HW バックエンド** | Metal / CUDA / Vulkan / SYCL / etc. | Metal / CUDA / ROCm | **MLX のみ**（Metal 内部利用） |
| **エキスパートグループ化プリフィル** | なし | なし | **○（148 t/s、24倍高速化）** |
| **`@mx.compile` JIT デコード** | なし | なし | **○（15-25 t/s デコード）** |
| **Eco / Perf デュアルモード** | なし | なし | **○（~10.4GB / ~13.5GB 切替）** |
| **80B モデル対応** | 対応（要十分な RAM） | 対応（284B 対応、要 96GB+） | **○（実験的、24GB で動作）** |
| **コード規模** | ~200,000行 C++ | ~30,000行 C + GPU カーネル | **~2,000行 Python** |
| **移植性** | ほぼ全ての環境 | Mac + Linux (CUDA/ROCm) | **Apple Silicon 専用** |
| **ライセンス** | MIT | MIT + GGML attribution | **Apache 2.0** |

### ElfMoon4 の独自性（3 エンジン中最も特長的な点）

- **最小メモリフットプリント**: llama.cpp / DS4 が 96GB+ を要求する中、**24GB で 35B/80B MoE モデルを実用速度で動作させる唯一のエンジン**。Xcode など他アプリとの同時起動も想定。
- **Python/MLX スタック**: コンパイル不要で即利用可能。C/C++ エンジンにはない `@mx.compile` JIT や MLX の自動 Metal ディスパッチを活用。
- **ハイブリッドアーキテクチャ対応**: Qwen3.6 の GatedDeltaNet（線形 attention）と full attention の混在をサポート。純粋 Transformer のみの DS4 や llama.cpp の汎用 MoE とは異なるレイヤーを扱える。
- **エキスパートグループ化プリフィル**: プリフィル時に同一エキスパートにルーティングされたトークンをまとめて計算。DS4 や llama.cpp にない独自最適化で 24 倍のプリフィル高速化を達成。
- **実環境での実証**: M4 Pro 24GB で Xcode 動作中に 12.6 t/s を達成する等、「研究室内のベンチマーク」ではなく**実際の開発ワークフローで使える**ことを検証済み。

---

## パフォーマンス（M4 Pro 24GB）

| モデル | デコード t/s | ピークメモリ |
|--------|:-:|:-:|
| **Qwen3.6-35B-A3B**（推奨） | ~15-25 | 12.7 GB |
| **Qwen3-Next-80B-A3B**（実験的） | ~9.7 | 12.9 GB |

35B はウォームで ~25 t/s、80B はコールド/ウォーム共に計算律速で ~9.7 t/s。

---

## 動作要件

| 項目 | 要件 |
|---|---|
| ハードウェア | Apple Silicon Mac（M1 以降）、**RAM 24GB 推奨**（容量を下げれば 16GB でも可） |
| OS | macOS 14 以降 |
| Python | 3.10 以降 |
| ディスク空き | 35B: ~47GB / 80B: ~84GB（元モデル + 分解済 expert） |
| 依存 | MLX / mlx-lm / **transformers==4.57.6**（5.x は非互換） |

---

## セットアップ

```bash
# 依存
pip install mlx mlx-lm "transformers==4.57.6" huggingface_hub hf_transfer

# モデルダウンロード（Qwen3.6-35B-A3B, MLX 4bit, ~19GB）← 推奨
HF_HUB_DISABLE_XET=1 hf download mlx-community/Qwen3.6-35B-A3B-4bit \
  --local-dir ./models/qwen3.6-35b-mlx

# expert 分解（40層 × 256 expert = 10240 ファイル）
cd elfmoon
python3 integrate.py split_all ../models/qwen3.6-35b-mlx
```

### 80B モデル（オプション）

```bash
HF_HUB_DISABLE_XET=1 hf download mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit \
  --local-dir ./models/qwen3-next-80b-mlx

cd elfmoon
python3 integrate.py split_all ../models/qwen3-next-80b-mlx spike/real_store_80b

export ELFMOON_MODEL_DIR=../models/qwen3-next-80b-mlx
export ELFMOON_STORE_DIR=spike/real_store_80b
python3 chat.py
```

### Coder-Next モデル（オプション、コード特化）

80B と同一トポロジ（hidden2048 / 48層 / 512expert / top_k10）のため、`stream_model.py` / `integrate.py` は無改造で動作する。

```bash
HF_HUB_DISABLE_XET=1 hf download mlx-community/Qwen3-Coder-Next-4bit \
  --local-dir ./models/qwen3-coder-next-4bit

cd elfmoon
python3 integrate.py split_all ../models/qwen3-coder-next-4bit spike/real_store_coder spike/real_gates_coder

export ELFMOON_MODEL_DIR=../models/qwen3-coder-next-4bit
export ELFMOON_STORE_DIR=spike/real_store_coder
export ELFMOON_GATE_DIR=spike/real_gates_coder
python3 chat.py
```

> ⚠️ 配布物の `tokenizer_config.json` は `extra_special_tokens` が list 形式で、transformers 4.57.6 は dict `{name: token}` を要求するため読み込みエラーになる。ダウンロード後、モデルディレクトリ内の `tokenizer_config.json` を以下で変換する（元ファイルは `tokenizer_config.json.orig` に退避、重み本体は無傷）:
> ```bash
> cd models/qwen3-coder-next-4bit
> cp tokenizer_config.json tokenizer_config.json.orig
> python3 -c "
> import json
> c = json.load(open('tokenizer_config.json'))
> lst = c['extra_special_tokens']
> c['extra_special_tokens'] = {t.strip('<|>').replace('/','_'): t for t in lst}
> json.dump(c, open('tokenizer_config.json','w'), ensure_ascii=False, indent=2)
> "
> ```

---

## 使い方

### 対話CLI: chat.py

```bash
cd elfmoon
python3 chat.py                                # 省メモリモード（既定）
python3 chat.py --perf                         # 性能モード
python3 chat.py 2048                           # 省メモリ（容量指定）
python3 chat.py --no-think                     # 思考プロセスを非表示
```

- 起動時にモード・実効容量・GB を表示
- モデルを 1 回ロードするだけで対話ループ。`exit` で終了
- 日本語・英語どちらでも可

### API サーバー: api_server.py

```bash
python3 elfmoon/api_server.py                  # 省メモリモード（既定）
python3 elfmoon/api_server.py --perf           # 性能モード
python3 elfmoon/api_server.py 8080 2048        # ポート・常駐容量を指定
# → http://127.0.0.1:11434 で起動
```

引数: `python3 elfmoon/api_server.py [port] [常駐expert数] [--no-think] [--perf]`
環境変数: `ELFMOON_PERF=1`（`--perf` と同等）

OpenAI 互換エンドポイント:
| エンドポイント | 用途 |
|---|---|
| `POST /v1/chat/completions` | チャット（stream / non-stream） |
| `GET /v1/models` | モデル一覧 |

**接続例:**
```bash
curl http://localhost:11434/v1/chat/completions \
  -d '{"model":"qwen3.6-35b","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

対応パラメータ: `messages`（必須）、`stream`、`max_tokens`（上限4096）、`temperature`（既定0.6）。
API キーは不要（必須のツールでは `sk-not-needed` 等を設定）。

> ⚠️ ポート 11434 は Ollama の既定ポートと同じ。併用時は別ポートを指定。

**バインド先（ELFMOON_HOST）:**
既定は `127.0.0.1`（ローカルのみ）。LAN 公開する場合のみ設定:
```bash
ELFMOON_HOST=0.0.0.0 python3 elfmoon/api_server.py
```
> ⚠️ 認証機構はない。信頼できるネットワークでのみ公開。

### 常駐容量の調整

```bash
python3 elfmoon/api_server.py 11434 6144          # 既定、10.4GB
python3 elfmoon/api_server.py 11434 6144 --perf   # 性能モード、13.5GB
python3 elfmoon/api_server.py 11434 2048          # 省メモリ、3.5GB
```

常駐 expert 数 × 1.69MB がキャッシュメモリ量の目安。`--perf` で最大 8000 experts（≈13.5GB）。

### KV Cache クリア

```bash
rm -rf ~/.cache/elfmoon/kv_cache
```

---

## テスト・検証

```bash
cd elfmoon
python3 verify_stream.py          # StreamingMoE と元モデルの一致検証
python3 test_moe.py               # MoE ブロックのテスト
python3 test_kv_manager.py        # KV Cache 永続化のテスト
python3 integrate.py verify ../models/qwen3.6-35b-mlx   # 分解の往復検証
```

---

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| `mlx_lm` の import エラー | `pip install "transformers==4.57.6"`（5.x 非互換） |
| モデル DL が遅い・SHA 不一致 | `HF_HUB_DISABLE_XET=1` で単一接続 DL |
| メモリ逼迫 | 常駐容量を下げる（例: `api_server.py 11434 2048`） |
| ポート競合（Ollama） | 別ポートで起動（例: `api_server.py 8080`） |
| 応答品質が急に劣化 | `rm -rf ~/.cache/elfmoon/kv_cache` でキャッシュクリア |

---

## 対応モデル

| モデル | サイズ | expert 数 | デコード t/s | 備考 |
|---|---|---|---|---|
| **[Qwen3.6-35B-A3B](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit)**（推奨） | 19 GB | 10240 | ~15-25 | 思考モード対応、高速 |
| **[Qwen3-Next-80B-A3B](https://huggingface.co/mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit)**（実験的） | 42 GB | 24576 | ~9.7 | 品質重視向け、環境変数切替 |
| **[Qwen3-Coder-Next](https://huggingface.co/mlx-community/Qwen3-Coder-Next-4bit)**（実験的） | 42 GB | 24576 | ~8-10 | コード特化、tokenizer_config.json 要変換 |

---

## ライセンス

Apache License 2.0。モデル本体のライセンスは配布元のモデルカードに従うこと。

## クレジット

- 着想元: [antirez/ds4 (DwarfStar)](https://github.com/antirez/ds4)
- 基盤: [MLX](https://github.com/ml-explore/mlx) / [mlx-lm](https://github.com/ml-explore/mlx-lm)

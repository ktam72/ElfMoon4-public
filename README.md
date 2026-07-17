# ElfMoon4🌙

> 24GB RAM の Apple Silicon Mac 上で 26B/35B/80BクラスのLLMを Streaming MoE あるいは オンメモリで実用速度で動かす MLX 推論エンジン（M4 Pro MacBookでのみ動作確認済み）

OpenAI 互換 API サーバーと対話 CLI を同梱。Claude Code / opencode 等のツールからも直接利用できる。

ElfMoon は全 expert を GPU に載せるのではなく、アクティブな expert だけを SSD からストリーミングロードする（Qwen3.6-35B / Qwen3-Next-80B 等の大規模 MoE 向け）。
ホットな expert は LRU キャッシュ（6144 スロット）に保持。

一方 Gemma4 / DeepSeek-R1-Distill / GLM / Bonsai 等の通常モデルは `mlx_lm` 経由で動作する。


### ElfMoon4 の独自性

- **デュアルモード推論**: ストリーミング MoE（大規模 expert 分解モデル）と `mlx_lm` 経由のオンメモリモデルの両方を同じ `chat.py` / `api_server.py` で透過的に扱える。
- **ストリーミング MoE**: 全 expert を GPU に載せず、アクティブな expert だけを SSD から LRU キャッシュにストリーミング。**24GB で 80B+ MoE モデルを実用速度で動作**。Xcode など他アプリとの同時起動も想定。
- **Python/MLX スタック**: コンパイル不要で即利用可能。C/C++ エンジンにはない `@mx.compile` JIT や MLX の自動 Metal ディスパッチを活用。Gemma4 系で最大 85 tok/s を達成。
- **エキスパートグループ化プリフィル**: プリフィル時に同一エキスパートにルーティングされたトークンをまとめて計算。DS4 や llama.cpp にない独自最適化で 24 倍のプリフィル高速化を達成。
- **広範なモデル対応**: Qwen3.6 (MoE+dense) / Qwen3-Next-80B / Qwen3-Coder-Next / Gemma4 / GLM-4.7 / DeepSeek-R1-Distill / Bonsai 等、MLX エコシステムの多様なモデルを同一インターフェースで利用可能。

---

## 動作要件

| 項目 | 要件 |
|---|---|
| ハードウェア | Apple Silicon Mac（M1 以降）、**RAM 24GB 推奨**（メモリ使用オプションを下げれば 16GB でも可のはず） |
| OS | macOS 14 以降 （26.5以降推奨）|
| Python | 3.10 以降 |
| ディスク空き | 35B: ~47GB / 80B・Coder-Next: ~84GB（元モデル + 分解済 expert、`ELFMOON_MODELS_ROOT` 配下） |
| 依存 | MLX / mlx-lm / **transformers==4.57.6**（5.x は非互換） |

---
## 動作確認済みモデル

| モデル | タイプ | ファイルサイズ | デコード t/s | 備考 |
|---|---|---|---|---|
| **[Gemma4-26B-A4B-it-4bit](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-4bit)**（最推奨） | オンメモリ | 15 GB | **72** | `mx.compile` で 5×高速化。品質・速度の最適バランス |
| **[Gemma4-Heretic](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-heretic-4bit)** | オンメモリ | 15.6 GB | **65** | Heretic 変種 |
| **[GLM-4.7-Flash](https://huggingface.co/mlx-community/GLM-4.7-Flash-4bit)** | オンメモリ | 16.9 GB | **61** | Zhipu 製、日本語可、高速 |
| **[Qwen3.6-35B-A3B](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit)**（推奨） | ストリーミング MoE | 19 GB | **37** | 思考モード対応、省メモリ |
| **[Qwen3.6-35B-HauhauCS](https://huggingface.co/dawncr0w/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-OptiQ-4bit-MLX)**（実験的） | ストリーミング MoE | 18 GB | **36** | 完全アンセンサード |
| **[Ornith-1.0-35B](https://huggingface.co/mlx-community/Ornith-1.0-35B-4bit)** | ストリーミング MoE | 37 GB | **35** | エージェンティックコーディング特化 |
| **[Qwen3.6-35B-Heretic](https://huggingface.co/froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit)**（実験的） | ストリーミング MoE | 19 GB | **35** | Heretic 変種 |
| **[Bonsai-27B-2bit](https://huggingface.co/mlx-community/Ternary-Bonsai-27B-2bit)** | オンメモリ | 8.5 GB | **24** | 2bit ternary、軽量 |
| **[Qwen3-Next-80B](https://huggingface.co/mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit)**（実験的） | ストリーミング MoE | 42 GB | **25** | 品質重視 80B |
| **[Qwen3-Coder-Next](https://huggingface.co/mlx-community/Qwen3-Coder-Next-4bit)** | ストリーミング MoE | 42 GB | **22** | コード特化 |
| **[Qwen3.6-27B](https://huggingface.co/mlx-community/Qwen3.6-27B-4bit)** | オンメモリ | 15 GB | **15** | dense 27B |
| **[Qwen3.5-REAP-97B](https://huggingface.co/mlx-community/Qwen3.5-REAP-97B-A10B-4bit)**（非推奨） | ストリーミング MoE | 51 GB | — | capacity 要大幅減 |
| **[DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)** | — | — | — | 24GB では未対応 |

---

## セットアップ

### AIモデルファイル置き場（環境変数：ELFMOON_MODELS_ROOT）

ElfMoon 本体は、環境変数`ELFMOON_MODELS_ROOT`に指定されたディレクトリをAIモデルファイル参照場所として認識する。

```
<ELFMOON_MODELS_ROOT>/
  <モデル名>/
    config.json, *.safetensors, tokenizer...   ← ダウンロードした元モデル
    store/                                      ← integrate.py split_all が自動生成
```

という規約でモデルを1つ1つ独立したディレクトリとして置くだけでよい。

> ⚠️ **初回セットアップ必須**: `ELFMOON_MODELS_ROOT` にAIモデルファイル置き場のパスを設定すること（シェルの起動ファイルに恒久登録推奨。外付けSSD推奨）。未設定時は `./models`（リポジトリ直下）にフォールバックするが空なので、モデルが1つも見つからずロードに失敗する。
> ```bash
> echo 'export ELFMOON_MODELS_ROOT=/path/to/your/models' >> ~/.zshrc
> source ~/.zshrc
> ```
>


2種類のモデル形式はディレクトリ構成から自動判別:
- **オンメモリモード（`mlx_lm` 経由）**: `store/` ディレクトリがない通常モデル。全重みをメモリにロードし `mx.compile` で高速化（Gemma4 は 70+ tok/s）
- **ストリーミング MoE**: `store/` ディレクトリがある分解済み MoE モデル。expert を SSD から LRU キャッシュにストリーミング


追加・削除は該当ディレクトリの追加・`rm -rf`のみ。外部SSD等どこに置いてもよく、指すのは環境変数1つだけ:　以下例。

```bash
export ELFMOON_MODELS_ROOT=/Volumes/990Pro_2TB/elfmoon/models   # 例: 外部SSD
# 未設定時は ./models（リポジトリ直下）が既定
```

### 依存ファイルのインストール
```bash
# 依存
pip install mlx mlx-lm "transformers==4.57.6" huggingface_hub hf_transfer
```



### ストリーミング MoE モデル（大規模 MoE）

`integrate.py split_all` で expert 分解が必要（`store/` ディレクトリを生成）。

```bash
# Qwen3.6-35B（推奨 MoE、16-22 tok/s）
HF_HUB_DISABLE_XET=1 hf download mlx-community/Qwen3.6-35B-A3B-4bit \
  --local-dir $ELFMOON_MODELS_ROOT/qwen3.6-35b-mlx
python3 elfmoon/integrate.py split_all $ELFMOON_MODELS_ROOT/qwen3.6-35b-mlx
python3 elfmoon/chat.py --model qwen3.6-35b-mlx

# Qwen3-Next-80B（実験的）
HF_HUB_DISABLE_XET=1 hf download mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit \
  --local-dir $ELFMOON_MODELS_ROOT/qwen3-next-80b-mlx
python3 elfmoon/integrate.py split_all $ELFMOON_MODELS_ROOT/qwen3-next-80b-mlx
python3 elfmoon/chat.py --model qwen3-next-80b-mlx

# Ornith-1.0-35B（エージェンティックコーディング）
HF_HUB_DISABLE_XET=1 hf download mlx-community/Ornith-1.0-35B-4bit \
  --local-dir $ELFMOON_MODELS_ROOT/ornith-1.0-35b-mlx
python3 elfmoon/integrate.py split_all $ELFMOON_MODELS_ROOT/ornith-1.0-35b-mlx
python3 elfmoon/chat.py --model ornith-1.0-35b-mlx --perf

# Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive（完全アンセンサード）
HF_HUB_DISABLE_XET=1 hf download dawncr0w/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-OptiQ-4bit-MLX \
  --local-dir $ELFMOON_MODELS_ROOT/qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive-mlx
python3 elfmoon/integrate.py split_all $ELFMOON_MODELS_ROOT/qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive-mlx
python3 elfmoon/chat.py --model qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive-mlx

# Qwen3-Coder-Next（コード特化）
HF_HUB_DISABLE_XET=1 hf download mlx-community/Qwen3-Coder-Next-4bit \
  --local-dir $ELFMOON_MODELS_ROOT/qwen3-coder-next-4bit
python3 elfmoon/integrate.py split_all $ELFMOON_MODELS_ROOT/qwen3-coder-next-4bit
python3 elfmoon/chat.py --model qwen3-coder-next-4bit
```

> ⚠️ Coder-Next / Qwen3.5-REAP の `tokenizer_config.json` は `extra_special_tokens` が list 形式で、transformers 4.57.6 は dict を要求する。以下の変換が必要:
> ```bash
> cd $ELFMOON_MODELS_ROOT/qwen3-coder-next-4bit
> cp tokenizer_config.json tokenizer_config.json.orig
> python3 -c "
> import json
> c = json.load(open('tokenizer_config.json'))
> lst = c['extra_special_tokens']
> c['extra_special_tokens'] = {t.strip('<|>').replace('/','_'): t for t in lst}
> json.dump(c, open('tokenizer_config.json','w'), ensure_ascii=False, indent=2)
> "
> ```



### オンメモリモデル（推奨: Gemma4-26B）

分解不要。`store/` ディレクトリがなくてもそのまま動作する。ダウンロードのみで完了:

```bash
HF_HUB_DISABLE_XET=1 hf download mlx-community/gemma-4-26B-A4B-it-4bit \
  --local-dir $ELFMOON_MODELS_ROOT/gemma-4-26b-a4b-it-4bit

python3 elfmoon/chat.py --model gemma-4-26b-a4b-it-4bit
```

> Gemma4 は `mx.compile` により約 5× 高速化（70-85 tok/s）、品質・速度・メモリの最適バランス。**最も推奨するモデル。**

### Heretic / アブリテイテッド変種

通常版と同一手順でダウンロードするだけ:

```bash
# Gemma4 Heretic 変種（同一速度・メモリ）
HF_HUB_DISABLE_XET=1 hf download mlx-community/gemma-4-26B-A4B-it-heretic-4bit \
  --local-dir $ELFMOON_MODELS_ROOT/gemma-4-26b-a4b-it-heretic-4bit

python3 elfmoon/chat.py --model gemma-4-26b-a4b-it-heretic-4bit

# Qwen3.6-35B Heretic 変種（ストリーミング MoE、分解必須）
HF_HUB_DISABLE_XET=1 hf download froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit \
  --local-dir $ELFMOON_MODELS_ROOT/qwen3.6-35b-uncensored-heretic-mlx
python3 elfmoon/integrate.py split_all $ELFMOON_MODELS_ROOT/qwen3.6-35b-uncensored-heretic-mlx

python3 elfmoon/chat.py --model qwen3.6-35b-uncensored-heretic-mlx
```

### GLM / Bonsai / DeepSeek-R1 等

すべて分解不要。`hf download` で `ELFMOON_MODELS_ROOT/<モデル名>` にダウンロードし、`--model` で指定するだけ:

```bash
HF_HUB_DISABLE_XET=1 hf download mlx-community/GLM-4.7-Flash-4bit \
  --local-dir $ELFMOON_MODELS_ROOT/glm-4.7-flash-4bit

HF_HUB_DISABLE_XET=1 hf download mlx-community/Ternary-Bonsai-27B-2bit \
  --local-dir $ELFMOON_MODELS_ROOT/ternary-bonsai-27b-2bit

HF_HUB_DISABLE_XET=1 hf download mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit \
  --local-dir $ELFMOON_MODELS_ROOT/deepseek-r1-distill-qwen-14b-4bit
```

### Qwen3.5-REAP-97B-A10B（非推奨）

REAP 刈込で 97B まで圧縮されているが active パラメータは 10B 級で decode ~3.4 t/s と実用ラインを下回る。

```bash
HF_HUB_DISABLE_XET=1 hf download mlx-community/Qwen3.5-REAP-97B-A10B-4bit \
  --local-dir $ELFMOON_MODELS_ROOT/qwen3.5-reap-97b-4bit
python3 elfmoon/integrate.py split_all $ELFMOON_MODELS_ROOT/qwen3.5-reap-97b-4bit

# expert サイズが大きい（~5.3MB/個）ため capacity を明示的に下げること
python3 elfmoon/chat.py --model qwen3.5-reap-97b-4bit 1900
```

> ⚠️ `tokenizer_class` が `TokenizersBackend` になっている。`PreTrainedTokenizerFast` に書き換えが必要:
> ```bash
> cd $ELFMOON_MODELS_ROOT/qwen3.5-reap-97b-4bit
> cp tokenizer_config.json tokenizer_config.json.orig
> python3 -c "
> import json; c = json.load(open('tokenizer_config.json'))
> c['tokenizer_class'] = 'PreTrainedTokenizerFast'
> json.dump(c, open('tokenizer_config.json','w'), ensure_ascii=False, indent=2)
> "
> ```

---

## 使い方

### 対話CLI: chat.py

```bash
python3 elfmoon/chat.py                                # 既定モデル（$ELFMOON_MODELS_ROOT 直下から自動選択）
python3 elfmoon/chat.py --model gemma-4-26b-a4b-it-4bit   # オンメモリモデル
python3 elfmoon/chat.py --model qwen3.6-35b-mlx            # ストリーミング MoE
python3 elfmoon/chat.py --perf                             # 性能モード（ストリーミング MoE 時）
python3 elfmoon/chat.py 2048                               # 常駐 expert 数指定（ストリーミング MoE 時）
python3 elfmoon/chat.py --no-think                          # 思考プロセスを非表示
python3 elfmoon/chat.py --list                              # モデル一覧
```

> ⚠️ モデルは `ELFMOON_MODELS_ROOT/<モデル名>/` 以下に置くだけで自動認識される（台帳ファイル不要）。

- 起動時にモデルパス・モード・実効容量・GB を表示
- モデルを 1 回ロードするだけで対話ループ。`exit` で終了
- 日本語・英語どちらでも可
- オンメモリモデル（`store/` なし）は `--perf` / expert 数指定が不要。`store/` のある MoE モデルは従来通り LRU キャッシュで容量調整可能

### API サーバー: api_server.py

```bash
python3 elfmoon/api_server.py                          # 省メモリモード（既定モデル）
python3 elfmoon/api_server.py --model gemma-4-26b-a4b-it-4bit   # オンメモリモデル
python3 elfmoon/api_server.py --model qwen3-next-80b-mlx         # ストリーミング MoE
python3 elfmoon/api_server.py --perf                             # 性能モード（ストリーミング MoE 時）
python3 elfmoon/api_server.py 8080 2048                          # ポート・常駐容量を指定
python3 elfmoon/api_server.py --list                             # モデル一覧
# → http://127.0.0.1:11434 で起動
```

引数: `python3 elfmoon/api_server.py [port] [常駐expert数] [--model NAME] [--no-think] [--perf]`
環境変数: `ELFMOON_PERF=1`（`--perf` と同等）、`ELFMOON_MODELS_ROOT`（モデル置き場）
- オンメモリモデル（`store/` なし）は `--perf` / expert 数指定が不要

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

### 常駐容量の調整（ストリーミング MoE モデルのみ）

```bash
python3 elfmoon/api_server.py 11434 6144          # 既定、10.4GB
python3 elfmoon/api_server.py 11434 6144 --perf   # 性能モード、13.5GB
python3 elfmoon/api_server.py 11434 2048          # 省メモリ、3.5GB
```

常駐 expert 数 × 1.69MB（Qwen の場合）がキャッシュメモリ量の目安。モデルにより expert サイズは異なる（1.7〜5.3MB）。`--perf` で最大 8000 experts（≈13.5GB）。
オンメモリモデルでは指定不要。

### KV Cache クリア / 保存先変更

```bash
# クリア
rm -rf ~/.cache/elfmoon/kv_cache
# または環境変数で指定した場所
rm -rf "$ELFMOON_KV_CACHE_DIR"
```

**保存先の変更**: `ELFMOON_KV_CACHE_DIR` 環境変数で任意のディレクトリを指定可能。

```bash
# 外部 SSD に保存（Macintosh HD の逼迫回避）
export ELFMOON_KV_CACHE_DIR=/Volumes/990Pro_2TB/elfmoon/kv_cache
python3 elfmoon/api_server.py
```

未設定時は `~/.cache/elfmoon/kv_cache`（既定）。

---
# その他

## テスト・検証

```bash
cd elfmoon
python3 verify_stream.py          # StreamingMoE と元モデルの一致検証（ELFMOON_MODEL で対象切替）
python3 test_kv_manager.py        # KV Cache 永続化のテスト
python3 integrate.py verify $ELFMOON_MODELS_ROOT/qwen3.6-35b-mlx /tmp/elfmoon_verify_test   # 分解の往復検証
```

`integrate.py verify` は `store_dir`（第3引数）が必須。検証用の使い捨てディレクトリ（例: `/tmp/elfmoon_verify_test`）を明示的に指定し、確認後に削除する。既定値を持たせていないのは、過去に固定パスの既定値が原因で別モデルの本番 store を誤って上書きした事故があったため（`--model` で複数モデルを切り替える運用と相性が悪い）。

---

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| `mlx_lm` の import エラー | `pip install "transformers==4.57.6"`（5.x 非互換） |
| モデル DL が遅い・SHA 不一致 | `HF_HUB_DISABLE_XET=1` で単一接続 DL |
| メモリ逼迫 / Metal OOM | 常駐容量を下げる（例: `api_server.py 11434 2048`）。expertサイズはモデルにより異なる（1.7〜5.3MB）ため、大きいexpertのモデルは既定値6144だと溢れることがある |
| ポート競合（Ollama） | 別ポートで起動（例: `api_server.py 8080`） |
| 応答品質が急に劣化 | `rm -rf ~/.cache/elfmoon/kv_cache` でキャッシュクリア |
| `--model NAME` でモデルが見つからない | `python3 elfmoon/chat.py --list` で `ELFMOON_MODELS_ROOT` 配下の認識状況を確認 |
| HauhauCS Aggressive 最終層の shared expert 不一致 | OptiQ 量子化の artifact（自動検出・スキップ済み、実用影響なし） |

---


## パフォーマンス（M4 Pro 24GB, warm A/B, 120 tokens, 2プロンプト平均）

| モデル | タイプ | gen t/s | decode t/s | ピークメモリ |
|--------|--------|:-:|:-:|:-:|
| **Gemma4-26B-A4B-it-4bit** | オンメモリ | **69** | **72** | 14.2 GB |
| **Gemma4-26B-A4B-it-heretic-4bit** | オンメモリ | **63** | **65** | 14.5 GB |
| **GLM-4.7-Flash** | オンメモリ | **60** | **61** | 16.9 GB |
| **Bonsai-27B-2bit** | オンメモリ | **24** | **24** | 7.6 GB |
| **Qwen3.6-35B**（推奨） | ストリーミング MoE | 20 | **37** | 12.1 GB |
| **Ornith-1.0-35B** | ストリーミング MoE | 17 | **35** | 12.1 GB |
| **Qwen3.6-35B-HauhauCS** | ストリーミング MoE | 19 | **36** | 12.1 GB |
| **Qwen3.6-35B-Heretic** | ストリーミング MoE | 16 | **35** | 12.1 GB |
| **Qwen3-Next-80B**（実験的） | ストリーミング MoE | 12 | **25** | 12.2 GB |
| **Qwen3-Coder-Next** | ストリーミング MoE | 10 | **22** | 12.2 GB |
| **Qwen3.6-27B** | オンメモリ | 15 | **15** | 15.1 GB |

- 計測条件: subprocess でモデル切替、warm 8tok + generate 120tok、2種類のプロンプト平均
- gen t/s: stream_generate 合計（prefill+decode） / decode t/s: デコードのみ（per-step loop）
- ストリーミング MoE 35B 級は decode 35-37 t/s で一律。80B 級は decode 22-25 t/s
- オンメモリモデルは `mx.compile` による JIT コンパイル（Gemma4/GLM は 5× 高速化）
- Gemma4 は品質・速度の最適バランス。ストリーミング MoE は低メモリで大規模モデル動作

---

## ディレクトリ構成

プログラム本体（このリポジトリ）とモデルの実体（`ELFMOON_MODELS_ROOT`）は完全に分離されている。前者にはモデルの重みや分解済みexpertは一切含まれない。

### プログラム本体（このリポジトリ）

```
ElfMoon4/
├── README.md
├── models/                    # ELFMOON_MODELS_ROOT 未設定時のフォールバック先（既定は空）
├── elfmoon/                   # 実装本体
│   ├── stream_model.py        #   StreamingMoE 本体、モデル名解決（resolve_model/list_models）
│   ├── chat.py                #   対話CLI（--model/--list対応）
│   ├── api_server.py          #   OpenAI互換APIサーバ（--model/--list対応）
│   ├── integrate.py           #   モデル分解ツール（split_all で <model_dir>/store/ を生成）
│   ├── expert_store.py        #   ExpertStore（分解済みexpertの読み込み）
│   ├── resident_cache.py      #   LRU常駐キャッシュ
│   ├── kv_manager.py          #   KV Cache永続化（~/.cache/elfmoon/kv_cache）
│   ├── verify_stream.py       #   元モデルとの層単位一致検証
│   ├── test_*.py              #   ユニットテスト
│   ├── bench/                 #   ベンチマークスクリプト
│   └── spike/                 #   プロトタイプ・レガシー検証フィクスチャ（本番モデルとは無関係）
├── docs/                      # 設計ドキュメント・DeepSeek引き継ぎ文書
├── evidence/                  # 計測結果の記録
└── ref-ds4/                   # 参考実装（DwarfStar4、着想元。ElfMoon自体の依存ではない）
```

### モデルの実体（`ELFMOON_MODELS_ROOT` 配下）

外部SSD等どこに置いてもよい。プログラム本体とはこの環境変数1本のみで疎結合になっている。各モデルは自己完結したディレクトリで、追加・削除は該当ディレクトリの追加・`rm -rf`のみで完結する。

```
$ELFMOON_MODELS_ROOT/
├── gemma-4-26b-a4b-it-4bit/          # オンメモリ（store/ なし）
│   ├── config.json, *.safetensors, tokenizer...
├── gemma-4-26b-a4b-it-heretic-4bit/  # オンメモリ heretic 変種
│   ├── config.json, *.safetensors, tokenizer...
├── glm-4.7-flash-4bit/               # オンメモリ
│   ├── config.json, *.safetensors...
├── ternary-bonsai-27b-2bit/          # オンメモリ
│   ├── config.json, *.safetensors...
├── deepseek-r1-distill-qwen-32b-japanese-4bit/
├── deepseek-r1-distill-qwen-14b-4bit/
├── qwen3.6-35b-mlx/                  # ストリーミング MoE（store/ あり）
│   ├── config.json, *.safetensors...
│   └── store/                        # integrate.py split_all が自動生成
│       ├── l0_e0.safetensors         # 層l・expert eごとに1ファイル
│       ├── l0_e1.safetensors
│       └── ...
├── qwen3.6-35b-uncensored-heretic-mlx/
│   ├── config.json, *.safetensors...
│   └── store/
├── qwen3-next-80b-mlx/
│   ├── config.json, ...
│   └── store/
├── qwen3-coder-next-4bit/
│   ├── config.json, ...
│   └── store/
└── qwen3.5-reap-97b-4bit/
    ├── config.json, ...
    └── store/
```

- `config.json` の有無で `python3 elfmoon/chat.py --list` がモデルとして認識するかを判定する（台帳ファイルは存在しない＝規約ベースの自動検出）
- `store/` が無い通常モデル（オンメモリ）はそのまま動作。`store/` がある MoE モデルのみ `integrate.py split_all` による分解が必須（`--list` は `⚠️ store/ 未生成` と表示）
- モデルと store は同一ディレクトリ配下にあるため、ディレクトリごと `mv` すれば対応関係が壊れずに移動できる

---

## ライセンス

Apache License 2.0。モデル本体のライセンスは配布元のモデルカードに従うこと。

## クレジット

- 着想元: [antirez/ds4 (DwarfStar)](https://github.com/antirez/ds4)
- 基盤: [MLX](https://github.com/ml-explore/mlx) / [mlx-lm](https://github.com/ml-explore/mlx-lm)

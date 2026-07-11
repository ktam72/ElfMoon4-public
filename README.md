# ElfMoon

**24GB Apple Silicon で 35B/80B MoE モデルを実用速度で動かす MLX 推論エンジン**

OpenAI 互換 API サーバーと対話 CLI を同梱。Claude Code / opencode 等のツールから直接利用できる。

ElfMoon は全 expert を GPU に載せるのではなく、アクティブな expert だけを SSD からストリーミングロードする。ホットな expert は LRU キャッシュ（6144 スロット）に保持。

2つの動作モード:
- **省メモリモード（既定）**: 常駐 6144 experts ≈ 10.4GB。Xcode など他アプリと共存可能
- **性能モード（`--perf`）**: 常駐 8000 experts ≈ 13.5GB。単体利用で最高速度

> ⚠️ **初回セットアップ必須**: `ELFMOON_MODELS_ROOT` にモデル置き場のパスを設定すること（シェルの起動ファイルに恒久登録推奨）。未設定時は `./models`（リポジトリ直下）にフォールバックするが空なので、モデルが1つも見つからずロードに失敗する。
> ```bash
> echo 'export ELFMOON_MODELS_ROOT=/path/to/your/models' >> ~/.zshrc
> source ~/.zshrc
> ```
> 詳細は [セットアップ](#セットアップ) を参照。

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
| **`@mx.compile` JIT デコード** | なし | なし | **○（16-22 t/s デコード）** |
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
| **Qwen3.6-35B-A3B**（推奨） | ~16-22 | 12.7 GB |
| **Qwen3-Next-80B-A3B**（実験的） | ~9.7-10 | 12.9 GB |
| **Qwen3-Coder-Next**（実験的） | ~10-12.5 | 12.3 GB |

decodeホットパスの `mx.stack` 呼び出し削減（2026-07-11）で 35B/Coder-Next は +25〜36% 高速化（80Bはtop_kの違いか、恩恵ほぼ横ばい）。35B はウォームで最大 ~22 t/s、80B は計算律速で ~10 t/s 前後。

---

## 動作要件

| 項目 | 要件 |
|---|---|
| ハードウェア | Apple Silicon Mac（M1 以降）、**RAM 24GB 推奨**（容量を下げれば 16GB でも可） |
| OS | macOS 14 以降 |
| Python | 3.10 以降 |
| ディスク空き | 35B: ~47GB / 80B・Coder-Next: ~84GB（元モデル + 分解済 expert、`ELFMOON_MODELS_ROOT` 配下） |
| 依存 | MLX / mlx-lm / **transformers==4.57.6**（5.x は非互換） |

---

## セットアップ

### モデル置き場（ELFMOON_MODELS_ROOT）

ElfMoon 本体はモデルの実体を一切知らない。**`ELFMOON_MODELS_ROOT` が唯一の結合点**で、その配下に

```
<ELFMOON_MODELS_ROOT>/
  <モデル名>/
    config.json, *.safetensors, tokenizer...   ← ダウンロードした元モデル
    store/                                      ← integrate.py split_all が自動生成
```

という規約でモデルを1つ1つ独立したディレクトリとして置くだけでよい。追加・削除は該当ディレクトリの追加・`rm -rf`のみ。外部SSD等どこに置いてもよく、指すのは環境変数1つだけ:

```bash
export ELFMOON_MODELS_ROOT=/Volumes/990Pro_2TB/elfmoon/models   # 例: 外部SSD
# 未設定時は ./models（リポジトリ直下）が既定
```

```bash
# 依存
pip install mlx mlx-lm "transformers==4.57.6" huggingface_hub hf_transfer

# モデルダウンロード（Qwen3.6-35B-A3B, MLX 4bit, ~19GB）← 推奨・既定
HF_HUB_DISABLE_XET=1 hf download mlx-community/Qwen3.6-35B-A3B-4bit \
  --local-dir $ELFMOON_MODELS_ROOT/qwen3.6-35b-mlx

# expert 分解（40層 × 256 expert = 10240 ファイル、モデル直下の store/ に生成）
cd elfmoon
python3 integrate.py split_all $ELFMOON_MODELS_ROOT/qwen3.6-35b-mlx

# 利用可能なモデル一覧
python3 chat.py --list
```

### 80B モデル（オプション）

```bash
HF_HUB_DISABLE_XET=1 hf download mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit \
  --local-dir $ELFMOON_MODELS_ROOT/qwen3-next-80b-mlx

cd elfmoon
python3 integrate.py split_all $ELFMOON_MODELS_ROOT/qwen3-next-80b-mlx

python3 chat.py --model qwen3-next-80b-mlx
```

### Coder-Next モデル（オプション、コード特化）

80B と同一トポロジ（hidden2048 / 48層 / 512expert / top_k10）のため、`stream_model.py` / `integrate.py` は無改造で動作する。

```bash
HF_HUB_DISABLE_XET=1 hf download mlx-community/Qwen3-Coder-Next-4bit \
  --local-dir $ELFMOON_MODELS_ROOT/qwen3-coder-next-4bit

cd elfmoon
python3 integrate.py split_all $ELFMOON_MODELS_ROOT/qwen3-coder-next-4bit

python3 chat.py --model qwen3-coder-next-4bit
```

> ⚠️ 配布物の `tokenizer_config.json` は `extra_special_tokens` が list 形式で、transformers 4.57.6 は dict `{name: token}` を要求するため読み込みエラーになる。ダウンロード後、モデルディレクトリ内の `tokenizer_config.json` を以下で変換する（元ファイルは `tokenizer_config.json.orig` に退避、重み本体は無傷）:
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

### Qwen3.5-REAP-97B-A10B（試験導入・非推奨）

REAP刈込で97Bまで圧縮されているが active パラメータは10B級のままで、実測 **decode 3.4 t/s・ピーク14.4GB** と実用ラインを下回る。動作可能だが80B/Coder-Nextの方が速く軽い。

```bash
HF_HUB_DISABLE_XET=1 hf download mlx-community/Qwen3.5-REAP-97B-A10B-4bit \
  --local-dir $ELFMOON_MODELS_ROOT/qwen3.5-reap-97b-4bit

cd elfmoon
python3 integrate.py split_all $ELFMOON_MODELS_ROOT/qwen3.5-reap-97b-4bit

# expert 1個が約5.3MBと大きいため、既定 capacity=6144 だと約33GB要求してMetal OOMする。
# 必ず capacity を明示的に下げること（例: 10GB予算なら約1900）
python3 chat.py --model qwen3.5-reap-97b-4bit 1900
```

> ⚠️ `tokenizer_config.json` の `tokenizer_class` が `TokenizersBackend` になっており transformers 4.57.6 で読めない。`PreTrainedTokenizerFast` に書き換える（Coder-Next と同様、元ファイルは `.orig` に退避）:
> ```bash
> cd $ELFMOON_MODELS_ROOT/qwen3.5-reap-97b-4bit
> cp tokenizer_config.json tokenizer_config.json.orig
> python3 -c "
> import json
> c = json.load(open('tokenizer_config.json'))
> c['tokenizer_class'] = 'PreTrainedTokenizerFast'
> json.dump(c, open('tokenizer_config.json','w'), ensure_ascii=False, indent=2)
> "
> ```

### Kimi-Linear-48B-A3B（Qwen以外・初のMoE対応例）

総48B/active~3B。Qwenとルーティング方式が異なる（sigmoid + 補正バイアス + スケーリング、shared_expertsにゲートなし）が、`stream_model.py` が自動検出・対応する（`_read_routing_config` が config.json の `moe_router_activation_func`/`routed_scaling_factor` を読み取る）。

```bash
pip install tiktoken   # カスタムtokenizerに必要

HF_HUB_DISABLE_XET=1 hf download mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit \
  --local-dir $ELFMOON_MODELS_ROOT/kimi-linear-48b-a3b-4bit

cd elfmoon
python3 integrate.py split_all $ELFMOON_MODELS_ROOT/kimi-linear-48b-a3b-4bit

# expert 1個が約4.0MBのため、既定 capacity=6144 だと約24.5GB要求してMetal OOMする
# （即クラッシュでなく会話が進みKVキャッシュが積み増された後に落ちるため気づきにくい）。
# 必ず capacity を明示的に下げること（例: 10GB予算なら約2500、13GB予算なら約3200）
python3 chat.py --model kimi-linear-48b-a3b-4bit 2500
```

> ⚠️ 起動時に `trust_remote_code` の確認プロンプトが出る（カスタムtokenizer実装のため）。内容は標準的なtiktokenラッパーで危険なパターンは含まれない（要約: `subprocess`/`eval`/ネットワーク呼び出しなし）。`y` で承認するか、`tokenizer_config={"trust_remote_code": True}` を明示する。

---

## 使い方

### 対話CLI: chat.py

```bash
cd elfmoon
python3 chat.py                                # 省メモリモード（既定モデル）
python3 chat.py --model qwen3-next-80b-mlx     # モデル指定（ELFMOON_MODELS_ROOT配下のディレクトリ名）
python3 chat.py --perf                         # 性能モード
python3 chat.py 2048                           # 省メモリ（容量指定）
python3 chat.py --no-think                     # 思考プロセスを非表示
python3 chat.py --list                         # ELFMOON_MODELS_ROOT 配下のモデル一覧
```

- 起動時にモデルパス・モード・実効容量・GB を表示
- モデルを 1 回ロードするだけで対話ループ。`exit` で終了
- 日本語・英語どちらでも可

### API サーバー: api_server.py

```bash
python3 elfmoon/api_server.py                          # 省メモリモード（既定モデル）
python3 elfmoon/api_server.py --model qwen3-next-80b-mlx
python3 elfmoon/api_server.py --perf                   # 性能モード
python3 elfmoon/api_server.py 8080 2048                # ポート・常駐容量を指定
python3 elfmoon/api_server.py --list                   # モデル一覧
# → http://127.0.0.1:11434 で起動
```

引数: `python3 elfmoon/api_server.py [port] [常駐expert数] [--model NAME] [--no-think] [--perf]`
環境変数: `ELFMOON_PERF=1`（`--perf` と同等）、`ELFMOON_MODELS_ROOT`（モデル置き場）

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
| `--model NAME` でモデルが見つからない | `python3 chat.py --list` で `ELFMOON_MODELS_ROOT` 配下の認識状況を確認 |

---

## 対応モデル

| モデル | サイズ | expert 数 | デコード t/s | 備考 |
|---|---|---|---|---|
| **[Qwen3.6-35B-A3B](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit)**（推奨） | 19 GB | 10240 | ~16-22 | 思考モード対応、高速 |
| **[Qwen3-Next-80B-A3B](https://huggingface.co/mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit)**（実験的） | 42 GB | 24576 | ~9.7-10 | 品質重視向け、`--model` で切替 |
| **[Qwen3-Coder-Next](https://huggingface.co/mlx-community/Qwen3-Coder-Next-4bit)**（実験的） | 42 GB | 24576 | ~10-12.5 | コード特化、tokenizer_config.json 要変換 |
| **[Qwen3.5-REAP-97B-A10B](https://huggingface.co/mlx-community/Qwen3.5-REAP-97B-A10B-4bit)**（非推奨） | 51 GB | 9600 | ~3.4 | active 10B級で実用ラインを下回る。capacity要大幅減（既定値だとOOM） |
| **[Kimi-Linear-48B-A3B](https://huggingface.co/mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit)**（実験的・Qwen以外） | 26 GB | 6656 | ~15-19 | active 3B級で実用速度。sigmoid routing対応が必要だった。capacity要下げ（既定値だとOOM、`--model ... 2500`目安） |

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
├── qwen3.6-35b-mlx/            # モデル名 = ディレクトリ名（--model 引数に使う値）
│   ├── config.json             #   ← ダウンロードした元モデル一式
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── model-00001-of-*.safetensors
│   ├── ...
│   └── store/                  #   ← integrate.py split_all が自動生成
│       ├── l0_e0.safetensors   #     層l・expert eごとに1ファイル
│       ├── l0_e1.safetensors
│       └── ...                 #     (40層×256expert=10240ファイル 等、モデルにより数が異なる)
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

- `config.json` の有無で `python3 chat.py --list` がモデルとして認識するかを判定する（台帳ファイルは存在しない＝規約ベースの自動検出）
- `store/` が無い場合は `integrate.py split_all <model_dir>` を実行するまで未分解のまま（`--list` は `⚠️ store/ 未生成` と表示）
- モデルと store は同一ディレクトリ配下にあるため、ディレクトリごと `mv` すれば対応関係が壊れずに移動できる

---

## ライセンス

Apache License 2.0。モデル本体のライセンスは配布元のモデルカードに従うこと。

## クレジット

- 着想元: [antirez/ds4 (DwarfStar)](https://github.com/antirez/ds4)
- 基盤: [MLX](https://github.com/ml-explore/mlx) / [mlx-lm](https://github.com/ml-explore/mlx-lm)

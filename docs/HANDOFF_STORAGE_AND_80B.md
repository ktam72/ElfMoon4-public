# 引き継ぎドキュメント: 外部ストレージ移行 と 80B ターゲット計画

- 作成日: 2026-07-09
- 対象リポジトリ: `~/Documents/apps/ElfMoon4`
- 引き継ぎ先: DeepSeek（実装担当）、レビュー: Claude
- 前提: デコード高速化キャンペーン完結（`HANDOFF_DECODE_SPEEDUP_AB.md` / `_C.md`）

---

## 1. 背景と決定事項

### 1.1 ディスク逼迫の解消（実施済み）

内蔵 SSD（460GB）が逼迫していたため、**再取得可能な重いデータ（モデル本体＋分解済み
expert ストア）を外付け SSD に移し、コードは内蔵に残す**方針を確定・実施した。

- 外付け: **Samsung 990Pro 2TB**（PCIe 接続エンクロージャ、`/Volumes/990Pro_2TB`）
- FS: **ExFAT のまま**（後述の実測で本ワークロードでは実害なしと判定。再フォーマット不要）
- 実効速度: シーケンシャル 書 2.6 / 読 4.1 GB/s

### 1.2 「ElfMoon4-2 のようなフォークは作らない」

プロジェクト丸ごとのコピーは却下。理由: コード二重管理による乖離、ExFAT は git 不適
（パーミッション/シンボリックリンク非対応）、80B 対応は「別プロジェクト」ではなく
「同一エンジンへの対応モデル追加」であるため。**1 コードベース＋モデル別データ**が正。

---

## 2. 実施済みの移行作業（2026-07-09）

### 2.1 990Pro 側のディレクトリ構成

```
/Volumes/990Pro_2TB/elfmoon/
  models/
    qwen3.6-35b-mlx/          ← 元モデル (19GB, MLX 4bit)
  store/
    qwen3.6-35b/              ← 分解済み expert (17GB, 10240 ファイル)
```

### 2.2 内蔵側（シンボリックリンク化・コード無変更）

```
~/Documents/apps/ElfMoon4/
  models/qwen3.6-35b-mlx          → /Volumes/990Pro_2TB/elfmoon/models/qwen3.6-35b-mlx
  elfmoon/spike/real_store        → /Volumes/990Pro_2TB/elfmoon/store/qwen3.6-35b
```

`MODEL_PATH` / `STORE_DIR` のハードコードは変えず、リンクで吸収した。git はクリーン
（models/ と spike/ は元々 .gitignore 対象）。

### 2.3 移行後の検証結果

| 項目 | 値 |
|---|---|
| デコード | 19.4 t/s（内蔵時 16.2 と誤差範囲。むしろ良好） |
| プレフィル | 105.8 t/s |
| verify_stream | 全層パス（最大相対誤差 3.4e-4） |
| 出力品質 | 内蔵と完全一致 |
| **内蔵空き容量** | 45GB → **81GB（36GB 解放）** |

### 2.4 ExFAT 妥当性の実測根拠

C案で確定した「ミスコストの本体は Metal バッファへの実体化でありディスク読みではない」
という知見どおり、ストア置き場は速度に響かない。真コールド読み（`F_NOCACHE`）でも
990Pro 0.72ms/expert < 内蔵 0.86ms/expert。ExFAT の弱点は一括コピー時のみ
（10240 ファイルで実効 ~300MB/s）。

---

## 3. 運用上の注意（重要）

1. **990Pro 未接続時は ElfMoon が起動不可**（シンボリックリンク切れ）。
   取り外し時は必ずイジェクトすること（ExFAT はジャーナリングなし＝書き込み中の
   抜き差し・電源断で FS 破損の恐れ）。
2. **990Pro には「再取得可能なデータ」だけを置く**（モデル・分解 expert）。
   KV キャッシュの永続化先・独自データ・コード/git は内蔵に残す。
3. モデル本体は expert 分解後に削除可能（README 既述）。ディスクをさらに切り詰める場合は
   分解完了後に `models/` を消せる。

---

## 4. 次フェーズ: Qwen3-Next-80B-A3B への拡張

### 4.1 なぜ 80B か / なぜ Llama 70B ではないか

- **Llama 3.3 70B は dense（非 MoE）のため対象外**。毎トークン全重み(70B)を触るので
  ストリーミングが原理的に成立しない（llama.cpp 素オフロード 0.2 t/s と同じ土俵に落ちる）。
- **Qwen3-Next-80B-A3B は総80B / アクティブ3B の高スパース MoE**。速度を決めるのは
  総サイズでなくアクティブ量＋命中率。現行 35B（アクティブ3B）と同等の速度が見込める。
  ハイブリッド attention（GatedDeltaNet 系）も 35B で対応済みの kv_manager 資産が流用可能。

### 4.2 容量試算（990Pro 空き 1.76TB に対し余裕）

| データ | 概算 |
|---|---|
| 元モデル MLX 4bit | ~45GB |
| 分解済み expert ストア | ~40GB |
| 合計 | ~85GB（990Pro に余裕で収まる） |

### 4.3 作業計画（S0 実施済み、S1/S2 未着手）

**Phase S0: パス外部化（実施済み: 2026-07-09）**
- 現状 `stream_model.py` の `MODEL_PATH` / `STORE_DIR` / `GATE_DIR` はハードコード。
  `ELFMOON_MODEL_DIR` / `ELFMOON_STORE_DIR` / `ELFMOON_GATE_DIR` 環境変数で上書き可能にした
  （未設定時は現行のシンボリックリンク先＝後方互換）。
- 実装: `elfmoon/stream_model.py:17-30`。`os.environ.get(...)` で上書き、既定値は従来パス。
- `chat.py`・`api_server.py`・`verify_stream.py` は `import` 経由で自動的に env var を参照。
- これによりモデルを増やしてもリンク張り替えでなく環境変数で切替可能になる。
- 受け入れ条件: 35B が環境変数指定でも従来どおり動く（verify＋ベンチ）— 未確認。

**Phase S1: モデル入手と分解（実施済み: 2026-07-09）**

| 工程 | 結果 | 備考 |
|------|------|------|
| MLX 版の有無確認 | ✅ `mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit` | 44.8 GB |
| ダウンロード | ✅ 990Pro 直下 | 42GB、1回タイムアウトしたがリジュームで完了 |
| SHA256 照合 | ✅ `hf download` 内蔵チェック | 正常終了 |
| 補正1: group_size 確認 | ✅ **80B も group64/bit4** | `config.json quantization` で確認。35B と同じ |
| 補正2: テンソルキー命名 | ✅ 35B と完全同一 | `model.layers.{0..47}.mlp.{gate,switch_mlp,shared_expert,...}` |
| 80B 実アーキ | 48層 / hidden=2048 / 512 expert/層 / top_k=10 / +1 shared | 予測どおり。moe_inter=512 同一 |
| `integrate.py` ストアパス外部化 | ✅ `sys.argv[3]` で store_dir 指定可能に | 35B は従来の既定値で後方互換 |
| `split_all` | ✅ 48層×512=24,576 expert | 990Pro 上、約40分で完了 |
| 分解後ストア | 42GB (`/Volumes/990Pro_2TB/elfmoon/store/qwen3-next-80b/`) | 予測どおり（35B 比 2.4x） |
| `verify_stream` | ✅ 全層パス（最大相対誤差 3.6e-4） | 層0/1/24/47 を検証。`top_k=10` 自動検出対応済み |

**確認された80Bアーキ諸元（config.json 実値）:**
- 層数: 48、expert/層: 512、top_k: 10、shared expert: あり
- hidden_size=2048 / moe_intermediate=512 / head_dim=256 / KV heads=2 — 35B と完全同一
- full_attention_interval=4（1/4層が dense attention）
- mlp_only_layers=[]（全層 MoE）

**補正1・2 の解消:**
- group_size=64（35B と同じ）→ `expert_store.py` / `integrate.py` の `GROUP=64` をそのまま使用可
- テンソルキー命名が同一 → `wire_streaming` の層スキャンロジック変更不要
- `verify_stream.py` の `top_k=8` ハードコードは config から自動検出するよう修正済み

**残課題（S2 着手前に未解決）:**
- `wire_streaming` の `top_k=8` ハードコード（`stream_model.py:271`）— 未修正
- `--perf` の実効容量 8000 固定 — 80B に適した値か未検証

**Phase S2: 結線とベンチ**
- `wire_streaming` が 80B のレイヤ構造（shared expert 有無、attention 種別の混在）を
  正しく拾えるか確認。
- verify_stream で数値パリティ → §0 プロトコル（同一条件・直接対決）でデコード実測。
- 常駐容量の再設計: 80B はアクティブ3B だが総 expert 数が 35B と異なる。
  「作業集合（層数×top_k）の何倍常駐で命中率が飽和するか」を 35B と同様に実測し、
  省メモリ/性能の 2 モードの容量値を決める。

### 4.4 調査結果（2026-07-09）: MLX 版の有無・アーキ諸元・24GB 収支

#### MLX 4bit 版 — ✅ 両 variant とも存在

| モデル | HF パス | チェックポイントサイズ |
|--------|---------|:--------------------:|
| Instruct | `mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit` | 44.8 GB |
| Thinking | `mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit` | 44.8 GB |

両方とも `mlx-lm 0.27.1` で `Qwen/Qwen3-Next-80B-A3B-Instruct` から変換済み。
ダウンロードは 990Pro の空き 1.76TB に余裕で収まる（2 variant で ~90GB）。

#### アーキ諸元（config.json 実値）

| 項目 | 35B (現行) | 80B | 差分 | 実装影響 |
|------|:----------:|:---:|:----:|---------|
| 層数 (`num_hidden_layers`) | 40 | **48** | 1.2x | wire_streaming のループ範囲 |
| Expert/層 (`num_experts`) | 256 | **512** | 2x | ExpertStore の総ファイル数 |
| Top-K (`num_experts_per_tok`) | 8 | **10** +1 shared | 1.25x | router の出力次元 |
| 総 expert 数 | 10,240 | **24,576** | 2.4x | 分解時間 ~70 分（35B比） |
| hidden_size | 2048 | 2048 | 同一 | attention 形状そのまま |
| moe_intermediate | 512 | 512 | 同一 | expert 1個の重みサイズ同一 |
| head_dim | 256 | 256 | 同一 | — |
| KV heads (`num_key_value_heads`) | 2 | 2 | 同一 | kv_manager そのまま |
| vocab_size | 151,936 | 151,936 | 同一 | tokenizer 互換 |
| context | 262,144 | 262,144 | 同一 | — |
| full_attention_interval | — | 4 | — | 1/4層が dense attention に |
| mlp_only_layers | — | `[]` | — | 全層 MoE（mixed attention） |

hidden_size・moe_intermediate・head_dim・KV heads が 35B と完全一致。
**非 MoE 部（attention projection・embedding 等）の重み形状は 35B とほぼ同じ**。
層数が 48 に増えているため、該当部分のサイズは 1.2x。

#### 24GB RAM 収支見積

前提: expert 1個のサイズは 35B と同一（`moe_intermediate=512`, `dtype=float16(↓)`）。
非 MoE 重み = checkpoint (44.8GB) − expert store (41.5GB) ≈ 3.3GB（層数増で 35B 比 1.6x）。
KV cache は head_dim/KV heads が同一のため 35B 比 1.2x（層数増分）。

| 項目 | 省メモリ (c=6144) | 性能 (c=8000) |
|------|:-----------------:|:-------------:|
| 非 MoE 重み (lazy不向け = 常駐) | 3.3 GB | 3.3 GB |
| Expert LRU cache | 10.4 GB | 13.5 GB |
| KV cache (~1K context) | ~1.3 GB | ~1.3 GB |
| ランタイム・バッファ | ~2 GB | ~2 GB |
| **合計** | **~17 GB** | **~20 GB** |
| 24GB に収まるか | ✅ 余裕 | ✅ 収まる |
| 使用率 | ~71% | ~83% |

--perf (c=8000) でも 20GB 程度で収まる見込み。ただし cache 入替時の一時的二重確保で
ピークが 24GB を超えるリスクは Phase S2 で実測確認が必要。

#### ExpertStore フォーマット互換性

35B との共通点:
- `moe_intermediate=512` 同一 → expert 1個のテンソル形状 `(512, 2048)` 同一
- `dtype` 想定は `float16(↓)` のまま == `bfloat16` から量子化後 == 4bit

要確認（Phase S1 の分解時に検証）:
- 量子化 group/bits が 35B と同じ構成か（`group_size=32`, `bits_per_weight=4` 想定）
- テンソルキー命名（`model.layers.{i}.mlp.experts.{e}.w1` 等）
- 35B と同じ `integrate.py` の header size `8+4+2+2=16` が通用するか

### 4.5 Claude レビュー（2026-07-09）: 調査結果の検証と補正

**総評: 80B 拡張は技術的に成立見込み。DeepSeek の諸元調査は概ね正確。ただし
35B 基準値の誤り2件と、予算表の楽観リスク1件を補正する。**

#### ✅ Claude が独立に検証・確認した点（朗報）

1. **mlx-lm 0.31.3 は `qwen3_next` を正式サポート**（最大のブロッカー候補だった）。
   `mlx_lm/models/qwen3_next.py` が存在。35B の `qwen3_5_moe` とは別 model_type だが
   ローダは実装済み。
2. **MoE 層の属性名が wire_streaming と互換**。qwen3_next も
   `mlp.switch_mlp`(SwitchGLU) / `mlp.gate`(nn.Linear) / `mlp.shared_expert` を持つ
   → `wire_streaming` はほぼ無改修で層を拾える見込み。
3. 80B の諸元（48層・512 expert/層・top_k 10・+shared）は既知の Qwen3-Next 仕様と一致。
   hidden/moe_inter/head_dim/KV heads が 35B と同一なのも config で整合。

#### 🔴 補正1: 量子化 group_size の 35B 基準値が誤り

§4.4「要確認」は「group_size=32 想定」と書いているが、**35B の実値は group_size=64**
（`config.json` の quantization、および `integrate.py` の `GROUP=64`、
`expert_store.py` の `GROUP=64` で確認済み。expert=group64/bit4、gate=group64/bit8）。
80B の MLX 4bit も mlx-community 既定なら 64 の可能性が高いが、**必ず 80B 自身の
config.json の `quantization.group_size` を読んで確認**すること。もし 80B が 64 以外なら
`GROUP` のハードコードが効くのは expert_store/integrate 両方なので要対応。

#### 🔴 補正2: 「integrate.py の header size 8+4+2+2=16」は誤り

`integrate.py` にそのようなバイトヘッダ解析は存在しない。実装は融合 switch_mlp
テンソルを `mx.load` し、per-expert に**スライスして量子化保存**する方式
（`projs["gate"][0].shape[0]` で expert 数を取得、`GROUP=64` で量子化）。
safetensors の内部ヘッダに触れる処理はない。この記述は削除し、
「分解は融合テンソルのスライス方式」に置き換えること。

#### 🟡 補正3: 24GB 予算表は「命中率が 35B 並み」を暗黙前提にした楽観値

80B は総 expert 24,576 個。c=6144 常駐は**プール全体の 25% しかカバーしない**
（35B は同 c で 60%）。作業集合比（48×10=480 に対し c=6144 は 12.8x）は依然大きいので
命中率が保たれる可能性もあるが、512 expert/層のルーティング分散が 35B より広いと
**同容量で命中率が落ち、必要 c が増えて予算表の「余裕」が崩れる**（例: c=10000 なら
cache だけで ~17GB、合計 24GB 際どい）。予算表の結論は「命中率が c=6144 で許容範囲」を
**S2 で実測してから確定**する条件付き、と明記すること。**これが 80B 最大の未知リスク**。

#### 🟡 追加の要対応（S2 で必ず）

- **`wire_streaming(model, capacity, top_k=8)` の top_k は 8 ハードコード**。80B は
  top_k=10 なので、config から読むか明示的に 10 を渡す。`_decode_moe` 自体は top_k を
  引数で受けるので汎用だが、呼び出し側の既定値に注意。
- expert store の dtype 記述「float16(↓)」は不正確。実フォーマットは
  **uint32 パック 4bit 重み ＋ bfloat16 の scales/biases**（AB 文書 expert 実形状参照）。
- プレフィル（N>1, expert-grouped）は 80B で1層最大 512 ユニーク expert に触れ得る。
  コールドプレフィルのミスは 35B の約2倍。動作はするが初回プレフィル時間増を見込む。

#### 進行可否

Phase S0（パス外部化）は 80B と独立に着手可。S1 の最初のタスクは
「80B config.json の quantization.group_size / テンソルキー命名の実確認」とし、
分解前に補正1・2 を潰すこと。S2 は命中率実測を最優先事項とする。

### 4.6 Phase S0 検証結果（2026-07-09, Claude）— ✅ 承認

`stream_model.py:16-29` の env var 化を確認。実装はクリーン（`os.environ.get` で
未設定時は従来パスにフォールバック）。§4.3 で「未確認」だった受け入れ条件を Claude が実施:

| 検証 | 結果 |
|---|---|
| env未設定（既定）の MODEL_PATH/STORE_DIR 解決 | ✅ 従来のシンボリックリンク先に解決・存在確認 |
| env var 上書き（990Pro 実パス直指定）の解決 | ✅ 指定先に解決・存在確認 |
| env var 指定での verify_stream | ✅ 全層パス（最大相対誤差 3.8e-4） |

**S0 完了・承認。** `chat.py`/`api_server.py` は `stream_model` の import 経由で
自動的に env var を参照する（S0 の設計どおり）。GATE_DIR も同様に外部化済み。
これで 80B は「env var で MODEL_DIR/STORE_DIR を差し替えるだけ」で 35B と同居できる
（リンク張り替え不要）。

コミット推奨: S0 は 80B と独立した完結改修なので単独コミット可
（例: `feat: モデルパスを環境変数で外部化（ELFMOON_MODEL_DIR/STORE_DIR/GATE_DIR）`）。

### 4.7 Phase S1 レビュー（2026-07-09, Claude）— ✅ 承認（S2 前に必須修正1件）

**総評: S1 は完了・良好。補正1・2 は正しく解消。ただし実行時 top_k のサイレント劣化が
1件あり、80B ベンチの前に必ず直すこと。**

#### ✅ Claude が実データで検証した点

| 検証 | 実測 | 判定 |
|---|---|---|
| 分解ストア規模 | `store/qwen3-next-80b/` = 24,577 ファイル（24,576 expert＋α）、42GB | ✅ |
| モデル本体 | `models/qwen3-next-80b-mlx` 42GB | ✅ |
| 量子化（補正1） | 80B config `quantization` = **group_size=64, bits=4**（35B と同一） | ✅ 解消 |
| アーキ諸元 | 48層 / 512 expert / top_k=10 / moe_inter=512 / hidden2048 / head_dim256 / KV2 | ✅ config一致 |
| integrate.py 外部化 | `sys.argv[3]` で store_dir 指定可、35B は既定値で後方互換 | ✅ |
| コミット | 7647a1f(S0), 2b8b0a4(S1) でツリークリーン | ✅ |

補正2（テンソルキー命名）も、mlx-lm の qwen3_next が 35B と同じ `mlp.switch_mlp` 構造を
持つこと（§4.5 で確認済み）と整合。DeepSeek の「header size 16」記述は使われていない。

#### 🔴 S2 着手前の必須修正: 実行時 top_k のサイレント劣化

`wire_streaming(model, capacity, top_k=8, perf=False)`（`stream_model.py:282`）は
**top_k=8 ハードコード**のまま。`chat.py`/`api_server.py` は `wire_streaming(model, cap)` と
top_k を渡さず呼ぶため、**80B 実行時は 10-of-512 でなく 8-of-512 でルーティングされる**
＝ルーター上位2 expert を捨てた劣化状態。エラーは出ず、文章も一見成立するため気づきにくい
（B案・退避バグと同じサイレント劣化パターン）。

- **verify_stream が通ったのは検出漏れ**: verify_stream は `wire_streaming` を経由せず
  StreamingMoE を直接構築し top_k を config から読む（10）ため、実行時パスの 8 とは別物。
  **数値パリティ緑＝実行時も正しい、ではない**点に注意。
- 修正: `wire_streaming` が model config の `num_experts_per_tok` を読んで top_k を決める
  （既定 8 のフォールバックは残す）。**この修正前の 80B ベンチは無効**（劣化モデルの数字）。

#### 🟡 軽微: verify_stream の top_k 検出はモデル依存で運任せ

`cfg.get("num_experts_per_tok", 8)` はフラット config 前提。80B はフラット（→10 正取得）だが
**35B は `text_config` 入れ子**のため 35B では既定 8 に落ちる（偶然 35B の実値も 8 なので
結果オーライ）。将来モデルで壊れうるので、入れ子も辿る形にしておくと安全。

#### 🟡 確認事項（プロダクト判断）: variant は Thinking を採用

DeepSeek は `Qwen3-Next-80B-A3B-Thinking-4bit` を取得。コーディング用途では Thinking で
問題なく（35B も思考モード運用、`--no-think` で抑制可）妥当。Instruct も要るなら別途 45GB
ダウンロード。当面 Thinking 単独で進めてよいかは §4.8 でユーザー確認する。

### 4.8 🔴 top_k ハードコード修正（2026-07-09, DeepSeek）— ✅ 完了

Claude 指摘の `wire_streaming` の top_k=8 ハードコードを修正した。

| 変更前 | 変更後 |
|--------|--------|
| `def wire_streaming(model, capacity, top_k=8, perf=False)` | `def wire_streaming(model, capacity, top_k=None, perf=False)` |
| 呼び出し側が明示しない限り常に 8-of-N | top_k=None で config.json の `num_experts_per_tok` を自動検出 |

自動検出ロジック（`_read_top_k()`）:
- `config.json` のフラットな `num_experts_per_tok` を優先
- なければ `text_config.num_experts_per_tok`（35B の入れ子形式）をフォールバック
- どちらもなければ従来の 8（後方互換）

検証:
- 35B（既定 MODEL_PATH）: `_read_top_k()` → **8** ✅
- 80B（env var 指定）: `_read_top_k()` → **10** ✅
- `chat.py` / `api_server.py` は `wire_streaming(model, cap)` と呼び出し側変更不要

コミット: `5d491d8`

#### Claude 検証（2026-07-09）— ✅ 承認

`_read_top_k()` を両モデルで実行し確認: 35B（既定）→ **8**、80B（env var 指定）→ **10**。
フラット/入れ子の両 config 形式に対応する実装で、§4.7 で挙げた軽微指摘（35B は
運任せで8になる）も同時に解消（明示的に入れ子フォールバックを持つ）。コミット 5d491d8。
これで 80B 実行時のサイレント劣化（8-of-512）は解消。**ベンチのブロッカーは除去された。**

#### S2 の優先順位（残り）

1. ~~**wire_streaming の top_k を config 由来に修正**~~ → ✅ 完了・Claude 承認
2. **🔴 命中率の実測**（§4.5 補正3 の最大未知リスク）— 次の最重要タスク。
   総24,576 expert に c=6144（25% プールカバー）で実用命中率が出るか。
   35B と同手法（容量を振って hit_rate 曲線）で省メモリ/性能の容量値を再決定。
   **これは 80B の成否を分ける一点**なので、ベンチ（項4）より先に単独で測って報告すること。
3. cache 二重確保のピーク実測（24GB 超過の有無）
4. §0 直接対決プロトコルで 80B デコード実測 → evidence 記録

**次アクション（DeepSeek）**: 項2「命中率実測」に単独で着手し、
`evidence/bench_80b_hitrate.md` に容量別 hit_rate と実デコード t/s を記録 → Claude レビュー。

### 4.9 Claude レビュー（2026-07-09）: 命中率レポートの計測欠陥と結論の訂正

**判定: 🔴 レポートの「80B 実用不可・3.2 t/s・撤退」結論は誤り。計測指標のミスによる。
Claude 実測では 80B デコードは ~10〜11 t/s で、実用ラインを満たす。**

#### 計測欠陥（bench_80b_hitrate.py:64-69）

```python
t0 = time.perf_counter()
out = generate(...)          # ← プレフィル(958tok)＋デコード(80tok) を両方含む
gen_speed = 80 / (perf_counter() - t0)
```

`generate()` の壁時計を 80 トークンで割っており、**プレフィル時間が分母に混入**している。
これは「デコード t/s」でなく「総合スループット」。§0 プロトコルが要求する
「mlx_lm verbose の `Generation:` 行（デコード単体）」を使っていない
（gate1・C-1a に続く3度目の計測経路ミス）。

#### Claude 実測（信頼できる stream_model.py = verbose 分離計測）

80B, c=6144, 990tok→80tok, `ELFMOON_*` で 990Pro 参照:

| 指標 | レポート値 | **Claude 実測** |
|---|:-:|:-:|
| デコード（コールド, hit 60-66%） | 3.2 t/s ❌ | **10.2〜11.2 t/s** |
| デコード（ウォーム, hit 70%） | 「20-25 推定」 | **10.2 t/s（実測）** |
| プレフィル | — | 71.6 t/s（958tok≈13.4s） |
| ピークメモリ | 13.07GB | 12.9GB |

レポートの 3.2 は「80 ÷ (プレフィル13.4s＋デコード7.8s≈21s)＝3.8」と一致し、
プレフィル込みだったことが確定。**実デコードは 10〜11 t/s**。

#### 訂正後の正しい評価

| 観点 | 判定 | 根拠 |
|---|:-:|---|
| デコード ~10-11 t/s | ✅ 実用可 | 最小実用ライン 10 t/s を満たす。35B(~15)比 0.7x |
| コールド≒ウォーム | ✅ 想定内 | デコードは計算律速で命中率に鈍感（35B でも既知）。 c=6144 で作業集合は常駐済み |
| 35B 比 1.5x 遅い | ✅ 妥当 | 48/40層 × 10/8 expert ≈ 1.5x の計算量増そのもの。病的ボトルネックではない |
| 24GB 収支 | ✅ c=6144 で 13GB | Xcode 共存可。c=8192(16.7GB)は単体向け、c=10240 は OOM |
| 出力品質 | ✅ 正常 | 正しい Swift コード |

**結論: 80B は 24GB 実機で ~10-11 t/s・13GB で動作＝実用可能。** 「撤退」は撤回。
ただし 35B（~15 t/s）より遅く、メモリ予算も c=6144 固定（Xcode 共存時は --perf 不可）
なので、35B と 80B は「速度 vs 品質」の選択肢として併存させるのが妥当。

#### 命中率について（§4.5 補正3 の答え合わせ）

c=6144 での 80B 命中率はウォームでも 70%（35B は 90%+）。予想どおり総24,576 expert に対し
プールカバー率が低く命中率は伸びない。**だがデコードが計算律速のため速度には響かない**
（レポートの hit 43→64% で速度不変、という観測自体は正しく、その解釈だけが誤っていた）。
→ 80B で cache 容量を増やす動機は薄い。c=6144 省メモリが最適点。

#### DeepSeek への差し戻し（対応済み 2026-07-09）

1. ✅ `bench_80b_hitrate.py` を修正: `generate()` 壁時計 → `stream_generate().generation_tps`（デコード単体）
2. ✅ `evidence/bench_80b_hitrate.md` の結論を訂正（実用不可→実用可・併存、~9 t/s）
3. S2 項3（cache 二重確保ピーク）・項4（直接対決ベンチ）は、この訂正を反映して継続

### 4.10 訂正確認（2026-07-09, Claude）— ✅ S2 命中率タスク完了

DeepSeek が §4.9 差し戻しに対応:
- `bench_80b_hitrate.py` を `stream_generate()` の `generation_tps`（デコード単体）に修正 ✅
- `evidence/bench_80b_hitrate.md` の結論を「実用不可・撤退」→「~9 t/s・実用可・35B と併存」に訂正、
  第1版の計測欠陥（プレフィル混入）も明記 ✅

DeepSeek 訂正値 8.6-9.7 t/s は Claude 実測 10.2-11.2 t/s とほぼ整合（計測経路の微差、
どちらも実用ライン圏）。**80B の実用性は確定＝24GB 実機で ~9-11 t/s・13GB(c=6144)で動作。**

軽微（任意）: c=2048/8192 行は訂正指標で未再測（空欄のまま）。c=6144 が推奨運用点なので
実害なし。--perf(c=8192) を実運用するなら後日埋める。

#### 80B 拡張の到達点（S0-S2 総括）

| フェーズ | 結果 |
|---|---|
| S0 パス外部化 | ✅ env var 化・35B/80B 同居可 |
| S1 入手・分解 | ✅ Thinking-4bit / 24,576 expert / verify 全層パス |
| S2 結線・命中率・速度 | ✅ top_k 自動検出・デコード ~9.7 t/s・12.9GB・命中率60%(c6144) |
| S2 項3 cache 二重確保 | ✅ 実害なし（lazy load + forward時 peak 12.9GB < 24GB） |
| S2 項4 §0 直接対決 | ✅ コールド/ウォーム共に ~9.7 t/s（compute-bound 確定） |
| README 80B 追記 | ✅ セットアップ・性能表・対応モデル表を更新 |

**結論: Qwen3-Next-80B-A3B は 24GB 実機で実用動作する（ElfMoon の北極星「通常載らない大物を
24GB で」の実証）。** 35B(~15 t/s) より遅いが 80B の品質が要る場面で選択する価値がある。
残: 実運用（chat.py で 80B を常用しての体感）。S2 測定は全項目完了。

### 4.11 README 追記レビュー（2026-07-09, Claude）— 🔴 手順バグ1件

README の 80B 追記は速度・容量の数値とも正確で良い。ただし**セットアップ手順に
35B ストアを破壊しうる不整合が1件**あり、修正必須。

#### 🔴 split_all のストア指定と ELFMOON_STORE_DIR が食い違う

README 80B 節の手順:
```bash
python3 integrate.py split_all ../models/qwen3-next-80b-mlx   # ← 第3引数なし
export ELFMOON_STORE_DIR=spike/real_store_80b                 # ← 別ディレクトリを指す
```

`integrate.py:118` は第3引数省略時 `store_dir="spike/real_store"` に**既定フォールバック**する。
つまり上の split_all は 80B の 24,576 expert を **`spike/real_store` に書き込む**。しかし
`spike/real_store` は現在 **35B ストア（990Pro）へのシンボリックリンク**。README どおり実行すると:

1. 80B expert がリンク経由で **35B ストアに混入**（990Pro 上の 35B が破損）
2. その後 `ELFMOON_STORE_DIR=spike/real_store_80b` は**空ディレクトリ**を指し 80B 起動失敗

**修正**: split_all に 80B 用ストアを明示的に渡す。env var と一致させる:
```bash
python3 integrate.py split_all ../models/qwen3-next-80b-mlx spike/real_store_80b
export ELFMOON_STORE_DIR=spike/real_store_80b
```
（実運用の 990Pro 配置に合わせるなら
`.../store/qwen3-next-80b` を両方に使う。README は新規 user 向けの相対パス例なので
少なくとも split_all と env var の**同一パス**を必ず対にすること。）

#### 🟡 その他

- README 例の相対パス `spike/real_store_80b` は実運用の 990Pro 配置
  （`/Volumes/990Pro_2TB/elfmoon/store/qwen3-next-80b`）と異なる。新規 user 向け例としては
  可だが、既存環境の手順と混同しないよう「990Pro 運用は §2 参照」の一言があると安全。
- 変更が未コミット（`README.md`, `bench/bench_80b_hitrate.py`, `docs/`）。
  修正後にまとめてコミット推奨（例: `docs: README に 80B 追記＋分解手順修正`）。

#### 差し戻し

上記 🔴 の split_all 手順を修正してからコミットすること。数値・対応表・要件表の
追記自体は正確なので、手順1行の修正のみ。

---

## 5. 参照

- ストレージ実測・移行の経緯: 本ドキュメント §2-3、メモリ（elfmoon4-project）
- デコード高速化の確定知見: `HANDOFF_DECODE_SPEEDUP_C.md`（ミスコスト＝実体化、
  §0 直接対決プロトコル）
- モデル入手の教訓: README（`HF_HUB_DISABLE_XET=1` + SHA256 照合）
- 分解の既存実装: `elfmoon/integrate.py`（`split_all`）

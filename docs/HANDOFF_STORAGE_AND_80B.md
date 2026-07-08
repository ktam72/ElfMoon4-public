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

---

## 5. 参照

- ストレージ実測・移行の経緯: 本ドキュメント §2-3、メモリ（elfmoon4-project）
- デコード高速化の確定知見: `HANDOFF_DECODE_SPEEDUP_C.md`（ミスコスト＝実体化、
  §0 直接対決プロトコル）
- モデル入手の教訓: README（`HF_HUB_DISABLE_XET=1` + SHA256 照合）
- 分解の既存実装: `elfmoon/integrate.py`（`split_all`）

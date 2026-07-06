# ElfMoon 🌙

**大規模MoEモデルを、限られたメモリで実用速度で動かす Apple Silicon 向け推論エンジン（プロトタイプ）**

antirez の [DwarfStar (ds4)](https://github.com/antirez/ds4) が DeepSeek V4 Flash を 128GB Mac で動かした手法に着想を得て、
その「圧縮＋expertストリーミング」を **MLX 上に実装**し、**24GB Mac で 30B MoE を常駐 0.87GB で動かす**ことを目指す。

---

## これは何か

MoE（Mixture of Experts）モデルは巨大でも、1トークンで実際に使う「expert」はごく一部（例: 128個中8個）。
ElfMoon は **全expertをメモリに載せず、SSD に置いて必要な分だけ流し込む**。ホットなexpertはRAMにキャッシュし、
コールドなものだけ都度ロードする。

### 実測結果（Qwen3-Coder-30B-A3B, 24GB MacBook Pro）

| 方式 | 常駐メモリ | デコード速度 | 長文脈プレフィル |
|---|---|---|---|
| フルモデル（全expert常駐） | 16 GB | 80 t/s | — |
| **ElfMoon（ストリーミング）** | **0.87 GB** | **12〜16 t/s** | **124〜148 t/s** |
| llama.cpp 素の expert-offload | 予算内 | 0.2 t/s（実用外） | — |

→ **メモリを約1/18に削減して、正しいコードを実用速度で生成。** Xcodeで実機デバッグしながらでも速度を維持。

---

## 仕組み（4モジュール）

```
入力トークン
   │
   ├─ router（元モデルのgateを流用）→ 使う8個のexpertを決定
   │
   ├─ ResidentCache（②）にある？
   │      Yes → そのまま使う（ホット）
   │      No  → ExpertStore（①）からSSDロード（コールド）→ キャッシュ投入
   │
   └─ ハイブリッドMoEブロック（③）で計算 → 次の層へ
```

| モジュール | ファイル | 役割 |
|---|---|---|
| ① ExpertStore | `elfmoon/expert_store.py` | expertを (層,番号) 単位でSSD保存・mmapロード |
| ② ResidentCache | `elfmoon/resident_cache.py` | バイト予算つき LRU 常駐キャッシュ |
| ③ StreamingMoE | `elfmoon/moe_block.py` / `stream_model.py` | router→cache/store→FFN のMoE計算 |
| ④ プリフェッチ | （未実装・実12GB機で実装予定） | コールド読みを計算と並行で隠す |

補助: `elfmoon/integrate.py`（実重み分解）, `elfmoon/hotlist.py`（使用頻度プロファイル）

---

## 動作環境

- Apple Silicon Mac（M系）/ macOS
- Python 3.11+（本開発は miniconda base の 3.13）
- 空きディスク 40GB 程度（モデル16GB + 分解済expert15GB）

---

## セットアップ（クリーンな状態から）

### 1. 依存パッケージ

```bash
pip install mlx mlx-lm "transformers==4.57.6" huggingface_hub hf_transfer
brew install aria2   # 任意（大容量DLの保険）
```

> ⚠️ **transformers は 4.57.6 を指定**。最新の 5.x だと mlx_lm が import エラーになる（既知の非互換）。

### 2. モデルをダウンロード（MLX 4bit, 約16GB）

```bash
# Xet経路は遅い/壊れることがあるので必ず無効化する
HF_HUB_DISABLE_XET=1 hf download mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
  --local-dir ./models/qwen3-coder-mlx
```

> `hf download` は完了時にハッシュ検証されるので破損しない。多接続DL（aria2 -x16 等）は
> HF の Xet CDN と相性が悪く**ファイル破損の実績あり**。単一ストリームが安全。

### 3. expert を分解（融合テンソル → per-expert, 約15GB）

```bash
cd elfmoon
python3 integrate.py split_all ../models/qwen3-coder-mlx
# → elfmoon/spike/real_store/ に 6144個のexpertファイル
#   elfmoon/spike/real_gates/ に 48個のrouter gate
```

---

## 使い方

### ElfMoon ストリーミングで生成

```bash
cd elfmoon

# 常駐expert数を指定（多いほど命中率↑・メモリ↑）
python3 stream_model.py 2800          # 短いプロンプトで生成
python3 stream_model.py 2800 long     # 長文脈プレフィルのデモ（958トークン）
```

出力例:
```
差し替え完了。常駐メモリ=0.87GB
func gcd(_ a: Int, _ b: Int) -> Int { ... }   ← 生成コード
Prompt: 958 tokens, 148 tokens-per-sec
Generation: 60 tokens, 13.3 tokens-per-sec
命中率=73.5% (hit=... miss=... 常駐=2800)
```

**常駐容量の目安**（expert 1個 ≈ 2.65MB）:
- `2800` ≈ 7.4GB（命中率〜85%、速度重視）
- `1200` ≈ 3.2GB（命中率〜67%、省メモリ）

生成するプロンプトを変えたい場合は `stream_model.py` 末尾の `prompt = ...` を編集。

### 参照: フルモデル（mlx_lm 標準, 全expert常駐 16GB）

```bash
python3 -c "from mlx_lm import load, generate; m,t=load('models/qwen3-coder-mlx'); print(generate(m,t,prompt='...',max_tokens=80,verbose=True))"
```

---

## 計測・検証ツール

| コマンド | 内容 |
|---|---|
| `elfmoon/test_moe.py` | モジュール②③の正しさ＋命中率→tok/s（合成データ） |
| `elfmoon/test_prefetch.py` | 容量→命中率の関係（合成データ） |
| `elfmoon/spike/expert_latency.py` | per-expert のロード/計算遅延ベンチ |
| `scripts/monitor.sh` | メモリ圧監視（別ターミナルで常駐） |
| `scripts/cap_ram.sh N` | 空きRAMを N GB に絞る（逼迫の再現。乱数mlock風） |
| `scripts/bench_ssd.sh` | SSD 実読み速度（ストリーミングの天井） |
| `scripts/run_coder.sh` | llama.cpp でのベースライン（要 GGUF・別途） |

---

## 開発状況

- ✅ エンジン①②③実装・実重みで動作（Qwen3-Coder-30B）
- ✅ デコードのハイブリッド融合バッチ化（〜15 t/s）
- ✅ プレフィルの expert-grouped 化（長文脈 148 t/s）
- ✅ 実Xcode共存で実用速度を実証
- ⬜ ④ 非同期プリフェッチ（実12GB機到着後に実装）
- ⬜ より大きいモデルへスケール（Qwen3-Next-80B / 128GB機で DeepSeek V4級）

詳細な設計と全実測は **[PHASE1_DESIGN.md](PHASE1_DESIGN.md)** を参照。

---

## ハマりどころメモ

- **HF ダウンロードが遅い/壊れる** → `HF_HUB_DISABLE_XET=1` で標準LFS経路にする。DL後は `shasum -a 256` を HF の `x-linked-etag` と照合。
- **mlx_lm が import できない** → `transformers==4.57.6` に固定。
- **`stream_model.py` が「何も起きない」ように見える** → 生成中。`verbose=True` でtok/sが出る。
- **メモリ不足でリブート？** → 本方式は mmap 主体なので原理的にOOMリブートしない（カーネルがページを捨てるだけ）。

---

## クレジット

- 着想元: [antirez/ds4 (DwarfStar)](https://github.com/antirez/ds4)
- 基盤: [MLX](https://github.com/ml-explore/mlx) / [mlx-lm](https://github.com/ml-explore/mlx-lm)
- モデル: [Qwen3-Coder-30B-A3B](https://huggingface.co/Qwen)（Alibaba Tongyi Lab）
</content>

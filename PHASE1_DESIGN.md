# ElfMoon フェーズ1 設計書

## 目的
Qwen3-Coder-30B-A3B（30B/active3B, 128experts/8active, 48層）を、**LLM常駐予算~10GB**（Xcode同時起動）で
**実用速度（目標10-30 t/s）**で動かす。DS4の常駐管理＋ストリーミング手法を **MLX** 上に実装する。

## ベースライン（2026-07-07 実測, llama.cpp, 24GB機）＝埋めるべきギャップ
| 構成 | 生成速度 | メモリ | 判定 |
|---|---|---|---|
| フル常駐 (-ngl 999) | **80 t/s** | 17.7GB | 速いが予算超過 |
| 素のexpert-CPUオフロード (-ncmoe 48) | **0.2 t/s** | 予算内 | 実用外 |
| **ElfMoon 目標** | **10-30 t/s** | **≤10GB** | ここを埋める |

## 設計を縛る2大事実（MLX実API調査で判明）
1. MLXに `MTLResidencySet` 相当なし。メモリは `set_wired_limit/set_memory_limit/set_cache_limit` の予算設定＋自動管理のみ。
2. mlx-lm はexpertを1個の融合テンソル(`switch_mlp`)で保持 → 個別expertの常駐/退避が自然にはできない。

→ **結論: expertを個別に分解し、自前で「常駐LRUキャッシュ＋SSDストリーム＋予測プリフェッチ」層を実装する。**
　MLXの残りの部分（DeltaNet不要な標準MoE・attention・量子化matmul）はそのまま活用。

---

## アーキテクチャ（4モジュール）

### ① 重みプリプロセッサ（オフライン）
- 融合expertテンソルを **per-expert（layer,expert）単位の配置**に分解し、**mmapで個別スライスを高速ロード**できる単一ファイル＋オフセット表を作る。
- 量子化はMLXのQ4を使用（またはGGUFから変換）。**非expert（attention/router/embed/norm）は別途、常に常駐**。
- DS4の `deepseek4-quantize.c` 相当。まずは既存MLX 4bit quantを分解する形でよい。

### ② 常駐マネージャ（Resident Manager）
- 起動時 `set_wired_limit` で GPU予算を確保（例: 8GB）。`get_active_memory()` で監視。
- **非expert重み＝常駐固定**。残り予算を **ホットexpertのLRUキャッシュ**（`dict[(layer,expert)] -> mx.array`）に割当。
- キャッシュ満杯時はLRUで退避（参照を落とし `clear_cache()`）。DS4の `ds4_ssd_auto_cache_plan`（予算×4/5, 非routed差引, expert数算出）を移植。

### ③ カスタムMoEブロック（mlx-lm の SwitchGLU を置換）
- router（gate→softmax→top-k）で当該トークンの必要expert IDを取得。
- 各expert: キャッシュにあれば即使用、無ければ **mmapファイルから該当スライスをmx.arrayでロード**→キャッシュ投入。
- gather→FFN計算→スコア加重集約（mlx-lmと同じ数式）。**融合ではなく必要分だけ материализ**。

### ④ プリフェッチャ（速度の肝）
- **層先読み**: 次層のrouterを先に評価し、必要expertをバックグラウンドスレッドでSSD→キャッシュへ先読み。I/Oと計算をオーバーラップ。
- **ホットリスト起動時プライム**: Qwen3-Coder用に実走プロファイルを取り、頻出expertでキャッシュを温める（DS4 `ds4_streaming_hotlist.inc` 相当）。

---

## トークン毎データフロー
```
入力 → [非expert常駐:attention等] → 各層:
   router → top-k expert ID
        → 常駐キャッシュ命中?  Yes→即計算 /  No→mmapロード→計算(+キャッシュ)
        → 次層expertをプリフェッチ（並行）
   → 集約 → 次層 → ... → logits
```

## メモリ予算（概算, 予算10GB）
- 非expert（Q4）: ~1GB（常駐固定）
- KV＋activations: ~1GB
- **ホットexpertキャッシュ: ~8GB** → expert1個~2.4MB(Q4)換算で約3300/6144個(≈54%)常駐
- 毎トークン: 8×48=384 expert-load。命中54%なら実SSD読み~420MB/token。**プリフェッチで隠せるかが速度を決める**。

## 最大リスクと最初のスパイク（作り込む前に測る）
**крux: per-expertのmmapロード遅延とLRU命中率で、本当に10-30 t/sに届くか。**
- スパイク: ②③の最小版（プリフェッチ無し）を実装し、「命中率→tok/s」カーブを実測。
- ここが崩れる（例: ロード遅延が支配的で数t/s止まり）なら、量子化を下げる/キャッシュを増やす/プリフェッチ必須度を判断。
- **プリフェッチ無しで既に0.2 t/sを大きく超えれば、方向は正しい。**

## スパイク実測（2026-07-07, MLX 0.31.2, 合成expert）
- expert 1個 2.95MB(Q4)。ロード+計算 **0.30ms/expert**（計算0.17+ロード0.13ms, ページキャッシュ温）。
- tok/s試算(384op/token): 命中0%→8.6, 50%→11, 70%→12.5, 90%→14.3 t/s。**llama.cpp 0.2 t/sの40-70倍＝方向は正しい。**
- **真コールド実測(purge後)**: ロード寄与 温0.13ms→**コールド1.07ms/expert(8倍)**。計算0.15ms。
  tok/s: 命中0%→2.14, 50%→3.82, 70%→5.57, **90%→10.27**。最悪の全コールドでもllama.cpp(0.2)の10倍。
- **★確定した工学課題**: 命中率がすべて。**「10GB予算内で命中率~90%維持＋1.07msコールドをプリフェッチで隠す」**が成否。
  activeなexpertは偏るので全体54%常駐でもactive90%命中は達成可能。2レバー=①ホットリスト+LRUで命中率↑ ②層先読みプリフェッチでコールド隠蔽。
- **注意②(設計反映)**: per-expert個別matmul(batch=1)はカーネル起動overheadで計算のみ~15 t/s頭打ち(融合80の1/5)。
  → **ホットexpertは融合テンソルで一括計算(SwitchGLU式)、コールドのみ個別ロード+計算のハイブリッドにする。**
  ③カスタムMoEブロックはこの「常駐融合バッチ＋コールド個別」二経路で実装する。

## 実重み統合の進捗（2026-07-07）
- MLX版 `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`(16GB) DL済 → `models/qwen3-coder-mlx`。
- 実アーキ確認: 48層/hidden2048/128experts/8top-k/moe_inter768/decoder_sparse_step1/norm_topk_prob=True。
  量子化: expert=group64/**bit4（ExpertStoreと一致→スライスのみ）**, router gate=group64/bit8。
- テンソルキー: `model.layers.{l}.mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`([128,...]融合), `...mlp.gate.*`(router)。
- **モジュール①(実重み分解)実装・検証済**(`integrate.py`): mlx_lm非依存でsafetensors直読み→per-expert分解。
  **layer0往復検証 誤差0.00e+00**。実expert=**2.65MB**。全6144experts≈16.3GB。10GB予算で約3770(61%)常駐可。
- **既知ブロッカー**: フルモデル実行(attention/embed/sampling)に要る `mlx_lm` が transformers版不整合でimport不可
  (`AttributeError: 'str' has no '__module__'`)。フル統合前に要修正（transformers pin か mlx-lm更新）。

## ★実重みエンドツーエンド成功（2026-07-07）＝ElfMoon構想の実証
- mlx_lm修復: `transformers==4.57.6` で mlx_lm 0.31.3 が動作（metadataは5.0+要求だが実際は4.57で可）。
- 全6144expert分解済(`integrate.py split_all`, 15GB → `elfmoon/spike/real_store`)。
- `elfmoon/stream_model.py`: 全層mlpをStreamingMoEに差し替え、融合switch_mlp解放。
- **結果(常駐1200exp): 常駐メモリ 16GB→0.87GB、正しいGCDコード生成、命中率66.6%、80tok/9.6s≈8t/s（llama.cpp素offload 0.2の40倍）。**
- 実行: `cd elfmoon && python3 stream_model.py <常駐expert数>`。参照(フル)=80t/s/16GB。
- 残: 命中率向上(容量↑)、ハイブリッド融合バッチ化(現状per-token Pythonループ)、真コールド計測、Xcode共存、④非同期プリフェッチ。

## 実装状況
- ✅ **モジュール①②③ 実装済**（`elfmoon/expert_store.py` `resident_cache.py` `moe_block.py`, テスト`test_moe.py`）
- ✅ **正しさ検証: MoEBlock == 全常駐参照 で誤差0.00e+00**（合成expert, 2026-07-07）
- ✅ キャッシュ挙動実証: 容量↑→命中率↑(0→85%)→tok/s↑(142→319, 4層/温)。定性的に設計通り。
- **重要な設計発見**: デコードは毎トークン全層を通るため、キャッシュ容量 < 1トークンの作業集合(top_k×層数) だと総取っ替え(命中0%)。
  → **常駐予算は“全48層分のホットexpert”を同時保持する必要がある**（(layer,expert)キーで全層またぐLRU）。
- 注: 上記tok/sは4層/温/均一ルーティングの定性値。実物は48層/SSDコールド/強偏在で実効10-30t/s帯見込み。

## モジュール④の検証結果と再定義（2026-07-07, 合成強偏在）
- **命中率は容量で素直に上昇**: 作業集合(層×top_k)の**1.5倍常駐で96%+、2倍で98%+**（`test_prefetch.py`）。
- **実モデル換算**: 作業集合=48×8=384。**10GB予算は約3300expert(≈作業集合8.6倍)常駐可能→命中率は余裕で高い→10-14t/s現実的。最大リスク(命中率)が低リスクに転落。**
- **静的ホットリスト・プライムは無意味**（96.5%→96.7%）: LRUが定常でホット集合を自動捕捉するため。→ **④から静的プライムは外す**。
- **④の残る価値**: 残り数%のコールドミス(1.07ms)を**非同期プリフェッチ(cold-load I/OとGPU計算のオーバーラップ)**で隠す。命中率が既に高いので**副次的最適化**。逐次依存(層L+1のrouterは層L出力依存)のため層先読みは限定的。実装は「1層内の複数コールドを並行ロード」「バックグラウンドthreadでページキャッシュ温存」が現実的。

## 残ビルド順序
1. mlx-lm導入＋Qwen3-Coder-30B MLX 4bitで**標準動作**を確認（正解リファレンス）※実重み17GB DL要
2. ①プリプロセッサを実重み対応（mlx-lmの融合switch_mlpを(layer,expert)分解）
3. ④プリフェッチ（層先読み）＋ホットリスト起動時プライム → コールド1.07ms隠蔽
4. 真コールド＋48層＋実偏在で「命中率→tok/s」実測、予算/量子化チューニング
5. Xcode同時での実運用検証
6. 将来: Qwen3-Next-80B / 128GB機へスケール
```
```

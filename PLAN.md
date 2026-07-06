# ElfMoon4

24GB / 12GB クラスの Apple Silicon Mac で **Qwen3.6-35B-A3B（MoE, 35B total / 3B active / 256 experts, 8+1 routed+shared）** を
DwarfStar(ds4) 方式 = **expert ストリーミング**で実用的に動かすための推論エンジンとその確立手順。

> antirez の DwarfStar が「128GB Mac で DeepSeek V4 を動かす」なら、ElfMoon4 はその**縮小プロト**。
> ここで作る機構はそのまま 128GB 機 × DeepSeek V4 に移植することを狙う。

## なぜ正攻法だと OS が死ぬか
量子化モデルを**丸ごと匿名メモリ（malloc, 破棄不能）**に展開する → 物理RAM超過 →
激しくスワップ → WindowServer/カーネルがハング → **強制リブート**。

## 中核アイデア
重みを匿名メモリに置かず、**mmap（ファイルバック＝破棄可能ページ）**で参照する。
MoE は 1 token で 3B(8+1/256 experts) しか触らないので、**hot expert だけ常駐・cold は SSD からストリーム**。
→ 原理的に OOM で死なない（遅くなるだけ）。

## モデル / 量子化
- フェーズ0: **GGUF Q2_K**（≈10GB, 混合kクオント。均一2bitより高品質）
  - repo: `unsloth/Qwen3.6-35B-A3B-GGUF`（Q2_K）
- フェーズ1: 混合bit MLX（attention/router/shared=4〜6bit常駐, expert=2bitストリーム）を自前生成

## 検証で確定した重要事実（2026-07-06, 24GB機）
- Q2_K(11GB) を `-ngl 999` フル常駐: **56 t/s で動作・OOMリブートなし**（24GBはデフォルトMetal予算が大きいため）
- 匿名メモリのバラストでは**モデルを絞れない**: モデルはGPUワイヤード、バラストはCPU匿名＋swap＝別プール。
  LRUがホットなモデルを常駐させ冷たいバラストをswap → 速度落ちず（=シミュレーション不能）
- **`iogpu.wired_limit_mb=8192`（実12GB機のデフォルト予算を再現）で `-ngl 999` → 即OOM**
  （`kIOGPUCommandBufferCallbackErrorOutOfMemory`）
  → **結論: `-ngl` は重みをGPUにワイヤ固定。予算超過はストリームせずハードOOM。**
  → **実12GB機では素の llama.cpp -ngl 999 + 11GB Q2_K もOOMする公算大**
- **フェーズ1の必須設計要件**: Metalに全モデルをワイヤさせない。GPU上のexpert常駐を自前管理し、
  hotだけワイヤ・coldはSSD/CPUからストリーム（=DwarfStarがMetalで実装している方式）

## フェーズ
- **フェーズ0（安全な土台・既存ツール）**: llama.cpp + mmap
  1. SSD 実読み速度を計測（`scripts/bench_ssd.sh`）— ストリーミング速度の天井が決まる
  2. Q2_K を丸ごと常駐で動作確認（死なない事実 & 正解リファレンス）
  3. **空きRAMを12GBに絞って**再実行（`scripts/cap_ram.sh` バラスト）→ 実12GB機の挙動を先取り、
     mmap ページキャッシュ任せの素の tok/s とスラッシング量を計測
- **フェーズ1（ElfMoon4 本体）**: MoE-aware 予測プリフェッチ + hot expert ピン留めで
  カーネル盲目LRUを上回るエンジンを MLX / C+Metal で実装

## 安全プロトコル（マシンをブリックさせない）
- 必ず mmap 経路のみ（`--no-mmap` 禁止 / 匿名フル展開禁止）
- 別ターミナルで `scripts/monitor.sh`（memory_pressure 監視）を常時起動
- 12GB 再現はまず控えめなバラストから、段階的に絞る

## 実機
開発・計測は手元の **24GB MacBook Pro**。実 12GB 機は待たず、RAM を絞って再現。

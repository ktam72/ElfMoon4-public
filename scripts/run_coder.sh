#!/usr/bin/env bash
# Qwen3-Coder-30B-A3B-Instruct を llama.cpp で走らせる（計測用ベースライン）。
# 使い方:
#   ./scripts/run_coder.sh                # フル常駐（リファレンス速度）
#   MODE=stream ./scripts/run_coder.sh    # expert-CPUオフロード＝ストリーミング素の版
#   NCMOE=48 MODE=stream ./scripts/run_coder.sh   # オフロード層数を調整
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=$(find ./models -name '*Qwen3-Coder-30B*UD-Q4_K_XL*.gguf' | head -1)
[[ -z "${MODEL}" ]] && { echo "!! モデル未取得。download2.log を確認"; exit 1; }

PROMPT="${1:-Swiftで2つのIntを受け取り最大公約数を返す関数gcdを書いて。コードのみ。}"
MODE="${MODE:-resident}"
NCMOE="${NCMOE:-48}"   # expertをCPU側へ回す層数（多いほど常駐↓・速度↓）

COMMON=(-m "$MODEL" -c 4096 -n 256 -st -p "$PROMPT")

echo "==> モデル: $MODEL"
echo "==> モード: $MODE  (別ターミナルで monitor.sh 推奨)"

if [[ "$MODE" == "stream" ]]; then
  # expertテンソルをCPU(=mmapページング)に置き、GPUには載せない＝Metalワイヤ回避。
  # これが「予算内でexpertストリーム」の既存ツール版（ElfMoon素の版）。
  echo "==> expert-CPUオフロード: --n-cpu-moe ${NCMOE}"
  llama-cli "${COMMON[@]}" -ngl 999 --n-cpu-moe "$NCMOE"
else
  # 全層GPU。Xcode閉じた状態のフル常駐リファレンス速度。
  llama-cli "${COMMON[@]}" -ngl 999
fi

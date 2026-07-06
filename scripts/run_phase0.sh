#!/usr/bin/env bash
# フェーズ0: llama.cpp + mmap で Qwen3.6-35B-A3B を動かす。
# mmap はデフォルト ON（--no-mmap は絶対に付けない）。
# 前提: brew install llama.cpp / モデルは ./models に取得済み
set -euo pipefail

cd "$(dirname "$0")/.."
PROMPT="${1:-日本語で自己紹介して。あなたは誰？}"

# ローカルモデルを優先（分割shardは -00001-of- を渡せば llama.cpp が自動連結）
# find で set -e に引っかからないよう安全に検出
MODEL=$(find ./models -name '*UD-Q2_K_XL*00001-of-*.gguf' 2>/dev/null | head -1)
[[ -z "${MODEL}" ]] && MODEL=$(find ./models -name '*UD-Q2_K_XL*.gguf' 2>/dev/null | head -1)
if [[ -z "${MODEL:-}" ]]; then
  echo "!! ./models に UD-Q2_K_XL の gguf が見つかりません。先にダウンロードを完了してください。"
  echo "   hf download unsloth/Qwen3.6-35B-A3B-GGUF --include '*UD-Q2_K_XL*' --local-dir ./models"
  exit 1
fi

echo "==> モデル: ${MODEL}"
echo "==> mmap ON / メモリ圧は別ターミナルの monitor.sh で監視すること"

# -ngl 999 で全層 Metal。--no-mmap は付けない。
# -st: single-turn。1ターン生成して終了（対話モードの > 入力待ちを避ける）
llama-cli \
  -m "${MODEL}" \
  -ngl 999 \
  -c 4096 \
  -n 256 \
  -st \
  -p "${PROMPT}"

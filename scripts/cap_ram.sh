#!/usr/bin/env bash
# 24GB 機で「空きRAMを実質 N GB に絞る」ためのホット・バラスト。
# 指定GBを確保し、定期的に触り続けて常駐させ、モデルとページを奪い合わせる。
# 使い方: ./cap_ram.sh 12   → 空きを ~12GB 想定に絞る（24-12=12GB を占有）
# 別ターミナルで起動しっぱなしにして、本体推論を走らせる。Ctrl-C で解放。
#
# 注意: これは近似。より厳密に GPU 側を絞るなら:
#   sudo sysctl iogpu.wired_limit_mb=<MB>   （Metal がワイヤできる上限を制限）
set -euo pipefail

TARGET_FREE_GB="${1:-12}"
TOTAL_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
BALLAST_GB=$(( TOTAL_GB - TARGET_FREE_GB ))
if (( BALLAST_GB <= 0 )); then echo "総RAM ${TOTAL_GB}GB。バラスト不要"; exit 0; fi

echo "==> 総RAM=${TOTAL_GB}GB / 目標空き=${TARGET_FREE_GB}GB → バラスト ${BALLAST_GB}GB を常駐"
python3 - "$BALLAST_GB" <<'PY'
import sys, os, time
gb = int(sys.argv[1])
GB = 1024*1024*1024
BLK = 64*1024*1024  # 64MBずつ乱数で埋める（メモリ圧縮を無効化＝真の圧力を作る）
chunks = []
for i in range(gb):
    b = bytearray(GB)
    for off in range(0, GB, BLK):
        b[off:off+BLK] = os.urandom(BLK)  # 非圧縮な乱数 → 実RAMを必ず消費
    chunks.append(b)
    print(f"  ballast {i+1}/{gb} GB 常駐（乱数）", flush=True)
print("==> バラスト保持中（Ctrl-C で解放）", flush=True)
# 定期的に触り続けてスワップアウト/圧縮退避を防ぐ
while True:
    for b in chunks:
        b[0] = (b[0] + 1) & 0xff
        _ = b[len(b)//2]
    time.sleep(1)
PY

#!/usr/bin/env bash
# SSD のシーケンシャル読み速度を実測する（ストリーミング速度の天井）。
# ページキャッシュを避けるため purge してから計測する。
set -euo pipefail

SIZE_GB="${1:-8}"
TMP="${TMPDIR:-/tmp}/elfmoon_ssd_bench.bin"

echo "==> ${SIZE_GB}GB のテストファイルを書き込み: $TMP"
dd if=/dev/zero of="$TMP" bs=1m count=$((SIZE_GB * 1024)) 2>&1 | tail -1

echo "==> ページキャッシュを破棄（sudo purge）"
sudo purge

echo "==> シーケンシャル読み計測"
# macOS の dd は経過時間と速度を stderr に出す
dd if="$TMP" of=/dev/null bs=1m 2>&1 | tail -1

rm -f "$TMP"
echo "==> 完了。上の 'bytes/sec' がおおよその SSD 読み天井。"

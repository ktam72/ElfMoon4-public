#!/usr/bin/env bash
# 別ターミナルで常時起動しておく。メモリ圧を監視し、危険域で警告する。
# green=正常 / warn=スワップ増加中 / critical=リブート危険。
set -uo pipefail

echo "==> memory_pressure 監視開始（Ctrl-C で終了）"
while true; do
  free_pct=$(memory_pressure 2>/dev/null | awk -F: '/System-wide memory free percentage/{gsub(/[ %]/,"",$2); print $2}')
  swap=$(sysctl -n vm.swapusage 2>/dev/null | awk '{print $6}')
  ts=$(date +%H:%M:%S)
  if [[ -z "${free_pct:-}" ]]; then free_pct=0; fi
  if   (( free_pct < 5 )); then tag="CRITICAL";
  elif (( free_pct < 15 )); then tag="warn";
  else tag="green"; fi
  printf "%s  free=%s%%  swap_used=%s  [%s]\n" "$ts" "$free_pct" "$swap" "$tag"
  sleep 2
done

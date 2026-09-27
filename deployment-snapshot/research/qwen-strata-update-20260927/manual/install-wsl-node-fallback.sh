#!/usr/bin/env bash
set -euo pipefail
stage=/mnt/c/Users/costa/Documents/Project_ANTIREZ/research/qwen-strata-agent-20260927/manual/pi-global-installed.sh
target=/home/costapad/.local/bin/pi
bash -n "$stage"
if cmp -s "$stage" "$target"; then exit 0; fi
cmp -s "$target" <(sed '/^# BEGIN STRATA NODE FALLBACK/,/^# END STRATA NODE FALLBACK/d' "$stage") || {
  echo 'Unexpected wrapper changes; nothing overwritten' >&2; exit 1;
}
backup=/home/costapad/.local/bin/pi.before-ista-strata-node-fallback-20260927.bak
[[ ! -e "$backup" ]] || { echo 'Node-fallback backup exists; not overwritten' >&2; exit 1; }
cp -p -- "$target" "$backup"
install -m755 -- "$stage" "$target"
cmp -s "$stage" "$target"
echo 'Node fallback added for noninteractive WSL only; existing Node choices and legacy branches preserved'

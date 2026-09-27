#!/usr/bin/env bash
set -euo pipefail
[[ "$(id -un)" == costapad ]] || { echo 'Wrong WSL user' >&2; exit 1; }
stage=/mnt/c/Users/costa/Documents/Project_ANTIREZ/research/qwen-strata-agent-20260927/manual/pi-global-installed.sh
target=/home/costapad/.local/bin/pi
[[ -f "$target" && ! -L "$target" ]] || { echo 'Unexpected global pi file' >&2; exit 1; }
bash -n "$stage"
if cmp -s "$stage" "$target"; then echo 'Global Pi Strata wrapper already installed'; exit 0; fi
# Fail closed if any unrelated global wrapper branch changed since discovery.
cmp -s "$target" <(sed -n '1,2p;/^# BEGIN THINKINGCAP DIRECT PROVIDER/,$p' "$stage") || {
  echo 'Global Pi wrapper differs from inspected legacy source; nothing changed' >&2; exit 1;
}
backup=/home/costapad/.local/bin/pi.before-ista-strata-20260927.bak
[[ ! -e "$backup" ]] || { echo 'Backup already exists; nothing changed' >&2; exit 1; }
sha256sum /home/costapad/.pi/agent/models.json /home/costapad/.pi/agent/settings.json
cp -p -- "$target" "$backup"
install -m 755 -- "$stage" "$target"
cmp -s "$stage" "$target"
sha256sum /home/costapad/.pi/agent/models.json /home/costapad/.pi/agent/settings.json
printf 'Global Pi updated; provider/settings/session files unchanged. Backup: %s\n' "$backup"

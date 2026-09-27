#!/usr/bin/env bash
set -euo pipefail
[[ "$(id -un)" == costapad ]] || { echo 'Wrong WSL user' >&2; exit 1; }
stage=/mnt/c/Users/costa/Documents/Project_ANTIREZ/research/qwen-strata-agent-20260927/manual/pi-global-installed.sh
target=/home/costapad/.local/bin/pi
[[ -f "$target" && ! -L "$target" ]] || { echo 'Unexpected global Pi file' >&2; exit 1; }
bash -n "$stage"
if cmp -s "$stage" "$target"; then echo 'Vision wrapper already installed'; exit 0; fi
# Exact preceding installed wrapper, no overwrite of unrelated user changes.
actual=$(sha256sum "$target" | cut -d' ' -f1)
[[ "$actual" == a9d99fd14a4bc620b8f75f6d089cfa0e7e605c5e8d948cec7533d9472e4477ed ]] || { echo 'Pi changed since inspected handoff; nothing changed' >&2; exit 1; }
backup=/home/costapad/.local/bin/pi.before-ista-strata-vision-20260927.bak
[[ ! -e "$backup" ]] || { echo 'Vision backup already exists; nothing changed' >&2; exit 1; }
sha256sum /home/costapad/.pi/agent/models.json /home/costapad/.pi/agent/settings.json
cp -p -- "$target" "$backup"
install -m 755 -- "$stage" "$target"
cmp -s "$stage" "$target"
sha256sum /home/costapad/.pi/agent/models.json /home/costapad/.pi/agent/settings.json
printf 'Same global Pi: CPU vision capability overlay. Backup: %s\n' "$backup"

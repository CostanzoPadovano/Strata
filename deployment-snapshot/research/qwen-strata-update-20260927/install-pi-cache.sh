#!/usr/bin/env bash
# Promotion step only, after native/cache/vision and staged actual Pi admission.
set -euo pipefail
[[ "$(id -un)" == costapad ]] || { echo 'Unexpected WSL user' >&2; exit 1; }
stage=/mnt/c/Users/costa/Documents/Project_ANTIREZ/research/qwen-strata-update-20260927/manual/pi-global-installed.sh
target=/home/costapad/.local/bin/pi
backup=/home/costapad/.local/bin/pi.before-cache-idle-20260927.bak
expected=8ae336c66ff1b76c0f2617ab079fd06500f4ba99f7784277874be261362cc4a8
[[ -f "$target" && ! -L "$target" ]] || { echo 'Unexpected Pi target' >&2; exit 1; }
bash -n "$stage"
if cmp -s "$stage" "$target"; then echo 'Cache wrapper already installed'; exit 0; fi
actual=$(sha256sum "$target" | cut -d' ' -f1)
[[ "$actual" == "$expected" ]] || { echo 'Pi changed since inspection; no overwrite' >&2; exit 1; }
[[ ! -e "$backup" ]] || { echo 'Rollback backup already exists; no overwrite' >&2; exit 1; }
before=$(sha256sum /home/costapad/.pi/agent/models.json /home/costapad/.pi/agent/settings.json)
cp -p -- "$target" "$backup"
install -m 755 -- "$stage" "$target"
cmp -s "$stage" "$target"
after=$(sha256sum /home/costapad/.pi/agent/models.json /home/costapad/.pi/agent/settings.json)
[[ "$before" == "$after" ]] || { echo 'Pi configuration changed unexpectedly' >&2; exit 1; }
printf 'Same global Pi updated. Models/settings unchanged. Rollback: %s\n' "$backup"
sha256sum "$target" "$backup"

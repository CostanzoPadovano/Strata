#!/usr/bin/env bash
set -euo pipefail
[[ "$(id -un)" == costapad ]] || { echo 'Wrong WSL user' >&2; exit 1; }
stage=/mnt/c/Users/costa/Documents/Project_ANTIREZ/research/qwen-strata-agent-20260927/manual/pi-global-installed.sh
target=/home/costapad/.local/bin/pi
[[ -f "$target" && ! -L "$target" ]] || { echo 'Unexpected Pi wrapper' >&2; exit 1; }
bash -n "$stage"
if cmp -s "$stage" "$target"; then echo 'Independent Pi wrapper already installed'; exit 0; fi
actual=$(sha256sum "$target" | cut -d' ' -f1)
[[ "$actual" == f5ee81db09f37fa6c0c64fc8edc9b5dfc7d6fa8cbec7895a9f4b4bf2e8756cdf ]] || {
  echo 'Pi changed since inspection; nothing overwritten' >&2; exit 1;
}
backup=/home/costapad/.local/bin/pi.before-independent-providers-20260927.bak
[[ ! -e "$backup" ]] || { echo 'Backup exists; nothing overwritten' >&2; exit 1; }
cp -p -- "$target" "$backup"
install -m 755 -- "$stage" "$target"
cmp -s "$stage" "$target"
echo "Independent provider wrapper installed; backup: $backup"

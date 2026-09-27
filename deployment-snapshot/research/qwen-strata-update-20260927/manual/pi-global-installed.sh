#!/usr/bin/env bash
set -euo pipefail
# PI INDEPENDENT PROVIDERS V1
# Optional adapters must NEVER gate the Pi UI on a running inference server.
strata_root=/mnt/c/Users/costa/Documents/Project_ANTIREZ/research/qwen-strata-update-20260927/manual
strata_node=/home/costapad/.nvm/versions/node/v22.23.0/bin/node
if ! command -v node >/dev/null 2>&1; then export PATH="/home/costapad/.nvm/versions/node/v22.23.0/bin:$PATH"; fi
unset ISTA_STRATA_BASE_URL ISTA_STRATA_VISION
pi_extras=()
# BEGIN STRATA READONLY CLI BYPASS
for pi_argument in "$@"; do
  [[ "$pi_argument" == -- ]] && break
  case "$pi_argument" in
    --help|-h|--version|-v|--list-models|--export)
      exec "$HOME/.local/bin/pi-qwen" --pi-default "$@" ;;
  esac
done
case "${1:-}" in
  install|remove|uninstall|update|list|config|auth)
    exec "$HOME/.local/bin/pi-qwen" --pi-default "$@" ;;
esac
# END STRATA READONLY CLI BYPASS
# BEGIN ISTA STRATA MANUAL - runtime-only overlay of the SAME global provider.
if strata_route="$(timeout --kill-after=1s 6s "$strata_node" "$strata_root/pi_global_dispatch.mjs" route "$@")"; then
  mapfile -t strata_fields <<< "$strata_route"
  if [[ "${#strata_fields[@]}" == 3 && "${strata_fields[0]}" =~ ^http://172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}:8038/v1$ \
        && "${strata_fields[1]}" =~ ^[01]$ && "${strata_fields[2]}" =~ ^[01]$ ]]; then
    export ISTA_STRATA_BASE_URL="${strata_fields[0]}" ISTA_STRATA_VISION="${strata_fields[1]}"
    # SAME-provider overlay also allows choosing Flash later in Pi's selector.
    pi_extras=(-e "$strata_root/pi_strata.mjs")
    if [[ "${strata_fields[2]}" == 1 ]]; then
      printf 'ISTA / Strata 98K pronto: stesso profilo, output nel contesto residuo.\n' >&2
      exec "$HOME/.local/bin/pi-qwen" --pi-default \
        --model local-qwen38/qwen3.8-flash-next-local --thinking xhigh "${pi_extras[@]}" "$@"
    fi
  else
    printf 'Avviso: rilevamento Strata non valido; avvio normale di Pi.\n' >&2
  fi
fi
# END ISTA STRATA MANUAL
# BEGIN THINKINGCAP DIRECT PROVIDER
if timeout 3s node /mnt/c/MYPROJECT/TEST_QWEN/scripts/configure_pi_thinkingcap_wsl.mjs --is-selected "$@"; then
  # Already configured localhost provider: no self-rewrite on every startup.
  if ! timeout --kill-after=1s 8s bash /mnt/c/MYPROJECT/TEST_QWEN/scripts/ensure_thinkingcap_wsl_bridge.sh >/dev/null 2>&1; then
    printf 'Avviso: ponte ThinkingCap non disponibile; puoi scegliere un altro modello in Pi.\n' >&2
  fi
  exec "$HOME/.local/bin/pi-qwen" --pi-default "${pi_extras[@]}" --thinking xhigh "$@"
fi
# END THINKINGCAP DIRECT PROVIDER
# BEGIN GSQ DIRECT PROVIDER
if timeout 3s python3 /mnt/c/MYPROJECT/TEST_QWEN/scripts/configure_pi_gsq.py --is-selected "$@"; then
  if ! timeout 6s python3 /mnt/c/MYPROJECT/TEST_QWEN/scripts/configure_pi_gsq.py --refresh; then
    printf 'Avviso: aggiornamento indirizzo GSQ non riuscito; avvio normale di Pi.\n' >&2
  fi
  exec "$HOME/.local/bin/pi-qwen" --pi-default "${pi_extras[@]}" "$@"
fi
# END GSQ DIRECT PROVIDER
# BEGIN SC5 DIRECT PROVIDER
if timeout 3s python3 "$HOME/.local/bin/pi-sc5-config" --is-selected "$@"; then
  if ! timeout 6s python3 "$HOME/.local/bin/pi-sc5-config" --refresh; then
    printf 'Avviso: aggiornamento indirizzo SC5 non riuscito; avvio normale di Pi.\n' >&2
  fi
  exec "$HOME/.local/bin/pi-qwen" --pi-default "${pi_extras[@]}" "$@"
fi
# END SC5 DIRECT PROVIDER
# Preserve normal global Pi arguments, working directory, settings and resources.
# Bridge setup never loads a model; failure must not prevent using the 27B.
if timeout 3s "$strata_node" "$strata_root/pi_global_dispatch.mjs" legacy-selected "$@"; then
    if ! timeout --kill-after=1s 8s /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe \
      -NoProfile -NonInteractive -ExecutionPolicy Bypass \
      -File 'C:\Users\costa\Documents\Project_ANTIREZ\scripts\start_qwen38_vision_proxy.ps1' >/dev/null 2>&1; then
      printf 'Avviso: ponte Flash non disponibile; gli altri provider Pi restano utilizzabili.\n' >&2
    fi
fi
# pi-qwen refreshes private gateway addresses and invokes the installed Pi.
exec "$HOME/.local/bin/pi-qwen" --pi-default "${pi_extras[@]}" "$@"

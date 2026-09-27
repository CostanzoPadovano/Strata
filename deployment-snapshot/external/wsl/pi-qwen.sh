#!/usr/bin/env bash
set -euo pipefail

readonly MODEL_SCOPE='local-qwen*/*,local-agentworld35b/*'

declare -Ar PROFILES=(
  [q4-general]='local-qwen38-decode/qwen3.8-27b-q4kxl-decode-150k'
  [qvir1]='local-qwen38-router/qwen3.8-27b-hybrid-vision-tensor-dflash2-nvfp4-ngram-mod-150k'
  [qvir1-dflash]='local-qwen38-router/qwen3.8-27b-hybrid-vision-tensor-dflash2-nvfp4-150k'
  [q4-dflash]='local-qwen38-router/qwen3.8-27b-unsloth-q4kxl-vision-tensor-dflash2-nvfp4-150k'
  [q4-lookup]='local-qwen38-router/qwen3.8-27b-unsloth-q4kxl-vision-tensor-lookup-dflash2-nvfp4-150k'
  [q4-mtp]='local-qwen38-router/qwen3.8-27b-unsloth-q4kxl-vision-150k'
  [hybrid-fast]='local-qwen38-router/qwen3.8-27b-hybrid-fast-mtp-98k'
  [hybrid-safe]='local-qwen38-router/qwen3.8-27b-hybrid-safe-98k'
  [hybrid-vision]='local-qwen38-router/qwen3.8-27b-hybrid-fast-mtp-vision-98k'
  [q38-flash]='local-qwen38/qwen3.8-flash-next-local'
  [q36]='local-qwen27b-mtp-tq/qwen3.6-27b-mtp-turboquant'
  [q36-vision]='local-qwen27b-nvfp4-mtp-vision/qwen3.6-27b-nvidia-nvfp4-mtp-vision'
  [q36-262k]='local-qwen27b-mtp-tq-262k/qwen3.6-27b-mtp-turboquant-262k-draft64k'
  [q36-35b]='local-qwen35b-mtp-tq/qwen3.6-35b-a3b-thinking'
  [q35-9b]='local-qwen9b-mtp-tq/Qwen3.5-9B-BF16-thinking-stable'
  [agentworld]='local-agentworld35b/qwen-agentworld-35b-a3b-q4-k-xl'
)

declare -Ar COMMAND_PROFILES=(
  [pi-qwen38]='q4-general'
  [pi-qwen38-fast]='hybrid-fast'
  [pi-qwen38-safe]='hybrid-safe'
  [pi-qwen38-vision]='hybrid-vision'
  [pi-qwen38-q4]='q4-general'
  [pi-qwen38-flash]='q38-flash'
  [pi-qwen36]='q36'
  [pi-qwen36-vision]='q36-vision'
  [pi-qwen36-262k]='q36-262k'
  [pi-qwen36-35b]='q36-35b'
  [pi-qwen35-9b]='q35-9b'
)

usage() {
  cat <<'EOF'
Uso:
  pi-qwen [profilo] [opzioni Pi o prompt]
  pi-qwen --profiles
  pi-qwen --list-models

Profili consigliati:
  q4-general     Qwen3.8 Q4_K_XL + R0 General DFlash2 Vision 150K (predefinito)
  qvir1          Qwen3.8 Hybrid + DFlash2 + Ngram 150K
  qvir1-dflash   Qwen3.8 Hybrid + DFlash2 150K
  q4-dflash      Qwen3.8 Q4_K_XL + DFlash2 Vision 150K
  q4-lookup      Qwen3.8 Q4_K_XL + Lookup + DFlash2 Vision 150K
  q4-mtp         Qwen3.8 Q4_K_XL + MTP Vision 150K
  hybrid-fast    Qwen3.8 Hybrid + MTP 98K
  hybrid-safe    Qwen3.8 Hybrid senza speculative decoding 98K
  hybrid-vision  Qwen3.8 Hybrid + MTP Vision 98K
  q38-flash      Qwen3.8 Flash Next locale
  q36            Qwen3.6 27B MTP TurboQuant 192K
  q36-vision     Qwen3.6 27B NVFP4 MTP Vision 214K
  q36-262k       Qwen3.6 27B MTP TurboQuant 262K
  q36-35b        Qwen3.6 35B A3B MTP TurboQuant 220K
  q35-9b         Qwen3.5 9B BF16 MTP TurboQuant 150K
  agentworld     Qwen-AgentWorld 35B A3B 168K

È possibile passare anche provider/modello direttamente.
EOF
}

find_pi() {
  local active_node candidate
  active_node="$(command -v node 2>/dev/null || true)"
  if [[ -n "$active_node" ]]; then
    candidate="$(dirname "$active_node")/pi"
    if [[ -x "$candidate" && "$(readlink -f "$candidate")" != "$(readlink -f "$0")" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  while IFS= read -r candidate; do
    if [[ -x "$candidate" && "$(readlink -f "$candidate")" != "$(readlink -f "$0")" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(printf '%s\n' "$HOME/.nvm/versions/node"/*/bin/pi | sort -Vr)

  candidate="$HOME/.local/bin/pi"
  if [[ -x "$candidate" && "$(readlink -f "$candidate")" != "$(readlink -f "$0")" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  return 127
}

run_pi() {
  local pi_bin
  pi_bin="$(find_pi)" || {
    echo "Eseguibile Pi reale non trovato." >&2
    return 127
  }
  export PATH="$(dirname "$pi_bin"):$PATH"
  refresh_local_gateway "$(dirname "$pi_bin")/node"
  exec "$pi_bin" "$@"
}

refresh_local_gateway() {
  local gateway models_path node_bin
  node_bin="$1"
  models_path="$HOME/.pi/agent/models.json"
  gateway="$(ip -4 route show default | awk 'NR == 1 { print $3 }')"
  if [[ ! "$gateway" =~ ^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.) ]]; then
    echo "Gateway WSL2 privato non valido: $gateway" >&2
    return 1
  fi
  [[ -x "$node_bin" && -f "$models_path" ]] || return 0

  "$node_bin" - "$models_path" "$gateway" <<'NODE'
const fs = require("node:fs");
const [modelsPath, gateway] = process.argv.slice(2);
const document = JSON.parse(fs.readFileSync(modelsPath, "utf8"));
const providers = document.providers || {};
const decode = providers["local-qwen38-decode"];
let changed = false;
if (decode?.baseUrl) {
  const url = new URL(decode.baseUrl);
  if (url.hostname !== gateway) {
    url.hostname = gateway;
    decode.baseUrl = url.toString().replace(/\/$/, "");
    changed = true;
  }
}
const router = providers["local-qwen38-router"];
const previousHost = router?.baseUrl ? new URL(router.baseUrl).hostname : null;
for (const [id, provider] of Object.entries(providers)) {
  if (!id.startsWith("local-") || !provider?.baseUrl) continue;
  const url = new URL(provider.baseUrl);
  if (!previousHost || previousHost === gateway || url.hostname !== previousHost) continue;
  url.hostname = gateway;
  provider.baseUrl = url.toString().replace(/\/$/, "");
  changed = true;
}
if (!changed) process.exit(0);

const temporaryPath = `${modelsPath}.tmp-gateway-${process.pid}`;
fs.writeFileSync(temporaryPath, `${JSON.stringify(document, null, 2)}\n`, { mode: 0o600 });
fs.renameSync(temporaryPath, modelsPath);
NODE
}

invoked_as="$(basename "$0")"
profile="${COMMAND_PROFILES[$invoked_as]:-q4-general}"

# ~/.local/bin/pi may be a legacy launcher that forces the old Qwen bridge on
# port 8032. The profile commands must address their configured provider
# directly. The real Pi binary ignores this compatibility variable.
export PI_QWEN_BRIDGE_SKIP=1

case "${1:-}" in
  --pi-default)
    shift
    run_pi "$@"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  --profiles)
    usage
    exit 0
    ;;
  --list-models)
    run_pi --list-models qwen
    ;;
  '')
    ;;
  *)
    if [[ -n "${PROFILES[$1]:-}" ]]; then
      profile="$1"
      shift
    elif [[ "$1" == */* ]]; then
      model="$1"
      shift
      run_pi --model "$model" --models "$MODEL_SCOPE" "$@"
    fi
    ;;
esac

run_pi --model "${PROFILES[$profile]}" --models "$MODEL_SCOPE" "$@"


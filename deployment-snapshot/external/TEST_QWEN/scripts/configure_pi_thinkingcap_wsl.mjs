import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const argv = process.argv.slice(2);
const agent = path.join(os.homedir(), '.pi', 'agent');
const modelsPath = path.join(agent, 'models.json');
const settingsPath = path.join(agent, 'settings.json');
const bin = path.join(os.homedir(), '.local', 'bin');
const piPath = path.join(bin, 'pi');
const provider = 'local-thinkingcap-qwen38';
const visionProvider = 'local-thinkingcap-qwen38-vision';
const model = 'thinkingcap-qwen3.8-27b-q6k-mtp-120k';

function selected(args) {
  let explicitProvider, explicitModel;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--provider') explicitProvider = args[++i];
    else if (args[i].startsWith('--provider=')) explicitProvider = args[i].slice(11);
    else if (args[i] === '--model') explicitModel = args[++i];
    else if (args[i].startsWith('--model=')) explicitModel = args[i].slice(8);
  }
  const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf8'));
  const qualifiedProvider = explicitModel?.includes('/') ? explicitModel.split('/', 1)[0] : undefined;
  const chosenProvider = explicitProvider ?? qualifiedProvider ?? settings.defaultProvider;
  const chosenModel = explicitModel?.split(':', 1)[0].split('/').at(-1) ?? settings.defaultModel;
  return (chosenProvider === provider || chosenProvider === visionProvider) && chosenModel === model;
}

if (argv.includes('--is-selected')) process.exit(selected(argv.slice(argv.indexOf('--is-selected') + 1)) ? 0 : 1);

const models = JSON.parse(fs.readFileSync(modelsPath, 'utf8'));
const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf8'));
models.providers ??= {};
models.providers[provider] = {
  name: 'ThinkingCap Qwen3.8 27B Q6_K Vision + MTP3 120K (Windows / WSL2)',
  baseUrl: 'http://127.0.0.1:18044/v1', api: 'openai-completions',
  apiKey: "!tr -d '\\r\\n' < /mnt/c/llama_official/router/api-key.txt", authHeader: true,
  compat: { supportsStore:false, maxTokensField:'max_tokens', supportsDeveloperRole:false,
    supportsReasoningEffort:true, supportsUsageInStreaming:true,
    requiresReasoningContentOnAssistantMessages:true, thinkingFormat:'qwen-chat-template' },
  models: [{ id:model, name:'ThinkingCap Qwen3.8 27B Q6_K Vision MTP3 120K', reasoning:true, input:['text','image'],
    contextWindow:122880, maxTokens:122880,
    thinkingLevelMap:{ off:null, minimal:null, low:'low', medium:'medium', high:null, xhigh:'xhigh', max:null },
    samplingParams:{ temperature:1.0, top_p:0.95, top_k:20, min_p:0.0, presence_penalty:0.0, repetition_penalty:1.0 },
    cost:{ input:0, output:0, cacheRead:0, cacheWrite:0 } }]
};
models.providers[visionProvider] = {
  ...models.providers[provider],
  name: 'ThinkingCap Qwen3.8 27B Q6_K Vision + MTP3 96K (Windows / WSL2)',
  models: [{ ...models.providers[provider].models[0],
    name: 'ThinkingCap Qwen3.8 27B Q6_K Vision MTP3 96K',
    input: ['text', 'image'], contextWindow:98304, maxTokens:98304 }],
};

function atomicWrite(target, content, mode) {
  if (fs.existsSync(target) && fs.readFileSync(target, 'utf8') === content) return null;
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const backup = fs.existsSync(target) ? `${target}.bak-thinkingcap-${stamp}` : null;
  if (backup) fs.copyFileSync(target, backup, fs.constants.COPYFILE_EXCL);
  const temporary = `${target}.tmp-thinkingcap-${process.pid}`;
  fs.writeFileSync(temporary, content, { mode, flag:'wx' });
  fs.renameSync(temporary, target); fs.chmodSync(target, mode);
  return backup;
}

const apply = argv.includes('--apply');
const refresh = argv.includes('--refresh');
if (!apply && !refresh) {
  console.log(JSON.stringify({ applied:false, provider, model, baseUrl:models.providers[provider].baseUrl, defaultThinking:'xhigh' }, null, 2));
  process.exit(0);
}
fs.mkdirSync(agent, { recursive:true, mode:0o700 }); fs.mkdirSync(bin, { recursive:true, mode:0o700 });
const backups = [];
const b1 = atomicWrite(modelsPath, `${JSON.stringify(models, null, 2)}\n`, 0o600); if (b1) backups.push(b1);
if (apply) {
  settings.defaultProvider = provider; settings.defaultModel = model;
  const b2 = atomicWrite(settingsPath, `${JSON.stringify(settings, null, 2)}\n`, 0o600); if (b2) backups.push(b2);
}
if (apply || refresh) {
  const direct = `#!/usr/bin/env bash\nset -euo pipefail\nnode /mnt/c/MYPROJECT/TEST_QWEN/scripts/configure_pi_thinkingcap_wsl.mjs --refresh\nbash /mnt/c/MYPROJECT/TEST_QWEN/scripts/ensure_thinkingcap_wsl_bridge.sh\nexec "$HOME/.local/bin/pi-qwen" --pi-default --provider ${provider} --model ${model} --thinking xhigh "$@"\n`;
  const b3 = atomicWrite(path.join(bin, 'pi-thinkingcap'), direct, 0o755); if (b3) backups.push(b3);
  const visionDirect = `#!/usr/bin/env bash\nset -euo pipefail\nnode /mnt/c/MYPROJECT/TEST_QWEN/scripts/configure_pi_thinkingcap_wsl.mjs --refresh\nbash /mnt/c/MYPROJECT/TEST_QWEN/scripts/ensure_thinkingcap_wsl_bridge.sh\nexec "$HOME/.local/bin/pi-qwen" --pi-default --provider ${visionProvider} --model ${model} --thinking xhigh "$@"\n`;
  const bVision = atomicWrite(path.join(bin, 'pi-thinkingcap-vision'), visionDirect, 0o755); if (bVision) backups.push(bVision);
  let pi = fs.readFileSync(piPath, 'utf8').replaceAll('\r\n','\n');
  const begin = '# BEGIN THINKINGCAP DIRECT PROVIDER\n', end = '# END THINKINGCAP DIRECT PROVIDER\n';
  const block = `${begin}if node /mnt/c/MYPROJECT/TEST_QWEN/scripts/configure_pi_thinkingcap_wsl.mjs --is-selected "$@"; then\n  node /mnt/c/MYPROJECT/TEST_QWEN/scripts/configure_pi_thinkingcap_wsl.mjs --refresh\n  bash /mnt/c/MYPROJECT/TEST_QWEN/scripts/ensure_thinkingcap_wsl_bridge.sh\n  exec "$HOME/.local/bin/pi-qwen" --pi-default --thinking xhigh "$@"\nfi\n${end}`;
  if (pi.includes('# PI INDEPENDENT PROVIDERS V1\n')) {
    // This wrapper owns nonblocking multi-provider dispatch. A direct
    // ThinkingCap refresh must not restore the old mandatory bridge branch.
  } else if (pi.includes(begin)) {
    const start = pi.indexOf(begin), finish = pi.indexOf(end, start);
    if (finish < 0) throw new Error('Incomplete ThinkingCap block in Pi wrapper');
    pi = pi.slice(0, start) + block + pi.slice(finish + end.length);
  } else {
    const marker = 'set -euo pipefail\n';
    if (!pi.startsWith('#!/usr/bin/env bash\n') || !pi.includes(marker)) throw new Error('Unexpected Pi wrapper');
    pi = pi.replace(marker, marker + block);
  }
  const b4 = atomicWrite(piPath, pi, 0o755); if (b4) backups.push(b4);
  execFileSync('bash', ['-n', piPath]); execFileSync('bash', ['-n', path.join(bin, 'pi-thinkingcap')]);
  execFileSync('bash', ['-n', path.join(bin, 'pi-thinkingcap-vision')]);
}
console.log(JSON.stringify({ applied:apply, refreshed:refresh, provider, visionProvider, model,
  baseUrl:models.providers[provider].baseUrl, defaultProvider:apply ? settings.defaultProvider : undefined,
  defaultModel:apply ? settings.defaultModel : undefined, defaultThinking:'xhigh', backups }, null, 2));

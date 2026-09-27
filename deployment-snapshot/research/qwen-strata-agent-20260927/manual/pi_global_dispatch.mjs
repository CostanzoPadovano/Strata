// Optional discovery only. No model launch, provider rewrite or inference.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import {mayUseStrata, validHealth, healthAt, MODEL, PROVIDER} from './pi_route.mjs';
const maintenance = new Set(['--help','-h','--version','-v','--list-models','--export',
  'install','remove','uninstall','update','list','config','auth']);

export function selection(args, settings = {}) {
  const normalized = [...args];
  let explicit = false, session = false, disabled = false;
  const end = args.indexOf('--') < 0 ? args.length : args.indexOf('--');
  for (let i = 0; i < end; i++) {
    const arg = args[i];
    if (maintenance.has(arg) && (i === 0 || arg.startsWith('-'))) disabled = true;
    if (['--no-extensions','-ne'].includes(arg)) disabled = true;
    if (['--continue','-c','--resume','-r','--session','--models'].includes(arg)
        || arg.startsWith('--session=') || arg.startsWith('--models=')) session = true;
    if (['--model','--provider'].includes(arg)) {
      explicit = true;
      if (arg === '--model' && args[i+1]) normalized[i+1] = args[i+1].split(':')[0];
      i++;
    } else if (arg.startsWith('--model=')) {
      explicit = true;
      normalized[i] = arg.split(':')[0];
    } else if (arg.startsWith('--provider=')) explicit = true;
  }
  const eligible = mayUseStrata(normalized);
  return {probe: !disabled, selectStrata: !disabled && eligible && (!session || explicit),
    legacyFlash: !disabled && eligible && (explicit
      || (!session && settings.defaultProvider === PROVIDER && settings.defaultModel === MODEL))};
}

export async function discover(args, dependencies) {
  const decision = selection(args);
  if (!decision.probe) return null;
  const host = dependencies.gateway();
  if (!/^172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}$/.test(host ?? '')) throw Error('gateway');
  const key = dependencies.key();
  if (!key) throw Error('authentication');
  const health = await dependencies.health(host, key);
  if (!validHealth(health)) throw Error('identity');
  return {baseUrl: `http://${host}:8038/v1`, images: health.images, selectStrata: decision.selectStrata};
}

async function main(args) {
  const command = args.shift();
  if (command === 'legacy-selected') {
    const settings = JSON.parse(fs.readFileSync(path.join(os.homedir(), '.pi/agent/settings.json'), 'utf8'));
    return selection(args, settings).legacyFlash ? 0 : 1;
  }
  if (command !== 'route') throw Error('command');
  const route = await discover(args, {
    gateway: () => execFileSync('ip', ['-4','route','show','default'], {encoding:'utf8',timeout:2000})
      .match(/\bvia\s+(\S+)/)?.[1],
    key: () => fs.readFileSync('/mnt/c/llama_official/router/api-key.txt','utf8').trim(),
    health: healthAt,
  });
  if (!route) return 1;
  process.stdout.write(`${route.baseUrl}\n${route.images ? '1':'0'}\n${route.selectStrata ? '1':'0'}\n`);
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try { process.exitCode = await main(process.argv.slice(2)); }
  catch (error) {
    // Never print health bodies, keys or arbitrary remote messages.
    if (!['ECONNREFUSED','EHOSTUNREACH','ENETUNREACH'].includes(error.code))
      console.error('Avviso: Strata non disponibile o non riconosciuto; Pi resta utilizzabile con gli altri modelli.');
    process.exitCode = 1;
  }
}

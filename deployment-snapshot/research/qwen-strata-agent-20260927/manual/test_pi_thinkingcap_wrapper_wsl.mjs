// Exercise the real configurator in an isolated fake home; no server/API calls.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'pi-thinkingcap-independent-'));
try {
  const agent=path.join(temporary,'.pi/agent'),bin=path.join(temporary,'.local/bin');
  fs.mkdirSync(agent,{recursive:true,mode:0o700});fs.mkdirSync(bin,{recursive:true,mode:0o700});
  const modelPath=path.join(agent,'models.json'),settingsPath=path.join(agent,'settings.json');
  const settings='{"defaultProvider":"other","defaultModel":"other-model"}\n';
  const other={baseUrl:'http://127.0.0.1:9999/v1',models:[{id:'other-model'}]};
  fs.writeFileSync(modelPath,JSON.stringify({providers:{other}}),{mode:0o600});
  fs.writeFileSync(settingsPath,settings,{mode:0o600});
  const wrapper=fs.readFileSync(new URL('./pi-global-installed.sh',import.meta.url),'utf8');
  fs.writeFileSync(path.join(bin,'pi'),wrapper,{mode:0o700});
  const source=fs.readFileSync('/mnt/c/MYPROJECT/TEST_QWEN/scripts/configure_pi_thinkingcap_wsl.mjs','utf8')
    .replaceAll('os.homedir()',JSON.stringify(temporary));
  const argv=process.argv,log=console.log;
  try {
    process.argv=[process.execPath,'isolated-thinkingcap-config','--refresh'];console.log=()=>{};
    await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
  } finally {process.argv=argv;console.log=log;}
  assert.equal(fs.readFileSync(path.join(bin,'pi'),'utf8'),wrapper);
  assert.equal(fs.readFileSync(settingsPath,'utf8'),settings);
  assert.deepEqual(JSON.parse(fs.readFileSync(modelPath,'utf8')).providers.other,other);
  console.log('ThinkingCap refresh preserves independent global wrapper, defaults and other provider (isolated home).');
} finally {
  // Exact unique directory created above, never a broad/global user path.
  assert.equal(path.dirname(path.resolve(temporary)),path.resolve(os.tmpdir()));
  assert.ok(path.basename(temporary).startsWith('pi-thinkingcap-independent-'));
  fs.rmSync(temporary,{recursive:true});
}

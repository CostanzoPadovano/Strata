import assert from 'node:assert/strict';
import {selection,discover} from './pi_global_dispatch.mjs';
import {VISION_ENGINE_SHA,MODEL} from './pi_route.mjs';
const settings={defaultProvider:'local-thinkingcap-qwen38',defaultModel:'other'};
const flash={defaultProvider:'local-qwen38',defaultModel:MODEL};
assert.deepEqual(selection([],settings),{probe:true,selectStrata:true,legacyFlash:false});
assert.equal(selection([],flash).legacyFlash,true);
for(const args of [['--model','other/model'],['--provider','openai'],['--model=other/model']]) {
  assert.equal(selection(args,flash).selectStrata,false);
  assert.equal(selection(args,flash).legacyFlash,false);
  assert.equal(selection(args,flash).probe,true);
}
for(const args of [['--help'],['--mode','rpc','--version'],['--no-extensions'],['-ne'],['install','pkg']])
  assert.equal(selection(args).probe,false);
for(const args of [['--continue'],['--resume'],['--session','a.jsonl'],['--models','other/*']])
  assert.equal(selection(args).selectStrata,false);
for(const arg of [MODEL,`local-qwen38/${MODEL}`,`local-qwen38/${MODEL}:xhigh`]) {
  assert.equal(selection(['--resume','--model',arg]).selectStrata,true);
  assert.equal(selection(['--model',arg]).legacyFlash,true);
}
assert.equal(selection(['--','--model','other/model']).selectStrata,true);
const health={status:'ok',runtime:'ista-strata-vision-v1',engine_sha256:VISION_ENGINE_SHA,
  images:true,vision_cpu:true,max_image_tokens:1024,max_images:8,encoder_residency:'on-demand',
  encoder_sha256:'98ffde98f5388b518d8581b2983973d785f20f00a21e5e82393b38f4188e094c',
  projector_sha256:'b2e9b5e4a44c107f8867e67dbf09b607fd99ae33c1a97a60a6720aeb252a9dad',
  model:MODEL,max_context:98304,max_output_tokens:98296,output_policy:'remaining-context-v1',
  output_context_reserve_tokens:8,experimental:true};
const deps={gateway:()=> '172.19.48.1',key:()=> 'test-only',health:async()=>health};
assert.equal((await discover([],deps)).selectStrata,true);
assert.equal((await discover(['--model','other/model'],deps)).selectStrata,false);
assert.equal((await discover(['--model','other/model'],deps)).images,true);
await assert.rejects(discover([],{...deps,health:async()=>({...health,max_output_tokens:1024})}),/identity/);
await assert.rejects(discover([],{...deps,gateway:()=> '8.8.8.8'}),/gateway/);
await assert.rejects(discover([],{...deps,key:()=>''}),/authentication/);
await assert.rejects(discover([],{...deps,health:async()=>{throw Error('timeout');}}),/timeout/);
assert.equal(await discover(['--help'],{gateway:()=>{throw Error('Should not probe');}}),null);
console.log('Independent discovery: valid-only overlay; other providers/session/CLI preserved; timeout/auth/identity rejected. No model calls.');

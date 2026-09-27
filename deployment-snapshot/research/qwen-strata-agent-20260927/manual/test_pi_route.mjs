import assert from 'node:assert/strict';
import {mayUseStrata,validHealth,MODEL,ENGINE_SHA,VISION_ENGINE_SHA} from './pi_route.mjs';
for(const args of [[],['prompt'],['--model',MODEL],['--model',`local-qwen38/${MODEL}`],['--provider','local-qwen38'],['--thinking','low']])assert.equal(mayUseStrata(args),true);
for(const args of [['--help'],['--list-models'],['--provider','local-thinkingcap-qwen38'],['--model','other'],['--model=other'],['--provider=other']])assert.equal(mayUseStrata(args),false);
for(const args of [['--provider'],['--model'],['--model','--help'],['--provider='],['--model='],['--thinking','low','--help']])assert.equal(mayUseStrata(args),false);
assert.equal(mayUseStrata(['--','--help']),true);
const h={status:'ok',runtime:'ista-strata-manual-v1',model:MODEL,max_context:98304,max_output_tokens:98296,
output_policy:'remaining-context-v1',output_context_reserve_tokens:8,engine_sha256:ENGINE_SHA,images:false,experimental:true};
assert.equal(validHealth(h),true);
for(const field of Object.keys(h)){const copy={...h};delete copy[field];assert.equal(validHealth(copy),false,field);}
assert.equal(validHealth({...h,images:true}),false);
assert.equal(validHealth({...h,max_context:131072}),false);
assert.equal(validHealth({...h,max_output_tokens:1024}),false);
assert.equal(validHealth({...h,output_context_reserve_tokens:0}),false);
assert.equal(validHealth({...h,output_policy:'unbounded'}),false);
const v={...h,runtime:'ista-strata-vision-v1',engine_sha256:VISION_ENGINE_SHA,images:true,vision_cpu:true,max_image_tokens:1024,max_images:8,encoder_residency:'on-demand',
  encoder_sha256:'98ffde98f5388b518d8581b2983973d785f20f00a21e5e82393b38f4188e094c',
  projector_sha256:'b2e9b5e4a44c107f8867e67dbf09b607fd99ae33c1a97a60a6720aeb252a9dad'};
assert.equal(validHealth(v),true);
for(const field of Object.keys(v)){const copy={...v};delete copy[field];assert.equal(validHealth(copy),false,field);}
for(const change of [{vision_cpu:false},{max_image_tokens:4096},{engine_sha256:ENGINE_SHA},{projector_sha256:'wrong'}])assert.equal(validHealth({...v,...change}),false);
console.log('Pi routing: admitted/bypass/malformed/help/identity tests passed');

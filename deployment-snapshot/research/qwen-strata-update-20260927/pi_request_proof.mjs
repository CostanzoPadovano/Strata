// Explicit qualification observer only; never loaded by the normal Pi wrapper.
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {healthAt, VISION_ENGINE_SHA, MODEL, PROVIDER} from './manual/pi_route.mjs';
export default function(pi){
  const path=process.env.PI_UPDATE_PROOF_PATH;
  const expectedBase=process.env.PI_UPDATE_PROOF_BASE;
  const nonce=process.env.PI_UPDATE_TEST_NONCE;
  if(!path?.startsWith('/mnt/c/Users/costa/Documents/Project_ANTIREZ/research/qwen-strata-update-20260927/runs/')
     || !/^http:\/\/172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}:8038\/v1$/.test(expectedBase??'')
     || nonce!=='CACHE_UPDATE_20260927')throw Error('Invalid explicit Pi proof boundary');
  fs.closeSync(fs.openSync(path,'ax'));
  const key=fs.readFileSync('/mnt/c/llama_official/router/api-key.txt','utf8').trim();
  let count=0;
  pi.on('before_provider_request',async(event,ctx)=>{
    const model=ctx.model;
    const payload=JSON.stringify(event.payload);
    const health=await healthAt(new URL(expectedBase).hostname,key);
    const row={sequence:++count,time_ms:Date.now(),base_url:model?.baseUrl,
      provider:model?.provider,model:model?.id,health_engine_sha256:health.engine_sha256,
      nonce_in_payload:payload.includes(nonce),payload_sha256:createHash('sha256').update(payload).digest('hex')};
    if(count>64)throw Error('Finite Pi proof request limit');
    row.passed=model?.baseUrl===expectedBase && model?.provider===PROVIDER && model?.id===MODEL
      && health.engine_sha256===VISION_ENGINE_SHA && row.nonce_in_payload;
    fs.appendFileSync(path,JSON.stringify(row)+'\n');
    if(!row.passed)throw Error('Pi test request is not addressed to the exact candidate');
    // No payload, headers, sampling or provider changes.
  });
}

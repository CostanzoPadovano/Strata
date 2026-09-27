// Real installed Pi SDK + fake in-memory HTTP SSE; no model/listener/network.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const install='/home/costapad/.nvm/versions/node/v22.23.0/lib/node_modules/@earendil-works/pi-coding-agent';
const compat=pathToFileURL(`${install}/node_modules/@earendil-works/pi-ai/dist/compat.js`).href;
const source=fs.readFileSync(new URL('./pi_strata.mjs',import.meta.url),'utf8').replace("'@earendil-works/pi-ai/compat'",JSON.stringify(compat));
const {boundedStream}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const {SettingsManager}=await import(pathToFileURL(`${install}/dist/core/settings-manager.js`));
const model={provider:'local-qwen38',id:'qwen3.8-flash-next-local',api:'openai-completions',
  baseUrl:'http://127.0.0.1:1/v1',contextWindow:98304,maxTokens:98296,reasoning:true,input:['text','image'],
  thinkingLevelMap:{xhigh:'xhigh'},cost:{input:0,output:0,cacheRead:0,cacheWrite:0},
  compat:{supportsStore:false,supportsDeveloperRole:false,supportsReasoningEffort:true,supportsUsageInStreaming:true,
    maxTokensField:'max_tokens',requiresReasoningContentOnAssistantMessages:true,thinkingFormat:'reasoning_effort'}};
const context={messages:[{role:'user',content:'Reply OK',timestamp:Date.now()}]};
const sse='data: '+JSON.stringify({id:'offline',object:'chat.completion.chunk',created:0,model:model.id,
  choices:[{index:0,delta:{role:'assistant',content:'OK'},finish_reason:null}]})+'\n\n' +
  'data: '+JSON.stringify({id:'offline',object:'chat.completion.chunk',created:0,model:model.id,
    choices:[{index:0,delta:{},finish_reason:'stop'}],usage:{prompt_tokens:31,completion_tokens:1,total_tokens:32}})+'\n\n' + 'data: [DONE]\n\n';
const stream=boundedStream();
for(const [extra,expected,reason] of [[{},98296,'xhigh'],[{maxTokens:16384},16384,'xhigh'],
  [{maxTokens:32},32,'xhigh'],[{cacheRetention:'none',maxTokens:17843},17843,'minimal']]){
  let payload;
  const result=await stream(model,context,{reasoning:'xhigh',apiKey:'offline-test-only',...extra,
    fetch:async(_url,init)=>{payload=JSON.parse(init.body);return new Response(sse,{status:200,headers:{'content-type':'text/event-stream'}});}}).result();
  assert.equal(payload.max_tokens,expected);
  assert.equal(payload.stream,true);
  assert.equal(payload.reasoning_effort,reason);
  assert.equal(result.stopReason,'stop');
  assert.equal(result.content.find(c=>c.type==='text')?.text,'OK');
}
const paths=['/home/costapad/.pi/agent/models.json','/home/costapad/.pi/agent/settings.json'];
const bytes=paths.map(p=>fs.readFileSync(p));
const settings=SettingsManager.create('/mnt/c/Users/costa/Documents/Project_ANTIREZ','/home/costapad/.pi/agent');
assert.equal(settings.getCompactionSettings(model).reserveTokens,22304);
paths.forEach((p,i)=>assert.equal(fs.readFileSync(p).equals(bytes[i]),true));
console.log('Actual Pi0.87.1 SDK: default98296/explicit16384+32/minimal-summary payloads and unchanged22304 compaction reserve passed; fake HTTP only.');

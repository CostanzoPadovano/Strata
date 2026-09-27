import assert from 'node:assert/strict';
import fs from 'node:fs';
const source=fs.readFileSync(new URL('./pi_strata.mjs',import.meta.url),'utf8').replace(
  "import {openAICompletionsApi} from '@earendil-works/pi-ai/compat';",
  'const openAICompletionsApi=()=>({stream:()=>{throw Error("No upstream in unit test")}});');
const {boundedOptions,boundedStream}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
assert.equal(boundedOptions({}).maxTokens,98296);
assert.equal(boundedOptions({maxTokens:16384}).maxTokens,16384);
assert.equal(boundedOptions({maxTokens:2**30}).maxTokens,98296);
assert.equal(boundedOptions({maxTokens:32}).maxTokens,32);
for(const n of [0,-1,1.5,'1024',NaN])assert.throws(()=>boundedOptions({maxTokens:n}));
assert.equal(boundedOptions({reasoning:'xhigh',onPayload:()=>{}}).reasoningEffort,'xhigh');
assert.equal(boundedOptions({reasoning:'xhigh',cacheRetention:'none'}).reasoningEffort,'minimal');
assert.equal(boundedOptions({reasoning:'xhigh',cacheRetention:'none',onPayload:()=>{}}).reasoningEffort,'xhigh');
assert.equal(boundedOptions({reasoning:'off',onPayload:()=>{}}).reasoningEffort,'minimal');
const model={provider:'local-qwen38',id:'qwen3.8-flash-next-local',api:'openai-completions',contextWindow:98304,maxTokens:98296};
let observed;
const stream=boundedStream((m,c,o)=>{observed={m,c,o};return 'stream';});
assert.equal(stream(model,{}, {reasoning:'xhigh',maxTokens:16384}),'stream');
assert.equal(observed.o.maxTokens,16384);
assert.equal(stream(model,{},{}),'stream');
assert.equal(observed.o.maxTokens,98296);
for(const field of Object.keys(model)){const bad={...model};delete bad[field];assert.throws(()=>stream(bad,{},{}));}
console.log('Pi Strata budget/shim: dynamic-context ceiling/explicit caps/thinking/summary/identity passed (mock stream; no model load)');

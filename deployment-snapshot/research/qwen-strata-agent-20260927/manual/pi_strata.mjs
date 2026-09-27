import fs from 'node:fs';
import {openAICompletionsApi} from '@earendil-works/pi-ai/compat';
const PROVIDER='local-qwen38';
const MODEL='qwen3.8-flash-next-local';
const MAX_OUTPUT=98304-8;
const COMPAT={supportsStore:false,supportsDeveloperRole:false,supportsReasoningEffort:true,
  supportsUsageInStreaming:true,maxTokensField:'max_tokens',
  requiresReasoningContentOnAssistantMessages:true,thinkingFormat:'reasoning_effort'};
export function boundedOptions(options={}){
  const {reasoning,thinkingBudgets,...rest}=options;
  const max=options.maxTokens??MAX_OUTPUT;
  if(!Number.isSafeInteger(max)||max<=0)throw Error('Invalid Strata output budget');
  const summary=options.cacheRetention==='none' && typeof options.onPayload!=='function';
  return {...rest,maxTokens:Math.min(max,MAX_OUTPUT),
    ...(summary || reasoning==='off'?{reasoningEffort:'minimal'}:
      reasoning?{reasoningEffort:reasoning}:{}),
    ...(thinkingBudgets?{thinkingBudgets}:{})};
}
export function boundedStream(upstream=openAICompletionsApi().stream){
  return (model,context,options)=>{
    if(model.provider!==PROVIDER||model.id!==MODEL||model.api!=='openai-completions'
      ||model.contextWindow!==98304||model.maxTokens!==MAX_OUTPUT)throw Error('Unexpected Strata provider/model/limits');
    // Non-simple upstream avoids Pi's fixed4096 estimate/clamp. The server
    // fits the budget to the EXACT rendered/image-expanded prompt; input is
    // never trimmed, and real length/context exhaustion remains visible.
    return upstream(model,context,boundedOptions(options));
  };
}
export default function(pi){
  const baseUrl=process.env.ISTA_STRATA_BASE_URL;
  if(!baseUrl || !/^http:\/\/172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}:8038\/v1$/.test(baseUrl))throw Error('Strata extension requires authenticated wrapper admission');
  const key=fs.readFileSync('/mnt/c/llama_official/router/api-key.txt','utf8').trim();
  if(!key)throw Error('Missing Strata authentication');
  const vision=process.env.ISTA_STRATA_VISION==='1';
  pi.registerProvider(PROVIDER,{
    baseUrl,api:'openai-completions',apiKey:key,
    models:[{id:MODEL,name:`ISTA / Strata 98K (sperimentale, ${vision?'vision CPU':'testo'})`,reasoning:true,
      thinkingLevelMap:{xhigh:'xhigh'},input:vision?['text','image']:['text'],contextWindow:98304,maxTokens:MAX_OUTPUT,
      compat:COMPAT,
      cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}],
    streamSimple:boundedStream(),
  });
  pi.on('session_start',(_event,ctx)=>{
    ctx.ui.notify(`ISTA / Strata: stesso profilo, 98K, xhigh predefinito; ${vision?'vision CPU':'testo'}, output fino al contesto residuo (thinking incluso). Cache numerica sperimentale.`,'warning');
  });
}

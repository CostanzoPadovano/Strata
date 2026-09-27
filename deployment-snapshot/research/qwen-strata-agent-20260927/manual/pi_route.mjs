import fs from 'node:fs';
import http from 'node:http';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
export const MODEL='qwen3.8-flash-next-local';
export const PROVIDER='local-qwen38';
export const ENGINE_SHA='59ef6d14bd5c1c032b84e2ca08ec11aab11bd8c24cca51a156eb45ee0a20fea6';
// Independent native vision build: parser/position/ownership gates are separate.
export const VISION_ENGINE_SHA='f1072687e794aa5958f121c8967983947b5ea1322fa336c60688cfdc7736742f';

export function mayUseStrata(args){
  let provider,model;
  const maintenance=new Set(['--help','-h','--version','-v','--list-models','--export','install','remove','uninstall','update','list','config','auth']);
  if(maintenance.has(args[0]))return false;
  for(let i=0;i<args.length;i++){
    if(args[i]==='--')break;
    if(['--help','-h','--version','-v','--list-models','--export'].includes(args[i]))return false;
    if(args[i]==='--provider'){
      if(!args[i+1]||args[i+1].startsWith('-'))return false;
      provider=args[++i];
    }
    else if(args[i].startsWith('--provider='))provider=args[i].slice(11);
    else if(args[i]==='--model'){
      if(!args[i+1]||args[i+1].startsWith('-'))return false;
      model=args[++i];
    }
    else if(args[i].startsWith('--model='))model=args[i].slice(8);
  }
  if(provider===''||model==='')return false;
  if(provider && provider!==PROVIDER)return false;
  if(model && model!==MODEL && model!==`${PROVIDER}/${MODEL}`)return false;
  return true;
}

export function validHealth(health){
  const text=health?.runtime==='ista-strata-manual-v1' && health.engine_sha256===ENGINE_SHA && health.images===false;
  const vision=health?.runtime==='ista-strata-vision-v1' && health.engine_sha256===VISION_ENGINE_SHA && health.images===true
    && health.vision_cpu===true && health.max_image_tokens===1024 && health.max_images===8 && health.encoder_residency==='on-demand'
    && health.encoder_sha256==='98ffde98f5388b518d8581b2983973d785f20f00a21e5e82393b38f4188e094c'
    && health.projector_sha256==='b2e9b5e4a44c107f8867e67dbf09b607fd99ae33c1a97a60a6720aeb252a9dad';
  return health?.status==='ok' && (text||vision)
    && health.model===MODEL && health.max_context===98304 && health.max_output_tokens===98304-8
    && health.output_policy==='remaining-context-v1' && health.output_context_reserve_tokens===8
    && health.experimental===true;
}

export function healthAt(host,key){
  return new Promise((resolve,reject)=>{
    const req=http.get({host,port:8038,path:'/health',headers:{authorization:`Bearer ${key}`},timeout:2500},res=>{
      let body='';
      res.setEncoding('utf8');
      res.on('data',chunk=>{body+=chunk;if(body.length>8192)req.destroy(Error('Health response exceeds8KiB'));});
      res.on('end',()=>{try{if(res.statusCode!==200)throw Error(`Health HTTP${res.statusCode}`);resolve(JSON.parse(body));}catch(error){reject(error);}});
    });
    req.on('timeout',()=>req.destroy(Error('Strata health timeout')));
    req.on('error',reject);
  });
}

async function main(){
  if(!mayUseStrata(process.argv.slice(2)))return 1;
  const host=execFileSync('ip',['-4','route','show','default'],{encoding:'utf8',timeout:3000}).match(/\bvia\s+(\S+)/)?.[1];
  if(!/^172\.(1[6-9]|2\d|3[01])\./.test(host??''))throw Error('No admitted WSL NAT gateway');
  const key=fs.readFileSync('/mnt/c/llama_official/router/api-key.txt','utf8').trim();
  if(!key)throw Error('Missing authentication');
  let health;
  try{health=await healthAt(host,key);}catch(error){
    if(['ECONNREFUSED','EHOSTUNREACH','ENETUNREACH'].includes(error.code))return 1;
    throw error; // Wrong auth/unresponsive service must never be silently selected.
  }
  if(!validHealth(health))throw Error('Port8038 is not the admitted manual Strata service');
  process.stdout.write(`http://${host}:8038/v1\n${health.images?'1':'0'}\n`);
  return 0;
}

if(process.argv[1] && import.meta.url===pathToFileURL(process.argv[1]).href){
  try{process.exitCode=await main();}catch(error){console.error(`Strata non selezionato: ${error.message}`);process.exitCode=2;}
}

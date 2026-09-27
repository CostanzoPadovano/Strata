import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import {execFileSync} from 'node:child_process';
import {timingSafeEqual} from 'node:crypto';
const gateway = execFileSync('wsl.exe', ['-d', 'Ubuntu-24.04', '-u', 'costapad', '--', 'ip', '-4', 'route', 'show', 'default'], {encoding:'utf8',timeout:15000,windowsHide:true}).match(/\bvia\s+(\d+\.\d+\.\d+\.\d+)\b/)?.[1];
if (!gateway || !/^172\.(1[6-9]|2\d|3[01])\./.test(gateway)) throw Error('WSL NAT private172.16/12 gateway missing; no LAN/all-address fallback');
if (!Object.entries(os.networkInterfaces()).some(([name,addresses]) => /WSL/i.test(name) && addresses?.some(a => a.family === 'IPv4' && a.address === gateway))) throw Error('Gateway does not belong to a Windows WSL virtual adapter');
const key = Buffer.from(fs.readFileSync('C:/llama_official/router/api-key.txt','utf8').trim());
if (!key.length) throw Error('Missing local authentication');
let active=0;
const proxy=http.createServer((req,res)=>{
  const supplied=Buffer.from((req.headers.authorization??'').replace(/^Bearer /i,''));
  if(supplied.length!==key.length||!timingSafeEqual(supplied,key)){res.writeHead(401);return res.end();}
  if(!['/health','/status','/v1/models','/v1/chat/completions'].includes(req.url) || !['GET','POST'].includes(req.method)){res.writeHead(404);return res.end();}
  const size=Number(req.headers['content-length']??0);
  if(!Number.isSafeInteger(size)||size<0||size>16*1024*1024||req.headers['transfer-encoding']){res.writeHead(413);return res.end();}
  if(active>=4){res.writeHead(503);return res.end();}
  active++;
  let released=false;
  const release=()=>{if(!released){released=true;active--;}};
  const upstream=http.request({host:'127.0.0.1',port:8037,path:req.url,method:req.method,
    headers:{...req.headers,host:'127.0.0.1:8037'},timeout:1200000},response=>{
    res.writeHead(response.statusCode,response.headers);response.pipe(res);
  });
  upstream.on('error',()=>{if(!res.headersSent)res.writeHead(502);res.end();release();});
  upstream.on('timeout',()=>upstream.destroy(Error('Request timeout')));
  res.on('close',()=>{release();upstream.destroy();});
  req.on('aborted',()=>upstream.destroy());
  req.pipe(upstream);
});
proxy.headersTimeout=10000;proxy.requestTimeout=60000;proxy.maxConnections=8;
proxy.listen(8038,gateway,()=>console.log(`Authenticated private WSL ${gateway}:8038 -> loopback8037`));

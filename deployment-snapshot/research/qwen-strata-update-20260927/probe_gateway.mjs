// Read-only explicit qualification probe; never prints authentication material.
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
import {healthAt} from './manual/pi_route.mjs';
const host=execFileSync('ip',['-4','route','show','default'],{encoding:'utf8',timeout:3000}).match(/\bvia\s+(\S+)/)?.[1];
if(!/^172\.(1[6-9]|2\d|3[01])\./.test(host??''))throw Error('Invalid WSL gateway');
const key=fs.readFileSync('/mnt/c/llama_official/router/api-key.txt','utf8').trim();
const health=await healthAt(host,key);
console.log(JSON.stringify({base_url:`http://${host}:8038/v1`,health}));

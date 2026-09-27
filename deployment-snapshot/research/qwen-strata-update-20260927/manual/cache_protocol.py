"""Numeric-only strict parser for the bound candidate's native DONE cache protocol."""
import math

def parse_cache_done(line):
    f=line.split()
    if len(f)!=9 or f[0]!='DONE': raise ValueError('Cache DONE protocol shape mismatch')
    generated,prompt=int(f[1]),int(f[2])
    prompt_ms,decode_ms=float(f[3]),float(f[4])
    accepted,offered,reused=map(int,f[6:9])
    if (generated<0 or prompt<1 or not 0<=reused<prompt or not 0<=accepted<=offered
            or not all(math.isfinite(x) and x>=0 for x in (prompt_ms,decode_ms))
            or f[5] not in ('stop','length','cancel')):
        raise ValueError('Invalid native cache/timing counters')
    return {'generated':generated,'prompt_tokens':prompt,'prompt_ms':prompt_ms,'decode_ms':decode_ms,
            'finish':f[5],'draft_accepted':accepted,'draft_offered':offered,'prompt_reused':reused,
            'fresh_prompt_tokens':prompt-reused,'prefill_processed_tokens':prompt-reused-1}

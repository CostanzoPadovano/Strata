"""Reuse the measured bounded observer with this campaign's isolated source."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'qwen-strata-stock-20260926'))
import native_probe as observer
from advanced_guard import validate_hybrid_cache_log


def main():
    observer.ROOT = ROOT
    observer.main()
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--mode',choices=['load','smoke','bench'],required=True)
    args=parser.parse_args()
    run=args.run.resolve()
    if not run.is_relative_to(ROOT/'runs'):raise ValueError('Probe evidence outside campaign')
    config=json.loads((run/'config.json').read_text())
    if args.mode=='load' or '--remote-expert-hybrid' not in config['args']:return
    result=json.loads((run/'probe.json').read_text())
    try:
        if not result.get('passed') or result.get('native_exit_code')!=0:
            raise RuntimeError('Hybrid cache validation needs a complete clean native trial')
        result['hybrid_cache_transfers']=validate_hybrid_cache_log(
            (run/'engine.log').read_text(),len(result['rows']),
            '--remote-expert-freeze-cache' in config['args'])
    except BaseException as error:
        result['passed']=False
        result['failure']=str(error)
        raise
    finally:
        (run/'probe.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__ == '__main__':
    main()

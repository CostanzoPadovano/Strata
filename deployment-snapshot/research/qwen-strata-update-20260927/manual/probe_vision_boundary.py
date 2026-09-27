"""Passive artifact/admission boundary probe. Never starts an encoder or LLM."""
import json
import os
from pathlib import Path
import sys
import time

from vision_campaign import ROOT, admitted
from vision_guard import source_checks

tag = sys.argv[1]
if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in tag):
    raise ValueError('Invalid diagnostic tag')
run = ROOT / 'runs' / tag
run.mkdir(exist_ok=False)


def snap(phase):
    row = {'phase': phase, 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'pid': os.getpid(), 'scope': 'No encoder/model; only source identity checks',
           **admitted.shared.counters()}
    with (run / 'boundary.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


snap('before-artifact-checks')
source_checks()
snap('after-artifact-checks')
report, _ = admitted.artifact_checks(admitted.PROFILE)
snap('after-resource-report')

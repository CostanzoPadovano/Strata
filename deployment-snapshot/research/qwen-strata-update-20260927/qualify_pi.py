"""Explicit bounded Pi tool/vision qualification; never part of normal startup."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent
WSL_ROOT = '/mnt/c/Users/costa/Documents/Project_ANTIREZ/research/qwen-strata-update-20260927'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--wrapper', choices=('staged', 'global'), required=True)
    parser.add_argument('--kind', choices=('text', 'vision'), required=True)
    parser.add_argument('--tag', default='')
    opts = parser.parse_args()
    run = opts.run.resolve()
    if not run.is_relative_to(ROOT / 'runs') or not (run / 'ready-accepted.json').exists():
        raise ValueError('Separately admitted candidate server required')
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in opts.tag):
        raise ValueError('Invalid immutable evidence suffix')
    prefix = f'pi-{opts.wrapper}-{opts.kind}' + ('-' + opts.tag if opts.tag else '')
    if (run / (prefix + '-result.json')).exists():
        raise FileExistsError('Immutable Pi evidence already exists')
    key = Path(r'C:\llama_official\router\api-key.txt').read_text().strip()
    request = urllib.request.Request('http://127.0.0.1:8037/health',
                                     headers={'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(request, timeout=5) as response:
        health = json.loads(response.read(8192))
    engine_sha = hashlib.sha256((ROOT / 'build/strata.exe').read_bytes()).hexdigest()
    if health.get('engine_sha256') != engine_sha or health.get('max_context') != 98304:
        raise ValueError('Unexpected live engine identity')
    cache = health.get('conversation_cache', {})
    if not cache.get('enabled') or cache.get('max_checkpoints') != 6 or cache.get('short_read') != 0:
        raise ValueError('Unexpected cache policy')
    wsl_base = ['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'costapad', '--']
    wsl_shell = wsl_base + ['bash', '-l', '-s']
    def shell(script, timeout):
        # Windows text-mode stdin translates LF into CRLF. Send Linux shell
        # bytes directly; otherwise the last argument silently gains a CR.
        result = subprocess.run(wsl_shell, input=(script + '\n').encode('utf-8'),
                                capture_output=True, timeout=timeout)
        result.stdout = result.stdout.decode('utf-8')
        result.stderr = result.stderr.decode('utf-8', errors='replace')
        return result
    node = '/home/costapad/.nvm/versions/node/v22.23.0/bin/node'
    # Check the actual gateway used by the wrapper, not only Windows localhost.
    def gateway_health():
        process = subprocess.run(wsl_base + [node, WSL_ROOT + '/probe_gateway.mjs'],
                                 capture_output=True, text=True, timeout=15)
        if process.returncode:
            error = process.stderr[:4096]
            (run / (prefix + '-gateway-failure.log')).write_text(error, encoding='utf-8')
            raise RuntimeError('Gateway probe failed: ' + error)
        record = json.loads(process.stdout)
        if record['health'].get('engine_sha256') != engine_sha:
            raise ValueError('Gateway is not the exact candidate')
        return record
    gateway_before = gateway_health()
    timing_path = run / 'timings.jsonl'
    before_rows = timing_path.read_text().splitlines() if timing_path.exists() else []
    proof_path = run / (prefix + '-requests.jsonl')
    if proof_path.exists():
        raise FileExistsError('Immutable request proof already exists')
    prompt = 'CACHE_UPDATE_20260927. '
    if opts.kind == 'text':
        prompt += ('Usa read per leggere smoke-fixture.txt nella directory corrente. '
                   'Rispondi solo con il valore di STRATA_GLOBAL_PI_OK. Non eseguire altre operazioni.')
    else:
        prompt += ('Usa read per aprire ../fixtures/alt-red.png e ../fixtures/blue.png. '
                   'Trascrivi il codice grande al centro di ciascuna immagine. Usa anche read per leggere '
                   'smoke-fixture.txt. Alla fine rispondi soltanto con JSON '
                   '{"red":"codice rosso","blue":"codice blu","text":"valore STRATA_GLOBAL_PI_OK"}. '
                   'Non eseguire altre operazioni.')
    wrapper = ['bash', WSL_ROOT + '/manual/pi-global-installed.sh'] if opts.wrapper == 'staged' else ['/home/costapad/.local/bin/pi']
    command = wrapper + ['--model', 'local-qwen38/qwen3.8-flash-next-local', '--thinking', 'off',
                         '--mode', 'json', '--no-session', '--tools', 'read', '--no-skills',
                         '--no-prompt-templates', '-e', WSL_ROOT + '/pi_request_proof.mjs', '-p', prompt]
    config_hash = 'sha256sum /home/costapad/.pi/agent/models.json /home/costapad/.pi/agent/settings.json'
    before_process = shell(config_hash, 10)
    if before_process.returncode:
        raise RuntimeError('Pi hash preflight: ' + before_process.stderr[:4096])
    before = before_process.stdout
    proof_relative = proof_path.relative_to(ROOT).as_posix()
    environment = ['env', 'PI_UPDATE_PROOF_PATH=' + WSL_ROOT + '/' + proof_relative,
                   'PI_UPDATE_PROOF_BASE=' + gateway_before['base_url'], 'PI_UPDATE_TEST_NONCE=CACHE_UPDATE_20260927']
    script = 'cd ' + shlex.quote(WSL_ROOT + '/manual') + ' && exec ' + shlex.join(environment + ['timeout', '--kill-after=5s', '240s'] + command)
    process = shell(script, 260)
    if len(process.stdout.encode()) > 1024 * 1024 or len(process.stderr.encode()) > 16 * 1024:
        raise ValueError('Pi evidence exceeds finite size limits')
    (run / (prefix + '.jsonl')).write_text(process.stdout, encoding='utf-8')
    (run / (prefix + '.stderr.log')).write_text(process.stderr, encoding='utf-8')
    after_process = shell(config_hash, 10)
    if after_process.returncode:
        raise RuntimeError('Pi hash postflight: ' + after_process.stderr[:4096])
    after = after_process.stdout
    gateway_after = gateway_health()
    request_proof = [json.loads(line) for line in proof_path.read_text().splitlines()] if proof_path.exists() else []
    all_timing_rows = timing_path.read_text().splitlines() if timing_path.exists() else []
    if all_timing_rows[:len(before_rows)] != before_rows:
        raise ValueError('Native timing evidence changed during Pi test')
    new_timings = [json.loads(line) for line in all_timing_rows[len(before_rows):]]
    events = [json.loads(line) for line in process.stdout.splitlines() if line.startswith('{')]
    messages = [event['message'] for event in events if event.get('type') == 'message_end']
    assistants = [message for message in messages if message.get('role') == 'assistant']
    tools = [message for message in messages if message.get('role') == 'toolResult']
    final = ''.join(item.get('text', '') for item in assistants[-1].get('content', []) if item.get('type') == 'text') if assistants else ''
    checks = {
        'exit0': process.returncode == 0,
        'same_provider_model': bool(assistants) and all(message.get('provider') == 'local-qwen38' and message.get('model') == 'qwen3.8-flash-next-local' for message in assistants),
        'config_unchanged': before == after,
        'exact_gateway_candidate': gateway_before['health']['engine_sha256'] == gateway_after['health']['engine_sha256'] == engine_sha,
        'payload_nonce_exact_route': bool(request_proof) and all(row.get('passed') and row.get('health_engine_sha256') == engine_sha for row in request_proof),
        'candidate_timings_correlated': bool(assistants) and len(assistants) == len(new_timings) == len(request_proof) and all(
            message.get('usage', {}).get('input') == row['native_timings']['prompt_tokens'] and
            message.get('usage', {}).get('output') == row['native_timings']['generated']
            for message, row in zip(assistants, new_timings)),
        'actual_read_text': any(message.get('toolName') == 'read' and not message.get('isError') and any(item.get('type') == 'text' and 'STRATA_GLOBAL_PI_OK' in item.get('text', '') for item in message.get('content', [])) for message in tools),
        'final_stop': bool(assistants) and assistants[-1].get('stopReason') == 'stop',
        'no_api_error': all(message.get('stopReason') != 'error' for message in assistants),
    }
    if opts.kind == 'vision':
        try:
            payload = json.loads(final.strip().removeprefix('```json').removesuffix('```').strip())
        except ValueError:
            payload = None
        checks['read_two_images'] = sum(item.get('type') == 'image' for message in tools for item in message.get('content', [])) == 2
        checks['correct_visual_text'] = payload == {'red': 'EMBER-9056', 'blue': 'COBALT-4428', 'text': 'ORCHID-7314'}
    else:
        checks['correct_text'] = final.strip() == 'ORCHID-7314'
    result = {'passed': all(checks.values()), 'checks': checks, 'engine_sha256': engine_sha,
              'gateway_health_before': gateway_before, 'gateway_health_after': gateway_after,
              'request_proof': request_proof, 'candidate_native_timings': new_timings,
              'config_hashes_before': before, 'config_hashes_after': after, 'final': final,
              'calls': [{'usage': message.get('usage'), 'stopReason': message.get('stopReason')} for message in assistants]}
    (run / (prefix + '-result.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()

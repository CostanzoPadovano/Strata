"""Auditable startup-only derivative of the frozen, admitted API observer."""

TIMING_FUNCTION = '''def timing_summary(timings):
    def rate(count, milliseconds):
        return f"{count * 1000 / milliseconds:.2f}" if milliseconds > 0 else "n/a"
    prompt = timings["prompt_tokens"]
    generated = timings["generated"]
    return (f"[timings] prefill {rate(prompt, timings['prompt_ms'])} tok/s "
            f"({prompt} token) | decode {rate(generated, timings['decode_ms'])} tok/s "
            f"({generated} token) | {timings['finish']}")


'''


def derive(source):
    source = source.replace('\r\n', '\n').rstrip() + '\n'
    def replace_once(before, after):
        nonlocal source
        if source.count(before) != 1:
            raise ValueError('Frozen observer shape changed; no automatic adaptation')
        source = source.replace(before, after, 1)
    replace_once('choices=["load", "smoke", "bench"]', 'choices=["bench"]')
    replace_once('        if opts.mode == "load":\n            result["passed"] = True\n            return\n', '')
    start = source.index('        smoke = {"model": MODEL, "messages":')
    stop = source.index('        if context != 98304:', start)
    removed = source[start:stop]
    if ('Fresh semantic smoke failed' not in removed or 'vision_smoke_passed=True' not in removed
            or 'if opts.mode == "smoke":' not in removed):
        raise ValueError('Expected isolated startup test block missing')
    source = source[:start] + '        result["startup_self_test"] = "disabled"\n        save()\n' + source[stop:]
    replace_once('def main() -> None:\n', TIMING_FUNCTION + 'def main() -> None:\n')
    replace_once('                            log.write(json.dumps(row) + "\\n")\n',
                 '                            log.write(json.dumps(row) + "\\n")\n'
                 '                        print(timing_summary(self.last), flush=True)\n')
    return source

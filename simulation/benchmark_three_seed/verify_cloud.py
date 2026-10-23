"""Cross-platform check before training on a new machine: the frozen protocol and sources are unchanged, and the native
backend of this machine reproduces the initial scores of every frozen validation and test start (eligibility must agree)."""
import json, sys
from pathlib import Path
import numpy as np
import modality3_study as M

out = M.ROOT / 'outputs' / M.NAME
M.configure(M.HISTORY); p = M.verify(out)
assert M.sha256(out / 'protocol.json') == M.read(out / 'protocol_digest.json')['protocol_sha256'], 'protocol changed'
res = {}
for name in ('validation_cases.json', 'test_cases.json'):
    cases = M.read(out / name); d = []; flips = []; scene = None; core = None
    for c in cases:
        if scene is None or scene.scene_id != c['scene_id']:
            scene = M.load_scene(M.resolve(c['row'])); core = M.HistoricalEnv([scene], max_steps=40)
        core.reset_start(c['start']); d.append(abs(core.score - c['initial_score']))
        if (core.score >= 20) != c['eligible']: flips.append(c['case_id'])
    res[name] = dict(n=len(cases), max_abs_diff=float(max(d)), eligibility_flips=flips)
print(json.dumps(dict(protocol_sha256=M.sha256(out / 'protocol.json'), **res), indent=1), flush=True)
sys.exit(1 if any(r['eligibility_flips'] for r in res.values()) else 0)

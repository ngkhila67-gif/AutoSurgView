"""Cross-platform check of the v2 (Linux) frozen cases against the v1 (Windows) frozen cases: same split, same starts, initial scores."""
import json, sys
from pathlib import Path
v1, v2 = Path(sys.argv[1]), Path(sys.argv[2]); rd = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
p1, p2 = rd(v1 / 'protocol.json'), rd(v2 / 'protocol.json'); res = dict(split_digest_equal=p1['split_digest'] == p2['split_digest'])
for n in ('validation_cases.json', 'test_cases.json'):
    a, b = rd(v1 / n), rd(v2 / n); assert [c['case_id'] for c in a] == [c['case_id'] for c in b]
    res[n] = dict(n=len(a), same_starts=all(x['start'] == y['start'] for x, y in zip(a, b)), max_abs_score_diff=max(abs(x['initial_score'] - y['initial_score']) for x, y in zip(a, b)),
                  eligibility_flips=[x['case_id'] for x, y in zip(a, b) if x['eligible'] != y['eligible']])
print(json.dumps(res, indent=1)); sys.exit(0 if res['split_digest_equal'] and not any(res[n]['eligibility_flips'] for n in ('validation_cases.json', 'test_cases.json')) else 1)

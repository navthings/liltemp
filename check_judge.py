import json
import math

from mlx_lm import load as mlx_load

import liltemp2 as lt

LN2 = math.log(2)
pairs = [tuple(p[:2]) for p in json.loads(open("runs2/prompts.json").read())[:100]]
ref, rtok = mlx_load(lt.REF)
h = lt.judge_nll(ref, rtok, pairs)
print(f"judge on 100 fresh 2026 passages: {sum(x['nll'] for x in h) / (LN2 * sum(x['bytes'] for x in h)):.3f} bpb", flush=True)
del ref, rtok
lt.free()
for name, repo in [("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B"), ("pythia-2.8b", "EleutherAI/pythia-2.8b")]:
    m, tok, _ = lt.load(repo, None)
    o = lt.own_nll(m, tok, pairs, 8, [1.0])
    print(f"{name} on the same passages: {sum(a[1.0] for a, _ in o) / (LN2 * sum(b for _, b in o)):.3f} bpb", flush=True)
    del m, tok
    lt.free()

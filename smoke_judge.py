import json
import math
import time

from mlx_lm import load as mlx_load

import liltemp2 as lt

pairs = [tuple(p[:2]) for p in json.loads(open("runs/prompts.json").read())[:200]]
t0 = time.time()
model, tok = mlx_load(lt.REF)
print(f"loaded judge in {time.time() - t0:.1f}s")
t0 = time.time()
out = lt.judge_nll(model, tok, pairs)
dt = time.time() - t0
nll = sum(o["nll"] for o in out)
b = sum(o["bytes"] for o in out)
print(f"7b-4bit judge on 200 wikitext continuations: {nll / (math.log(2) * b):.3f} bpb (qwen2.5-3b bf16 gave ~0.766), {dt:.1f}s")
print("early/late split adds up:", all(abs(o["nll_a"] + o["nll_b"] - o["nll"]) < 1e-3 for o in out),
      "bytes:", all(o["bytes_a"] + o["bytes_b"] == o["bytes"] for o in out))
print(out[0])

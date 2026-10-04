import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OUT", "runs"))
BOOT = 300
LN2 = math.log(2)


def read(name: str) -> list[dict]:
    p = OUT / name
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def bpb(rows, nll="ref_nll", nbytes="ref_bytes") -> float:
    b = sum(r[nbytes] for r in rows)
    return sum(r[nll] for r in rows) / (LN2 * b) if b else float("nan")


# first temperature where the curve crosses the human level, linearly interpolated
def crossing(temps: list[float], vals: list[float], target: float) -> float | None:
    for (t0, v0), (t1, v1) in zip(zip(temps, vals), zip(temps[1:], vals[1:])):
        if (v0 - target) * (v1 - target) <= 0 and v0 != v1:
            return t0 + (target - v0) * (t1 - t0) / (v1 - v0)
    return None


# share of distinct words: loops score low, word salad scores high, human text sits between
def distinct(text: str) -> float:
    w = text.lower().split()
    return len(set(w)) / len(w) if w else 0.0


def tstar(by_temp: dict, human: dict, idxs: list[int]):
    temps = sorted(by_temp)
    b_h = bpb([human[i] for i in idxs])
    b = [bpb([by_temp[t][i] for i in idxs]) for t in temps]
    d_h = sum(human[i]["distinct"] for i in idxs) / len(idxs)
    d = [sum(by_temp[t][i]["distinct"] for i in idxs) / len(idxs) for t in temps]
    return crossing(temps, b, b_h), crossing(temps, d, d_h), b, b_h


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    icpt = my - slope * mx
    ss_res = sum((y - icpt - slope * x) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return slope, icpt, 1 - ss_res / ss_tot if ss_tot else float("nan")


def main():
    gens, own, human = read("gens.jsonl"), read("own.jsonl"), {h["idx"]: h for h in read("human.jsonl")}
    if not gens:
        sys.exit(f"no results in {OUT}/ yet")
    prompts = json.loads((OUT / "prompts.json").read_text())
    for i, h in human.items():
        h["distinct"] = distinct(prompts[i][1])
    for r in gens:
        r["distinct"] = distinct(r["text"])
    g = defaultdict(lambda: defaultdict(dict))
    for r in gens:
        g[r["model"]][r["temp"]][r["idx"]] = r
    o = defaultdict(list)
    params = {}
    for r in own:
        o[r["model"]].append(r)
        params[r["model"]] = r.get("params")

    rng = random.Random(0)
    res = []
    for m, by_temp in g.items():
        idxs = sorted(set.intersection(*(set(v) for v in by_temp.values())) & set(human))
        own_bpb = bpb([r for r in o[m] if r["idx"] in set(idxs)], "nll", "bytes")
        t_b, t_r, curve, b_h = tstar(by_temp, human, idxs)
        boots = []
        for _ in range(BOOT):
            s = [rng.choice(idxs) for _ in idxs]
            t = tstar(by_temp, human, s)[0]
            if t is not None:
                boots.append(t)
        boots.sort()
        lo, hi = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]) if boots else (None, None)
        res.append({"model": m, "params": params.get(m), "n_prompts": len(idxs), "own_bpb": own_bpb,
                    "tstar": t_b, "tstar_lo": lo, "tstar_hi": hi, "tstar_distinct": t_r,
                    "human_bpb": b_h, "curve": dict(zip(sorted(by_temp), curve))})
    res.sort(key=lambda r: r["own_bpb"], reverse=True)

    fmt = lambda x: "  -  " if x is None else f"{x:.2f}"
    print(f"{'model':18} {'params':>7} {'n':>4} {'own bpb':>8} {'T*':>5} {'95% ci':>11} {'T*dist':>6}")
    for r in res:
        p = f"{r['params'] / 1e6:.0f}M" if r["params"] else "?"
        print(f"{r['model']:18} {p:>7} {r['n_prompts']:>4} {r['own_bpb']:8.3f} {fmt(r['tstar']):>5} "
              f"{fmt(r['tstar_lo']):>5}-{fmt(r['tstar_hi']):<5} {fmt(r['tstar_distinct']):>6}")

    ok = [r for r in res if r["tstar"] is not None]
    law = None
    if len(ok) >= 3:
        slope, icpt, r2 = fit([r["own_bpb"] for r in ok], [r["tstar"] for r in ok])
        law = {"slope": slope, "intercept": icpt, "r2": r2, "n_models": len(ok)}
        print(f"\nlaw: T* = {icpt:.3f} {'+' if slope >= 0 else '-'} {abs(slope):.3f} x own_bpb   (r2 {r2:.3f}, {len(ok)} models)")
    (OUT / "results.json").write_text(json.dumps({"models": res, "law": law}, indent=1))

    try:
        import matplotlib
    except ImportError:
        print("matplotlib not installed, skipping plots (results.json has everything)")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in res:
        a.plot(list(r["curve"]), list(r["curve"].values()), marker=".", label=r["model"])
    a.axhline(res[0]["human_bpb"], color="grey", ls="--", label="human text")
    a.set(xlabel="temperature", ylabel="reference bits per byte", title="how surprising each model's text is")
    a.legend(fontsize=7)
    for r in ok:
        err = [[r["tstar"] - r["tstar_lo"]], [r["tstar_hi"] - r["tstar"]]] if r["tstar_lo"] is not None else None
        b.errorbar(r["own_bpb"], r["tstar"], yerr=err, fmt="o", capsize=3)
        b.annotate(r["model"], (r["own_bpb"], r["tstar"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    if law:
        xs = [min(r["own_bpb"] for r in ok), max(r["own_bpb"] for r in ok)]
        b.plot(xs, [law["intercept"] + law["slope"] * x for x in xs], color="grey", ls="--")
    b.set(xlabel="model loss (bits per byte, lower = better)", ylabel="optimal temperature T*", title="T* vs model quality")
    fig.tight_layout()
    fig.savefig(OUT / "tstar.png", dpi=150)
    print(f"saved {OUT / 'tstar.png'}")


if __name__ == "__main__":
    main()

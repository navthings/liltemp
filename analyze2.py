import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OUT", "runs2"))
BOOT = 300
LN2 = math.log(2)


def read(name: str) -> list[dict]:
    p = OUT / name
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def bpb(rows, nll, nbytes) -> float:
    b = sum(r[nbytes] for r in rows)
    return sum(r[nll] for r in rows) / (LN2 * b) if b else float("nan")


# first temperature where a rising curve crosses the target, linearly interpolated
def crossing(temps, vals, target):
    for (t0, v0), (t1, v1) in zip(zip(temps, vals), zip(temps[1:], vals[1:])):
        if (v0 - target) * (v1 - target) <= 0 and v0 != v1:
            return t0 + (target - v0) * (t1 - t0) / (v1 - v0)
    return None


def family(name: str) -> str:
    return name.split("-")[0]


def measure(by_temp, own, human, idxs, part=""):
    temps = sorted(by_temp)
    h = bpb([human[i] for i in idxs], f"ref_nll{part}", f"ref_bytes{part}")
    curve = [bpb([by_temp[t][i] for i in idxs], f"ref_nll{part}", f"ref_bytes{part}") for t in temps]
    return crossing(temps, curve, h), curve, h


def self_temp(by_temp, own, idxs):
    # entropy calibration: the T where the model's surprise at its own T-samples equals its T-loss on human text
    temps = [t for t in sorted(by_temp) if str(t) in own[idxs[0]]["nll_t"]]
    gap = []
    for t in temps:
        s = bpb([by_temp[t][i] for i in idxs], "self_nll", "self_bytes")
        hb = sum(own[i]["bytes"] for i in idxs)
        l = sum(own[i]["nll_t"][str(t)] for i in idxs) / (LN2 * hb)
        gap.append(s - l)
    return crossing(temps, gap, 0.0)


def cal_temp(own, idxs):
    ts = sorted(own[idxs[0]]["nll_t"], key=float)
    vals = [sum(own[i]["nll_t"][t] for i in idxs) for t in ts]
    return float(ts[int(np.argmin(vals))])


def main():
    gens, own_rows = read("gens.jsonl"), read("own.jsonl")
    human = {h["idx"]: h for h in read("human.jsonl")}
    if not gens:
        sys.exit(f"no results in {OUT}/ yet")
    g = defaultdict(lambda: defaultdict(dict))
    for r in gens:
        g[r["model"]][r["temp"]][r["idx"]] = r
    own = defaultdict(dict)
    params = {}
    for r in own_rows:
        own[r["model"]][r["idx"]] = r
        params[r["model"]] = r["params"]

    rng = random.Random(0)
    res = []
    for m, by_temp in g.items():
        idxs = sorted(set.intersection(*(set(v) for v in by_temp.values())) & set(human) & set(own[m]))
        L = bpb([own[m][i] for i in idxs], "nll", "bytes")
        t_star, curve, h = measure(by_temp, own[m], human, idxs)
        t_a = measure(by_temp, own[m], human, idxs, "_a")[0]
        t_b = measure(by_temp, own[m], human, idxs, "_b")[0]
        boots = []
        for _ in range(BOOT):
            s = [rng.choice(idxs) for _ in idxs]
            t = measure(by_temp, own[m], human, s)[0]
            if t is not None:
                boots.append(t)
        boots.sort()
        lo, hi = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]) if boots else (None, None)
        res.append({"model": m, "family": family(m), "params": params.get(m), "n": len(idxs), "loss": L, "h": h,
                    "tstar": t_star, "lo": lo, "hi": hi, "tstar_early": t_a, "tstar_late": t_b,
                    "tself": self_temp(by_temp, own[m], idxs), "tcal": cal_temp(own[m], idxs),
                    "curve": dict(zip(map(str, sorted(by_temp)), curve))})
    res.sort(key=lambda r: r["loss"])

    f = lambda x: "  -  " if x is None else f"{x:.3f}"
    print(f"{'model':18} {'params':>7} {'n':>4} {'loss':>6} {'T*':>6} {'95% ci':>13} {'early':>6} {'late':>6} {'Tself':>6} {'Tcal':>6}")
    for r in res:
        print(f"{r['model']:18} {r['params'] / 1e6:6.0f}M {r['n']:>4} {r['loss']:6.3f} {f(r['tstar']):>6} "
              f"{f(r['lo'])}-{f(r['hi'])} {f(r['tstar_early']):>6} {f(r['tstar_late']):>6} {f(r['tself']):>6} {f(r['tcal']):>6}")

    ok = [r for r in res if r["tstar"] is not None]
    fits = {}
    if len(ok) >= 4:
        y = np.array([r["tstar"] for r in ok])
        h = ok[0]["h"]
        xs = {"loss": np.array([r["loss"] for r in ok]), "h/loss": np.array([h / r["loss"] for r in ok]),
              "log params": np.log10([r["params"] for r in ok])}
        for k, x in xs.items():
            A = np.column_stack([x, np.ones_like(x)])
            c = np.linalg.lstsq(A, y, rcond=None)[0]
            pred = A @ c
            loo = [abs(A[i] @ np.linalg.lstsq(np.delete(A, i, 0), np.delete(y, i), rcond=None)[0] - y[i]) for i in range(len(y))]
            fits[k] = {"slope": c[0], "icpt": c[1], "r2": 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                       "loo_mae": float(np.mean(loo)), "loo_max": float(np.max(loo))}
            print(f"T* ~ {c[1]:.3f} + {c[0]:.3f} x {k:10}  r2 {fits[k]['r2']:.3f}  loo {fits[k]['loo_mae']:.3f} (max {fits[k]['loo_max']:.3f})")
        # fit on pythia only, predict every other family
        tr = [r for r in ok if r["family"] == "pythia"]
        c = np.polyfit([h / r["loss"] for r in tr], [r["tstar"] for r in tr], 1)
        fits["transfer"] = []
        for r in ok:
            if r["family"] != "pythia":
                p = float(np.polyval(c, h / r["loss"]))
                inside = r["lo"] is not None and r["lo"] <= p <= r["hi"]
                fits["transfer"].append({"model": r["model"], "pred": p, "actual": r["tstar"], "inside": inside})
                print(f"  transfer {r['model']:14} predicted {p:.3f} actual {r['tstar']:.3f} {'inside ci' if inside else ''}")
    (OUT / "results.json").write_text(json.dumps({"models": res, "fits": fits}, indent=1))

    try:
        import matplotlib
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"pythia": "#1d4ed8", "smollm2": "#c2410c", "lilbase": "#15803d"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    a, b, c3 = axes
    for r in res:
        a.plot([float(t) for t in r["curve"]], list(r["curve"].values()), color=colors.get(r["family"], "grey"), alpha=0.6, lw=1)
    a.axhline(res[0]["h"], color="black", ls="--", lw=1, label="human text")
    a.set(xlabel="temperature", ylabel="judge bits per byte", title="how surprising each model's text is")
    a.legend(frameon=False)
    for r in ok:
        col = colors.get(r["family"], "grey")
        err = [[r["tstar"] - r["lo"]], [r["hi"] - r["tstar"]]] if r["lo"] is not None else None
        b.errorbar(r["loss"], r["tstar"], yerr=err, fmt="o", color=col, capsize=2, ms=4)
        if r["tself"] is not None:
            b.plot(r["loss"], r["tself"], "s", color=col, mfc="none", ms=4)
    b.set(xlabel="model loss (bits per byte)", ylabel="temperature", title="T* (filled) vs self-consistent T (open)")
    for r in ok:
        c3.plot(r["params"], r["tstar"], "o", color=colors.get(r["family"], "grey"), ms=4)
    c3.set(xscale="log", xlabel="parameters", ylabel="T*", title="T* vs size")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "results.png", dpi=150)
    print(f"saved {OUT / 'results.png'}")


if __name__ == "__main__":
    main()

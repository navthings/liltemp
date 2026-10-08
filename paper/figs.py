import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
R = json.load(open("../runs/results.json"))["models"]
h = R[0]["human_bpb"]
col = {"pythia": "#1d4ed8", "smollm2": "#c2410c", "lilbase": "#15803d"}
lab = {"pythia": "Pythia", "smollm2": "SmolLM2", "lilbase": "ours (297M)"}
fam = lambda m: m.split("-")[0]
L = np.array([r["own_bpb"] for r in R]); T = np.array([r["tstar"] for r in R])
a, b = np.polyfit(h / L, T, 1)[::-1]

# fig 1: T* against loss, fit drawn as a curve
fig, ax = plt.subplots(figsize=(4.4, 3.2))
xs = np.linspace(0.72, 1.23, 200)
ax.plot(xs, a + b * h / xs, color="0.6", lw=1, ls="--", zorder=0)
ck = sorted([r for r in R if r["model"].startswith("pythia-410m")], key=lambda r: -r["own_bpb"])
ax.plot([r["own_bpb"] for r in ck], [r["tstar"] for r in ck], color=col["pythia"], lw=0.8, alpha=0.5, zorder=1)
seen = set()
for r in R:
    f = fam(r["model"])
    ax.errorbar(r["own_bpb"], r["tstar"], yerr=[[r["tstar"] - r["tstar_lo"]], [r["tstar_hi"] - r["tstar"]]],
                fmt="o", ms=4, color=col[f], capsize=2, lw=0.8, label=None if f in seen else lab[f])
    seen.add(f)
for r in ck:
    step = r["model"].split("@")[1] if "@" in r["model"] else "143k"
    ax.annotate(step, (r["own_bpb"], r["tstar"]), xytext=(5, -9), textcoords="offset points", fontsize=7, color="0.35")
ax.set_xlabel("model loss on the prompts (bits per byte)")
ax.set_ylabel("best temperature $T^*$")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figs/tstar_loss.pdf")

# fig 2: judge surprise against temperature, coloured by loss
fig, ax = plt.subplots(figsize=(4.4, 3.2))
cm = plt.get_cmap("viridis")
lo_, hi_ = L.min(), L.max()
for r in sorted(R, key=lambda r: r["own_bpb"]):
    ts = [float(t) for t in r["curve"]]
    ax.plot(ts, list(r["curve"].values()), color=cm((r["own_bpb"] - lo_) / (hi_ - lo_)), lw=1)
ax.axhline(h, color="black", ls="--", lw=1)
ax.text(0.12, h + 0.06, "human text", fontsize=8)
sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(lo_, hi_))
fig.colorbar(sm, ax=ax, label="model loss (bits per byte)")
ax.set_xlabel("sampling temperature"); ax.set_ylabel("judge bits per byte of samples")
fig.tight_layout(); fig.savefig("figs/curves.pdf")

# fig 3: size vs loss as the predictor
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
pair = {"pythia-410m", "pythia-1.4b@16k"}
for r in R:
    f = fam(r["model"]); mk = "D" if r["model"] in pair else "o"
    a1.plot(r["params"], r["tstar"], mk, color=col[f], ms=4.5)
    a2.plot(r["own_bpb"], r["tstar"], mk, color=col[f], ms=4.5)
a1.set_xscale("log"); a1.set_xlabel("parameters"); a1.set_ylabel("$T^*$")
a2.set_xlabel("model loss (bits per byte)")
lp = np.log10([r["params"] for r in R])
r2p = np.corrcoef(lp, T)[0, 1] ** 2
pred = a + b * h / L; r2l = 1 - ((T - pred) ** 2).sum() / ((T - T.mean()) ** 2).sum()
a1.set_title(f"size: $r^2$ = {r2p:.2f}", fontsize=9); a2.set_title(f"loss: $r^2$ = {r2l:.2f}", fontsize=9)
fig.tight_layout(); fig.savefig("figs/size_vs_loss.pdf")
print("fit", a, b, "r2 size", r2p, "r2 loss", r2l)

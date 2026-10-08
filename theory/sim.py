import numpy as np
from scipy.optimize import minimize_scalar, brentq

rng = np.random.default_rng(0)
V, N = 4000, 1500
# true next-token distributions: zipf with a random steepness per context, so some contexts are near-certain and some are open
ranks = np.arange(1, V + 1)
alphas = rng.uniform(0.9, 2.2, size=N)
logp = np.stack([-a * np.log(ranks) for a in alphas])
logp = logp - np.log(np.exp(logp).sum(1, keepdims=True))
p = np.exp(logp)
eps = rng.standard_normal((N, V))
H = -(p * logp).sum(1).mean()

def lsm(x):
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))

def stats(logits, T):
    lq = lsm(logits / T); q = np.exp(lq)
    judge = -(q * logp).sum(1).mean()        # how surprising the model's samples are to the truth
    ent = -(q * lq).sum(1).mean()             # the model's own entropy
    ce = -(p * lq).sum(1).mean()              # log loss on real text
    return judge, ent, ce

def model(sigma, flat=1.0):
    raw = flat * logp + sigma * eps
    # a trained model is calibrated for log loss, so rescale its logits to the CE-optimal temperature
    b = minimize_scalar(lambda b: -(p * lsm(b * raw)).sum(1).mean(), bounds=(0.05, 5), method="bounded").x
    return b * raw

def solve(f, lo=0.2, hi=1.6):
    try:
        return brentq(f, lo, hi, xtol=1e-4)
    except ValueError:
        return float("nan")

rows = []
for sigma in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
    lg = model(sigma)
    L = stats(lg, 1.0)[2]
    t_judge = solve(lambda T: stats(lg, T)[0] - H)
    t_self = solve(lambda T: stats(lg, T)[1] - stats(lg, T)[2])
    t_cal = minimize_scalar(lambda T: stats(lg, T)[2], bounds=(0.3, 2), method="bounded").x
    rows.append((sigma, L, H / L, t_judge, t_self, t_cal))
    print(f"sigma {sigma:4.2f}  loss {L:6.3f}  H/L {H/L:5.3f}  T_judge {t_judge:5.3f}  T_self {t_self:5.3f}  T_cal {t_cal:5.3f}")
print(f"H(p) = {H:.3f} nats")
r = np.array(rows)
ok = ~np.isnan(r[:, 3])
c = np.polyfit(r[ok, 2], r[ok, 3], 1)
pred = np.polyval(c, r[ok, 2]); r2 = 1 - ((r[ok, 3] - pred) ** 2).sum() / ((r[ok, 3] - r[ok, 3].mean()) ** 2).sum()
print(f"T_judge ~ {c[1]:.3f} + {c[0]:.3f} * H/L   r2 {r2:.3f}")
np.save("sim_rows.npy", r)

import numpy as np
exec(open("sim.py").read().split("rows = []")[0])
# first-order prediction: T* ~ 1 - J / C, J = KL(p||q)+KL(q||p) at T=1, C = Cov_q(log q, log p)
for sigma in [0.25, 0.5, 0.75, 1.0, 1.25]:
    lg = model(sigma)
    lq = lsm(lg); q = np.exp(lq)
    kl_pq = (p * (logp - lq)).sum(1).mean()
    kl_qp = (q * (lq - logp)).sum(1).mean()
    excess = stats(lg, 1.0)[0] - H
    eq = lambda f: (q * f).sum(1, keepdims=True)
    C = (eq(lq * logp) - eq(lq) * eq(logp)).mean()
    t_pred = 1 - (kl_pq + kl_qp) / C
    t_true = solve(lambda T: stats(lg, T)[0] - H)
    L = stats(lg, 1.0)[2]
    print(f"sigma {sigma}: excess at T=1 {excess:.4f}  KL(p||q)+KL(q||p) {kl_pq+kl_qp:.4f}  loss gap {L-H:.4f}  C {C:.3f}  T* pred {t_pred:.3f} true {t_true:.3f}")

"""Toy T1: action-disturbance interaction ratio rho for a damped pendulum (CPU, numpy, seconds).

Checks the paper's section 3.2 identity (eq. 9-10) and the section 4.1 / Fig. 4 pattern on a toy:
additive force -> rho = 0; gain fault -> rho does not shrink with duration h; damping fault -> rho shrinks
roughly like h^2; saturating actuator -> small rho. Parameter values are our choice, not the paper's, so only
the qualitative pattern is compared with the paper.

Pendulum (paper Fig. 4 left): theta'' = -sin(theta) - 0.2 c theta' + u, observed pose = theta after duration h.
Factual action a, alternative a', disturbance eps ~ U(-0.5, 0.5) shared by both actions.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_interaction_results.json"
rng = np.random.default_rng(0)
N_STATES, N_DRAWS, STEPS = 400, 256, 40


def simulate(theta, omega, u, c, h):
    """RK4 integration of the pendulum over duration h; theta, omega, u, c broadcast together."""
    dt = h / STEPS

    def f(th, om):
        return om, -np.sin(th) - 0.2 * c * om + u

    for _ in range(STEPS):
        k1 = f(theta, omega)
        k2 = f(theta + dt / 2 * k1[0], omega + dt / 2 * k1[1])
        k3 = f(theta + dt / 2 * k2[0], omega + dt / 2 * k2[1])
        k4 = f(theta + dt * k3[0], omega + dt * k3[1])
        theta = theta + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        omega = omega + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return theta


MECH = {
    "additive": lambda a, e: (a + e, np.ones_like(e)),
    "gain": lambda a, e: (a * (1 + e), np.ones_like(e)),
    "saturating": lambda a, e: (1.2 * np.tanh((a + e) / 1.2), np.ones_like(e)),
    "damping": lambda a, e: (a, 1 + 1.5 * e),
}


def run(mech, h, theta0, omega0, a, a2, eps):
    """Outcome pose under factual action a and alternative a2 for every (state, draw): shape (states, draws)."""
    u_fn = MECH[mech]
    out = {}
    for name, act in (("a", a), ("a2", a2)):
        u, c = u_fn(act[:, None] * np.ones_like(eps)[None], eps[None, :] * np.ones((len(a), 1)))
        out[name] = simulate(theta0[:, None] * np.ones_like(eps)[None], omega0[:, None] * np.ones_like(eps)[None],
                             u, c, h)
    return out["a"], out["a2"]


def rho_and_identity(mech, h, seed=0):
    r = np.random.default_rng(seed)
    theta0, omega0 = r.uniform(-1, 1, N_STATES), r.uniform(-1, 1, N_STATES)
    a, a2 = r.uniform(-1, 1, N_STATES), r.uniform(-1, 1, N_STATES)
    eps = r.uniform(-0.5, 0.5, N_DRAWS)
    y, y2 = run(mech, h, theta0, omega0, a, a2, eps)
    delta = y2 - y                                          # individual action contrast, per (state, draw)
    rho = delta.var(axis=1, ddof=1).mean() / y2.var(axis=1, ddof=1).mean()
    # residual reuse with exact conditional means (mean over the disturbance draws, per state)
    mu1, mu2 = y.mean(1), y2.mean(1)                        # Monte Carlo conditional means under each action
    tr = y + (mu2 - mu1)[:, None]                           # residual reuse, eq. (8)
    tr_error = np.mean((tr - y2) ** 2)                      # eq. (9)-(10): risk of residual reuse
    tr_var = np.mean(delta.var(axis=1, ddof=0))             # tr Var(Delta) with the same ddof (population)
    return float(rho), float(tr_error), float(tr_var)


def main():
    hs = [0.1, 0.2, 0.4, 0.8]
    res = {}
    for mech in MECH:
        res[mech] = []
        for h in hs:
            rho, te, tv = rho_and_identity(mech, h)
            res[mech].append(dict(h=h, rho=float(f"{rho:.6g}"), residual_error=float(f"{te:.6g}"),
                                  tr_var_contrast=float(f"{tv:.6g}")))
    # slope of log(rho) against log(h) for the damping fault (expect about 2 if the rule is O(h^2))
    slopes = {}
    for mech in MECH:
        vals = [d["rho"] for d in res[mech]]
        if min(vals) > 0:
            slopes[mech] = round(float(np.polyfit(np.log(hs), np.log(vals), 1)[0]), 2)
        else:
            slopes[mech] = None
    checks = dict(
        additive_rho_below_1e4=all(abs(d["rho"]) < 1e-4 for d in res["additive"]),
        residual_error_equals_trvar_rel_1e9=all(abs(d["residual_error"] - d["tr_var_contrast"])
                                                <= 1e-9 * max(d["tr_var_contrast"], 1e-30) for m in MECH for d in res[m]),
    )
    OUT.write_text(json.dumps(dict(hs=hs, n_states=N_STATES, n_draws=N_DRAWS, rho=res, loglog_slope=slopes,
                                   checks=checks), indent=1, ensure_ascii=False))
    print(json.dumps(dict(loglog_slope=slopes, checks=checks), ensure_ascii=False))
    for m in MECH:
        print(m, [d["rho"] for d in res[m]])


if __name__ == "__main__":
    main()

"""Numerical checks of the key formulations in arXiv:2611.26600 (tutorial on
complementarity-based contact models). CPU only, numpy + scipy, runs in seconds.

Each check V<n> corresponds to a statement in analysis/03_derivations.md.
Output: analysis/formulation_checks_results.json
"""
import json
import os

import numpy as np
from scipy.optimize import minimize

RNG = np.random.default_rng(0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "formulation_checks_results.json")
results = {}


# ---------------------------------------------------------------- V1
# Projection onto the circular friction cone K = {x : ||x_t|| <= mu * x_n} (paper eq. 18-19)
def proj_circ(x, mu):
    xn, xt = x[0], x[1:]
    nt = np.linalg.norm(xt)
    if nt <= mu * xn:                       # x in K
        return x.copy()
    if xn <= -mu * nt:                      # x in polar cone K° = -K* = {s_n <= -mu ||s_t||}
        return np.zeros(3)
    that = xt / nt if nt > 0 else np.array([1.0, 0.0])
    g = np.concatenate([[1.0], mu * that]) / np.sqrt(1 + mu ** 2)   # boundary generator (eq. 18)
    return (g @ x) * g


def proj_brute(x, mu):
    """Argmin ||x - z|| over z in K, by SLSQP from several starts."""
    cons = [{"type": "ineq", "fun": lambda z: mu * z[0] - np.linalg.norm(z[1:])}]
    best = None
    for s in [np.array([1.0, 0, 0]), x.copy() + 1e-3, np.array([0.5, 0.2, 0.1])]:
        r = minimize(lambda z: np.sum((x - z) ** 2), s, constraints=cons, method="SLSQP",
                     options={"ftol": 1e-14, "maxiter": 500})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


errs = []
mu = 0.6
for _ in range(300):
    x = RNG.normal(size=3) * RNG.uniform(0.2, 2.0)
    p1 = proj_circ(x, mu)
    p2 = proj_brute(x, mu)
    errs.append(np.linalg.norm(p1 - p2))
# Moreau decomposition x = Pi_K(x) + Pi_{K°}(x) with Pi_{K°}(x) = -Pi_{K*}(-x) (text after eq. 18)
def proj_dual(x, mu):
    # K* = {s_n >= mu ||s_t||}; Pi_{K*}(y) = y in K*, else closed form via projection on its generator
    # K* is the circular cone with slope 1/mu: generator (1, t/mu)/sqrt(1+1/mu^2)
    yn, yt = x[0], x[1:]
    nt = np.linalg.norm(yt)
    if yn >= mu * nt:
        return x.copy()
    if yn <= -nt / mu:            # polar of K*: K*° = -K
        return np.zeros(3)
    that = yt / nt if nt > 0 else np.array([1.0, 0.0])
    nu = 1.0 / mu
    g = np.concatenate([[1.0], nu * that]) / np.sqrt(1 + nu ** 2)
    return (g @ x) * g

moreau_err = []
for _ in range(300):
    x = RNG.normal(size=3)
    polar = -proj_dual(-x, mu)
    moreau_err.append(np.linalg.norm(x - proj_circ(x, mu) - polar))
results["V1_projection_vs_bruteforce"] = {
    "mu": mu, "n": 300, "max_abs_err": float(np.max(errs)),
    "note": "closed-form eq.(19) vs SLSQP minimizer over K",
}
results["V1_moreau_decomposition"] = {
    "max_abs_err": float(np.max(moreau_err)),
    "note": "x = Pi_K(x) + Pi_Kpolar(x), Pi_Kpolar(x) = -Pi_K*(-x)",
}


# ---------------------------------------------------------------- V2
# Second-order-cone Fischer-Burmeister function (eqs. 27-29, 33-36)
def jordan(x, y):
    return np.concatenate([[x @ y], x[0] * y[1:] + y[0] * x[1:]])


def arw(x):
    M = np.zeros((3, 3))
    M[0, 0] = x[0]
    M[0, 1:] = x[1:]
    M[1:, 0] = x[1:]
    M[1:, 1:] = x[0] * np.eye(2)
    return M


def jsqrt(z):
    zt = np.linalg.norm(z[1:])
    c1, c2 = z[0] + zt, z[0] - zt
    zh = z[1:] / zt if zt > 0 else np.zeros(2)
    return 0.5 * np.concatenate([[np.sqrt(c1) + np.sqrt(c2)], (np.sqrt(c1) - np.sqrt(c2)) * zh])


def socfb(x, y):
    return x + y - jsqrt(jordan(x, x) + jordan(y, y))


def in_L(x):
    return x[0] >= np.linalg.norm(x[1:]) - 1e-12


# complementary pair in Lorentz cone L
x0 = np.array([1.0, 1.0, 0.0])
y0 = np.array([1.0, -1.0, 0.0])
res_compl = np.linalg.norm(socfb(x0, y0))
# a non-complementary pair in L (x'o y' != 0) must give nonzero residual
res_noncompl = np.linalg.norm(socfb(np.array([2.0, 0.5, 0.0]), np.array([1.0, 0.0, 0.3])))

# Jacobian d_x psi = I - Arw(s)^{-1} Arw(x)  (eq. 35) vs central finite differences
x = np.array([1.2, 0.3, -0.4])
y = np.array([0.9, -0.2, 0.5])
s = jordan(x, x) + jordan(y, y)
Jx_closed = np.eye(3) - np.linalg.solve(arw(jsqrt(s)), arw(x))
h = 1e-6
Jx_fd = np.zeros((3, 3))
for k in range(3):
    e = np.zeros(3)
    e[k] = h
    Jx_fd[:, k] = (socfb(x + e, y) - socfb(x - e, y)) / (2 * h)
results["V2_socfb"] = {
    "residual_complementary_pair": float(res_compl),
    "residual_noncomplementary_pair": float(res_noncompl),
    "jacobian_max_abs_diff_eq35_vs_fd": float(np.max(np.abs(Jx_closed - Jx_fd))),
    "note": "psi = 0 at a complementary pair, nonzero otherwise; closed-form Jacobian matches FD",
}


# ---------------------------------------------------------------- V3
# Stewart-Trinkle style pyramid (Appendix A.2): f_t = B_K lambda_t, 1^T lambda_t <= mu lambda_n.
# The admissible tangential set is conv{mu * that_k}; its inscribed radius in direction theta is
# mu * cos(pi/K) (edge midpoints). Measure it directly.
pyr = {}
for K in [3, 4, 6, 8]:
    angles = 2 * np.pi * np.arange(K) / K
    verts = np.stack([np.cos(angles), np.sin(angles)], axis=1) * mu
    thetas = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    radii = []
    for th in thetas:
        d = np.array([np.cos(th), np.sin(th)])
        # largest r with r*d in conv(verts): ray-polygon intersection
        best = np.inf
        for k in range(K):
            a, b = verts[k], verts[(k + 1) % K]
            M = np.array([d, -(b - a)]).T
            if abs(np.linalg.det(M)) < 1e-15:
                continue
            r, t = np.linalg.solve(M, a)
            if r >= 0 and 0 <= t <= 1:
                best = min(best, r)
        radii.append(best)
    radii = np.array(radii)
    pyr[str(K)] = {"min_radius_numeric": float(radii.min()),
                   "max_radius_numeric": float(radii.max()),
                   "min_radius_formula_mu_cos_pi_over_K": float(mu * np.cos(np.pi / K)),
                   "anisotropy_min_over_max": float(radii.min() / radii.max())}
results["V3_pyramid_anisotropy"] = {"mu": mu, "by_K": pyr}


# ---------------------------------------------------------------- 2D helpers for V4-V8
# 2D contact: f = (f_n, f_t), force cone K = {|f_t| <= mu f_n}, dual K* = {s_n >= mu |s_t|}
def proj_wedge(x, mu):
    xn, xt = x
    if abs(xt) <= mu * xn:                  # x in K
        return x.copy()
    if xn <= -mu * abs(xt):                 # x in polar cone K°
        return np.zeros(2)
    r1 = np.array([1.0, mu]) / np.sqrt(1 + mu ** 2)     # boundary rays of K
    r2 = np.array([1.0, -mu]) / np.sqrt(1 + mu ** 2)
    cands = [max(0.0, r @ x) * r for r in (r1, r2)]
    return min(cands, key=lambda p: np.sum((p - x) ** 2))


# Planar box (point mass) on a table, 2D (n, t): mass m, unit dt
m_box = 1.0
dt = 0.01

# ---------------------------------------------------------------- V4
# Lift-off with the convex (Anitescu) relaxation: drop Gamma(v) = mu |v_t| n.
# Single contact, G = dt/m I (2x2 identity-scaled), free velocity v_f = (v_nf, v_tf).
G = dt / m_box * np.eye(2)
v_f = np.array([-0.5, 1.0])        # approaching the table and sliding to the right
mu_lift = 0.5


def qcqp_relax(G, vf, mu, R=None):
    """min 1/2 f^T G f + vf^T f  s.t. f in K (2D wedge)."""
    if R is None:
        R = np.zeros_like(G)
    A = G + R
    cons = [{"type": "ineq", "fun": lambda f: mu * f[0] - abs(f[1]) + 0.0}]
    # use smooth form: mu*fn - sign*ft >= 0 for both signs => two inequalities
    cons = [{"type": "ineq", "fun": lambda f: mu * f[0] - f[1]},
            {"type": "ineq", "fun": lambda f: mu * f[0] + f[1]}]
    r = minimize(lambda f: 0.5 * f @ A @ f + vf @ f, np.array([1.0, 0.0]),
                 constraints=cons, method="SLSQP", options={"ftol": 1e-15, "maxiter": 1000})
    return r.x


f_rel = qcqp_relax(G, v_f, mu_lift)
vplus_rel = v_f + G @ f_rel
# relaxed (no de Saxce) complementarity: slipping => v_n+ = mu |v_t+|
lift_rel = {"f": f_rel.tolist(), "v_plus": vplus_rel.tolist(),
            "v_n_plus": float(vplus_rel[0]),
            "mu_abs_vt_plus": float(mu_lift * abs(vplus_rel[1])),
            "lift_off_rate_rel": float(vplus_rel[0] - mu_lift * abs(vplus_rel[1]))}
# exact (de Saxce) sliding solution in closed form: f_n = -v_nf/g (contact kept, v_n+=0),
# f_t = -mu f_n sign(v_t+) (Coulomb slip), so v_n+ = 0
g = dt / m_box
fn_ex = -v_f[0] / g
ft_ex = -mu_lift * fn_ex * np.sign(v_f[1])
vplus_ex = v_f + np.array([g * fn_ex, g * ft_ex])
lift_ex = {"f": [float(fn_ex), float(ft_ex)], "v_plus": vplus_ex.tolist(),
           "v_n_plus": float(vplus_ex[0]), "note": "exact N-CCP sliding branch, closed form"}
results["V4_lift_off_relaxation_vs_exact"] = {"relaxed": lift_rel, "exact_closed_form": lift_ex,
                                              "mu": mu_lift, "v_f": v_f.tolist()}


# ---------------------------------------------------------------- V5
# Pen held between two fingers (Section 6.2): regularization R > 0 removes static friction.
# Unconstrained interior minimizer of 1/2 f^T (G+R) f + v^T f is f* = -(G+R)^{-1} v.
# Tangential load W, normal press P: v_f = dt * (-P, -W)/m; the cone is not binding if W <= mu P.
P, W = 10.0, 1.0
F_ext = np.array([-P, -W])
v_pen = dt * F_ext / m_box
res_pen = {}
for R_scalar in [0.0, 0.2, 1.0]:
    R = R_scalar * np.eye(2) * dt / m_box          # R scaled like G (so it acts on the same timescale)
    # numeric minimizer on K
    f_num = qcqp_relax(G, v_pen, mu_lift, R=R)
    f_expected_t = W / (1 + R_scalar)
    res_pen[str(R_scalar)] = {"f_t_numeric": float(f_num[1]), "f_n_numeric": float(f_num[0]),
                              "f_t_analytic_W_over_1plusR": float(f_expected_t),
                              "required_to_hold_W": W,
                              "slip_fraction_deficit": float(1 - f_num[1] / W)}
# independence from pressing force
res_pen_P = {}
for P_try in [2.0, 10.0, 100.0]:
    v_try = dt * np.array([-P_try, -W]) / m_box
    f_try = qcqp_relax(G, v_try, mu_lift, R=1.0 * dt / m_box * np.eye(2))
    res_pen_P[str(P_try)] = float(f_try[1])
results["V5_regularization_static_friction_loss"] = {
    "P": P, "W": W, "mu": mu_lift, "by_R": res_pen, "f_t_vs_pressing_force_R1": res_pen_P,
    "note": "R=0 holds W exactly (f_t = W) while mu*P >= W; R>0 gives f_t = W/(1+R), independent of P",
}


# ---------------------------------------------------------------- V6
# NLS (eq. 61) vs regularized QCQP (eq. 68) vs KKT-consistent force from eq. (71), on a random
# 2-contact planar problem with 3 generalized coordinates.
nv, Nc = 3, 2
Mq = np.diag(RNG.uniform(0.5, 2.0, nv))
Mq = Mq + 0.1 * np.eye(nv)
J = RNG.normal(size=(2 * Nc, nv))                       # stacked 2D contact Jacobians
R_blocks = [RNG.uniform(0.05, 0.3) * np.eye(2) for _ in range(Nc)]  # R_ii: one scalar per contact (paper eq. 56)
R_full = np.zeros((2 * Nc, 2 * Nc))
for i, Ri in enumerate(R_blocks):
    R_full[2 * i:2 * i + 2, 2 * i:2 * i + 2] = Ri
mu_nls = 0.7
qdd_f = RNG.normal(size=nv)
r_vec = RNG.normal(size=2 * Nc)                          # r = v_f + gamma
a_ref = J @ qdd_f - r_vec / dt
Minv = np.linalg.inv(Mq)
Gmat = J @ Minv @ J.T


def proj_all(x):
    out = np.zeros_like(x)
    for i in range(Nc):
        out[2 * i:2 * i + 2] = proj_wedge(x[2 * i:2 * i + 2], mu_nls)
    return out


def nls_obj(qdd):
    dq = qdd - qdd_f
    z = a_ref - J @ qdd
    val = 0.5 * dq @ Mq @ dq
    for i in range(Nc):
        Ri_inv = np.linalg.inv(R_blocks[i])
        p = proj_wedge(z[2 * i:2 * i + 2], mu_nls)
        val += 0.5 * p @ Ri_inv @ p
    return val


r_nls = minimize(nls_obj, np.zeros(nv), method="BFGS", options={"gtol": 1e-10, "maxiter": 20000})
qdd_nls = r_nls.x
z_nls = a_ref - J @ qdd_nls
f_nls = np.concatenate([np.linalg.inv(R_blocks[i]) @ proj_wedge(z_nls[2 * i:2 * i + 2], mu_nls)
                        for i in range(Nc)])

# QCQP (68): min 1/2 f^T ((G+R) dt) f + r^T f  over f in K^Nc (block-diagonal cone)
A_q = (Gmat + R_full) * dt


def qcqp_multi(A, b):
    cons = []
    for i in range(Nc):
        cons.append({"type": "ineq", "fun": (lambda f, i=i: mu_nls * f[2 * i] - f[2 * i + 1])})
        cons.append({"type": "ineq", "fun": (lambda f, i=i: mu_nls * f[2 * i] + f[2 * i + 1])})
    best = None
    for s in [np.ones(2 * Nc) * 0.5, np.zeros(2 * Nc), f_nls + 0.05]:
        r = minimize(lambda f: 0.5 * f @ A @ f + b @ f, s, constraints=cons, method="SLSQP",
                     options={"ftol": 1e-15, "maxiter": 2000})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


f_qcqp = qcqp_multi(A_q, r_vec)
qdd_from_qcqp = qdd_f + Minv @ J.T @ f_qcqp
results["V6_nls_vs_regularized_qcqp"] = {
    "max_abs_diff_force": float(np.max(np.abs(f_nls - f_qcqp))),
    "max_abs_diff_accel": float(np.max(np.abs(qdd_nls - qdd_from_qcqp))),
    "force_norm": float(np.linalg.norm(f_nls)),
    "note": "eq.(61) NLS minimizer and eq.(68) SOCQP minimizer coincide; f = R^{-1} Pi_K(a_ref - J qdd)",
}


# ---------------------------------------------------------------- V7
# Kinetic energy identity (eq. 75): KE(q+)/dt^2 = const + 1/2 f^T G f + (v_f/dt)^T f
qdot_f = RNG.normal(size=nv)
v_f_full = J @ qdot_f
f_test = RNG.normal(size=2 * Nc)
qdot_plus = qdot_f + Minv @ J.T @ f_test * dt
KE = 0.5 * qdot_plus @ Mq @ qdot_plus
lhs = KE / dt ** 2
rhs = 0.5 * f_test @ Gmat @ f_test + (v_f_full / dt) @ f_test
const = 0.5 * qdot_f @ Mq @ qdot_f / dt ** 2
results["V7_kinetic_energy_identity"] = {
    "max_abs_residual_after_removing_const": float(abs((lhs - const) - rhs)),
    "note": "paper eq.(75) identity; gamma = 0 case",
}


# ---------------------------------------------------------------- V8
# Pen-style exact sliding solution vs relaxed solution (V4) is done. Here: static friction under
# compliant normal (Table 1, penalty row). Penalty normal force k*p balances m g at p = m g / k.
g_acc = 9.81
pen = {}
for k in [1e2, 1e3, 1e4]:
    pen[str(k)] = {"penetration_m": float(m_box * g_acc / k)}
results["V8_penalty_penetration_scaling"] = {"m": m_box, "g": g_acc, "by_k": pen,
                                             "note": "penalty steady penetration = m g / k; "
                                                     "complementarity time-stepping gives 0 exactly"}

# ---------------------------------------------------------------- V9
# Force non-uniqueness with rank-deficient G (Table 1 / Section 6.1, "box on four corner contacts").
# 3D rigid box with 6 dof; only z, wx, wy are constrained by four normal-only corner contacts.
# Normal velocity at corner (x_i, y_i): v_z + y_i*wx - x_i*wy  -> row [1, y_i, -x_i] on (vz, wx, wy).
corners = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
Jn = np.zeros((4, 6))
for i, (x, y) in enumerate(corners):
    Jn[i, 2] = 1.0
    Jn[i, 3] = y
    Jn[i, 4] = -x
Mb = np.diag([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])
Gb = Jn @ np.linalg.inv(Mb) @ Jn.T
rank_G = int(np.linalg.matrix_rank(Gb))
U, S, Vt = np.linalg.svd(Gb)
ns = U[:, -1]
results["V9_rank_deficient_contacts"] = {
    "rank_G": rank_G, "n_contacts": 4,
    "forces_null_dimension": int(4 - rank_G),
    "null_vector_of_G": [float(v) for v in ns],
    "Jn_T_null_vector_norm": float(np.linalg.norm(Jn.T @ ns)),
    "note": "rank 3 (vz, wx, wy): sum AND both moments of the forces are fixed by the motion; "
            "one free force direction remains. Text says only the sum is unique -> stricter than rank gives",
}

# ---------------------------------------------------------------- V10
# Section 6.1 says regularization "selects the one with the smallest weighted norm f^T R f" among the
# exact solutions. Test that on a frictionless (R+ cone) rank-deficient case: 4 normal contacts, 3 dof.
# Exact solution set S = {f >= 0 : G f = b}. Compare argmin_S f^T R f with the regularized minimizer.
Gn = Gb * dt                                   # use same G as V9 (3D box), time-scaled
f0 = np.array([1.0, 2.0, 0.5, 1.5])            # a strictly positive feasible force
b = Gn @ f0
w_R = np.diag([1.0, 2.0, 3.0, 0.5])
# exact set S is a segment f0 + t*d (d spans null(G)); minimize the quadratic in t over f >= 0
_, _, Vt_n = np.linalg.svd(Gn)
d_null = Vt_n[-1]
lo, hi = -np.inf, np.inf
for fi, di in zip(f0, d_null):
    if abs(di) < 1e-14:
        continue
    if di > 0:
        lo = max(lo, -fi / di)
    else:
        hi = min(hi, fi / (-di))
ts = np.linspace(lo, hi, 200001)
vals = np.array([(f0 + t * d_null) @ w_R @ (f0 + t * d_null) for t in ts])
f_sel = f0 + ts[np.argmin(vals)] * d_null
r_vec_n = -b                                    # linear term making the exact minimizer set = S
reg_rows = {}
for eps in [1e-1, 1e-3, 1e-5, 1e-7]:
    Rn = eps * w_R
    reg = minimize(lambda f: 0.5 * f @ (Gn + Rn) @ f + r_vec_n @ f, f0, jac=lambda f: (Gn + Rn) @ f + r_vec_n,
                   method="L-BFGS-B", bounds=[(0, None)] * 4, options={"ftol": 1e-16, "gtol": 1e-12, "maxiter": 10000})
    reg_rows[str(eps)] = {"f_reg": [float(v) for v in reg.x],
                          "gap_to_selection": float(np.linalg.norm(reg.x - f_sel)),
                          "residual_Gf_minus_b": float(np.linalg.norm(Gn @ reg.x - b))}
results["V10_regularization_selection_claim"] = {
    "f_selected_argmin_fRf_over_exact_set": [float(v) for v in f_sel],
    "by_eps": reg_rows,
    "note": "regularized minimizer is not the exact-set selection; the gap shrinks with eps only as "
            "a perturbation, so 'selects the smallest weighted norm' holds only approximately",
}

with open(OUT, "w") as fh:
    json.dump(results, fh, indent=2, ensure_ascii=False)
print(json.dumps(results, indent=2, ensure_ascii=False)[:4000])

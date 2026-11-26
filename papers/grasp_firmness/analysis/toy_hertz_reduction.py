"""토이 T1: 식 (1)(2)(5)의 힘 감소율(그림 2 형태) 재구성.
Hertz 접촉 A(F)와 µF + τA = m g/2 (턱당 하중 가정)를 풀어, Coulomb-only 힘 대비 감소율을 E에 따라 계산한다.
R*, τ, 턱당 분배는 논문에 없는 가정이므로 값을 바꿔 민감도를 본다. 실행: python3 analysis/toy_hertz_reduction.py
"""
import json, numpy as np
from scipy.optimize import brentq  # noqa
E_GRIP, NU = 245e3, 0.48   # 부록 S2.2
g = 9.81

def area(F, E_obj, R):
    inv = (1-NU**2)/E_obj + ((1-NU**2)/E_GRIP if E_GRIP else 0.0)
    Es = 1/inv
    return np.pi*(3*F*R/(4*Es))**(2/3)

def fmin(m, mu, tau, E_obj, R, per_jaw_share):
    load = m*g*per_jaw_share
    f = lambda F: mu*F + tau*area(F, E_obj, R) - load
    if f(1e-9) >= 0: return 0.0
    return brentq(f, 1e-9, load/mu)

def reduction(m, mu, tau, E_obj, R, share):
    fc = m*g*share/mu
    return 100*(1-fmin(m, mu, tau, E_obj, R, share)/fc)

res = {}
Es = [10e3, 50e3, 100e3, 200e3, 400e3, 700e3, 1000e3]
for share in (1.0, 0.5):
  for tau in (17e3, 5e3, 1e3):
    for R in (0.02, 0.04):
        key = f"share{share}_tau{int(tau/1e3)}kPa_R{int(R*1e3)}mm"
        res[key] = {
          "m0.05": [round(reduction(0.05, 0.5, tau, e, R, share), 1) for e in Es],
          "m0.50": [round(reduction(0.50, 0.5, tau, e, R, share), 1) for e in Es]}
res["E_kPa"] = [e/1e3 for e in Es]
# 그림 6c 네크타린(그림 판독 값): m=192 g, mu=0.59, F_opt=1.12 N/턱, E=42 kPa
m, mu, Fopt = 0.192, 0.59, 1.12
need = m*g/2
res["nectarine_fig6"] = {"load_per_jaw_N": need, "coulomb_only_F": need/mu,
    "tauA_needed_N": need-mu*Fopt, "tauA_share_%": 100*(need-mu*Fopt)/need,
    "tau_implied_kPa_if_R40mm": (need-mu*Fopt)/area(Fopt, 42e3, 0.04)/1e3,
    "A_cm2_R40mm": area(Fopt, 42e3, 0.04)*1e4}
# 그림 2 판독 값(E=50 kPa 가벼운 쪽 18.7 %, 무거운 쪽 8.7 %; E=1000 kPa 2.4 %, 1.2 %)에 τ를 맞추고 나머지를 예측
fit = {}
for share in (1.0, 0.5):
    for R in (0.02, 0.04):
        tau = brentq(lambda t: reduction(0.05, 0.5, t, 50e3, R, share) - 18.7, 1e1, 5e4)
        fit[f"share{share}_R{int(R*1e3)}mm"] = {"tau_fit_Pa": round(tau, 1),
            "pred_m0.50_E50": round(reduction(0.50, 0.5, tau, 50e3, R, share), 1),
            "pred_m0.05_E1000": round(reduction(0.05, 0.5, tau, 1000e3, R, share), 1),
            "pred_m0.50_E1000": round(reduction(0.50, 0.5, tau, 1000e3, R, share), 1)}
res["fit_to_fig2"] = fit
# 같은 맞춤을 강체 그리퍼(E_grip -> inf)로 반복
E_GRIP = None
fit2 = {}
for share in (0.5,):
    for R in (0.04,):
        tau = brentq(lambda t: reduction(0.05, 0.5, t, 50e3, R, share) - 18.7, 1e1, 5e4)
        fit2[f"rigid_gripper_share{share}_R{int(R*1e3)}mm"] = {"tau_fit_Pa": round(tau, 1),
            "pred_m0.50_E50": round(reduction(0.50, 0.5, tau, 50e3, R, share), 1),
            "pred_m0.05_E1000": round(reduction(0.05, 0.5, tau, 1000e3, R, share), 1),
            "pred_m0.50_E1000": round(reduction(0.50, 0.5, tau, 1000e3, R, share), 1)}
res["fit_to_fig2_rigid_gripper"] = fit2
E_GRIP = 245e3
res["fig2_read"] = {"m0.05_E50": 18.7, "m0.50_E50": 8.7, "m0.05_E1000": 2.4, "m0.50_E1000": 1.2}
json.dump(res, open("analysis/toy_hertz_reduction_results.json", "w"), indent=1)
for k, v in res.items(): print(k, v)

"""LeWorldModel (arXiv:2603.19312 v3) 표 값 재계산과 일관성 검사.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json
필요: numpy, scipy (CPU, 1초 안팎). 네트워크·학습·공식 코드 실행 없음.

검사 목록 (내부 라벨 V = 검증):
  V1 표 4 Overall 행이 여덟 행의 평균과 일치하는지
  V2 표 1 전 행에서 LeWM이 PLDM보다 나은지 (MSE 낮음, r 높음)
  V3 그림 6 값의 이항 신뢰구간과 대비 검정 (에피소드 50개 한 번, 또는 시드 3개 × 50 = 150 가정)
  V4 표 5 (시드 3개) LeWM 대 PLDM, 대 DINO-WM+prop의 Welch 검정 (시드 수준)
  V5 "평균 ± 값" 표 5–10의 SD 정의(모집단/표본)와 성공 횟수 격자(시도 50, 2%p 단위) 가능성
  V6 그림 3 속도 배수 47 s / 0.98 s
  V7 SIGReg 식 검산 (논문 식과 공식 코드 수식을 독립 numpy로 옮겨 N(0,I)와 붕괴 임베딩을 비교)
"""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

# ---------- V1: 표 4 Overall ----------
t4 = {  # (linear MSE, linear r, MLP MSE, MLP r)
    "DINO-WM": {
        "Joint Position": (0.960, 0.808, 0.200, 0.870),
        "Joint Velocity": (0.792, 0.763, 0.263, 0.852),
        "End-Effector Position": (0.024, 0.996, 0.004, 0.999),
        "End-Effector Yaw": (3.317, 0.828, 0.167, 0.917),
        "Gripper": (0.114, 0.943, 0.038, 0.982),
        "Block Position": (0.085, 0.991, 0.007, 0.998),
        "Block Quaternion": (1.596, 0.257, 0.769, 0.411),
        "Block Yaw": (4.223, 0.176, 0.916, 0.304),
    },
    "PLDM": {
        "Joint Position": (0.372, 0.695, 0.340, 0.728),
        "Joint Velocity": (1.016, 0.115, 0.661, 0.536),
        "End-Effector Position": (0.052, 0.974, 0.013, 0.993),
        "End-Effector Yaw": (0.996, 0.056, 0.985, 0.117),
        "Gripper": (0.234, 0.876, 0.066, 0.967),
        "Block Position": (0.031, 0.985, 0.003, 0.999),
        "Block Quaternion": (1.021, 0.066, 0.989, 0.218),
        "Block Yaw": (0.996, 0.061, 0.989, 0.106),
    },
    "LeWM": {
        "Joint Position": (0.352, 0.706, 0.330, 0.742),
        "Joint Velocity": (1.021, 0.095, 0.818, 0.386),
        "End-Effector Position": (0.018, 0.991, 0.003, 0.998),
        "End-Effector Yaw": (0.980, 0.124, 0.952, 0.213),
        "Gripper": (0.121, 0.938, 0.048, 0.976),
        "Block Position": (0.007, 0.997, 0.002, 0.999),
        "Block Quaternion": (1.019, 0.087, 0.963, 0.224),
        "Block Yaw": (0.996, 0.062, 0.973, 0.164),
    },
}
reported_overall = {
    "DINO-WM": (1.162, 0.725, 0.290, 0.799),
    "PLDM": (0.611, 0.464, 0.503, 0.600),
    "LeWM": (0.592, 0.477, 0.525, 0.584),
}
v1 = {}
for m in t4:
    arr = np.array(list(t4[m].values()))
    mean = arr.mean(axis=0)
    v1[m] = {
        "recomputed_mean_of_8_rows": [round(float(x), 3) for x in mean],
        "reported_overall": list(reported_overall[m]),
        "diff": [round(float(a - b), 3) for a, b in zip(mean, reported_overall[m])],
    }
res["V1_table4_overall"] = v1

# ---------- V2: 표 1 LeWM vs PLDM 전 행 ----------
t1 = {  # property: ((lin MSE, lin r, MLP MSE, MLP r) for LeWM, PLDM, DINO-WM)
    "Agent Location": ((0.052, 0.974, 0.004, 0.998), (0.090, 0.955, 0.014, 0.993), (1.888, 0.977, 0.003, 0.999)),
    "Block Location": ((0.029, 0.986, 0.001, 0.999), (0.122, 0.938, 0.011, 0.994), (0.006, 0.997, 0.002, 0.999)),
    "Block Angle": ((0.187, 0.902, 0.021, 0.990), (0.446, 0.745, 0.056, 0.972), (0.050, 0.979, 0.009, 0.995)),
}
v2 = {}
for prop, (l, p, d) in t1.items():
    v2[prop] = {
        "LeWM_better_than_PLDM_all_4": bool(l[0] < p[0] and l[1] > p[1] and l[2] < p[2] and l[3] > p[3]),
        "LeWM_linear_r_beats_DINO": bool(l[1] > d[1]),
        "DINO_linear_MSE_lower_than_LeWM": bool(d[0] < l[0]),
    }
res["V2_table1_checks"] = v2

# ---------- V3: 그림 6 값 (시각 판독, v1과 v3 동일) ----------
fig6 = {
    "Two-Room": {"LeWM": 87, "DINO-WM+prop": 100, "PLDM": 97, "DINO-WM": 100, "GCBC": 100, "GCIQL": 100, "GCIVL": 100, "Random": 0},
    "Reacher": {"LeWM": 86, "PLDM": 78, "DINO-WM": 79, "Random": 10},
    "Push-T": {"LeWM": 96, "DINO-WM+prop": 92, "PLDM": 78, "DINO-WM": 74, "GCBC": 75, "GCIQL": 20, "GCIVL": 33, "Random": 2},
    "OGBench-Cube": {"LeWM": 74, "PLDM": 65, "DINO-WM": 86, "GCBC": 84, "GCIQL": 64, "GCIVL": 56, "Random": 48},
}


def two_prop_z(p1, p2, n1, n2):
    """두 비율 차이의 정규 근사 z 검정 (독립 가정)."""
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = (p1 - p2) / se if se > 0 else np.nan
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def wilson(p, n, z=1.96):
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


v3 = {"assumption": "그림 6 값을 k/50 비율로 본다(표 5의 50 궤적). 시드 3개 합산이면 n = 150으로 본다.", "cells": {}}
pairs = [("Two-Room", "LeWM", "PLDM"), ("Two-Room", "LeWM", "DINO-WM"),
         ("Reacher", "LeWM", "PLDM"), ("Reacher", "LeWM", "DINO-WM"),
         ("Push-T", "LeWM", "PLDM"), ("Push-T", "LeWM", "DINO-WM+prop"), ("Push-T", "LeWM", "DINO-WM"),
         ("OGBench-Cube", "LeWM", "DINO-WM"), ("OGBench-Cube", "LeWM", "PLDM")]
for env, a, b in pairs:
    pa, pb = fig6[env][a] / 100, fig6[env][b] / 100
    row = {}
    for n in (50, 150):
        z, p = two_prop_z(pa, pb, n, n)
        row[f"n{n}"] = {"diff_pp": round(100 * (pa - pb), 1), "z": round(z, 2), "p": round(p, 4)}
    v3["cells"][f"{env}: {a} vs {b}"] = row
v3["wilson95_n50"] = {
    f"{env}/{m}": [round(x * 100, 1) for x in wilson(fig6[env][m] / 100, 50)]
    for env in fig6 for m in ("LeWM", "PLDM", "DINO-WM")
    if m in fig6[env]
}
# 50 궤적 한 번으로 가능한 정수 백분율인지 (k/50 은 2%p 단위)
possible50 = {env: {m: (v, bool(abs(v / 2 - round(v / 2)) < 1e-9)) for m, v in d.items()} for env, d in fig6.items()}
v3["k_over_50_feasible"] = {env: {m: ok for m, (v, ok) in d.items() if not ok} for env, d in possible50.items()}
res["V3_fig6_tests"] = v3

# ---------- V4: 표 5 시드 수준 Welch ----------
def welch(m1, s1, m2, s2, n=3, ddof=1):
    """보고된 평균과 SD (ddof=1 가정)로 Welch t 검정."""
    v1_ = (s1 ** 2) * n / (n - 1) if ddof == 0 else s1 ** 2
    v2_ = (s2 ** 2) * n / (n - 1) if ddof == 0 else s2 ** 2
    se2 = v1_ / n + v2_ / n
    t = (m1 - m2) / np.sqrt(se2)
    df = se2 ** 2 / ((v1_ / n) ** 2 / (n - 1) + (v2_ / n) ** 2 / (n - 1))
    p = 2 * stats.t.sf(abs(t), df)
    return round(float(t), 2), round(float(df), 2), round(float(p), 4)


v4 = {
    "LeWM_vs_PLDM_pushT_ddof0": welch(96.0, 2.83, 78.0, 5.0, ddof=0),
    "LeWM_vs_PLDM_pushT_ddof1": welch(96.0, 2.83, 78.0, 5.0, ddof=1),
    "LeWM_vs_DINOWMprop_pushT_ddof0": welch(96.0, 2.83, 92.0, 1.63, ddof=0),
    "LeWM_vs_DINOWMprop_pushT_ddof1": welch(96.0, 2.83, 92.0, 1.63, ddof=1),
    "note": "시드 3개뿐이라 검정력이 매우 낮다. SD 정의를 모르므로 두 경우를 함께 적는다.",
}
res["V4_seed_welch"] = v4

# ---------- V5: 시드 3개 × 50 궤적 격자에서 보고된 평균과 SD를 만들 수 있는지 ----------
def feasible(mean_str, sd_str, n=3, k=50):
    """성공률 후보 = 2%p 단위 (0,2,...,100). 평균과 SD가 보고 자릿수로 반올림되는 세 값의 조합이 있는지."""
    dp_m = len(mean_str.split(".")[1]) if "." in mean_str else 0
    dp_s = len(sd_str.split(".")[1]) if "." in sd_str else 0
    mean_v, sd_v = float(mean_str), float(sd_str)
    grid = [2 * i for i in range(0, k + 1)] if k == 50 else list(range(0, 101))
    out = {}
    for ddof in (0, 1):
        ok = False
        for combo in itertools.combinations_with_replacement(grid, n):
            m = np.mean(combo)
            s = np.std(combo, ddof=ddof)
            if round(m, dp_m) == mean_v and round(s, dp_s) == sd_v:
                ok = True
                break
        out[f"ddof{ddof}"] = ok
    return out


cases = {
    "T5 LeWM PushT": ("96.0", "2.83"), "T5 DINO-WM PushT(+prop?)": ("92.0", "1.63"), "T5 PLDM PushT": ("78.0", "5.0"),
    "T6 tiny": ("80.67", "6.54"), "T6 small": ("96.0", "2.83"), "T6 base": ("86.7", "3.06"),
    "T7 with decoder": ("86.0", "7.54"), "T8 ResNet-18": ("94.0", "3.27"),
    "T9 p=0.0": ("78", "6.54"), "T9 p=0.2": ("85.33", "5.74"), "T9 p=0.5": ("66.67", "4.11"),
    "T10 SGD LeWM": ("26", "4.32"), "T10 PLDM SGD": ("4.67", "0.06"), "T10 RMSProp LeWM": ("67.33", "2.49"),
    "T10 Adam LeWM": ("84", "7.12"),
}
v5 = {}
for name, (m, s) in cases.items():
    v5[name] = {"reported": f"{m} ± {s}", "feasible_with_2pp_grid_3seeds_50ep": feasible(m, s)}
res["V5_granularity"] = v5

# ---------- V6: 속도 배수 ----------
res["V6_speedup"] = {"dino_full_planning_s": 47, "lewm_full_planning_s": 0.98,
                    "ratio": round(47 / 0.98, 2), "reported": "48x"}

# ---------- V7: SIGReg 식 검산 (독립 numpy 구현, 공식 코드 수식 기준) ----------
def sigreg(Z, knots=17, num_proj=1024, rng=None):
    """Z: (T, B, D). 공식 module.py의 SIGReg 수식을 옮긴 것. t in [0, 3], 가중치 exp(-t^2/2)."""
    rng = rng or np.random.default_rng(0)
    t = np.linspace(0, 3, knots)
    dt = 3 / (knots - 1)
    w = np.full(knots, 2 * dt)
    w[[0, -1]] = dt
    phi = np.exp(-t**2 / 2)
    weights = w * phi
    A = rng.standard_normal((Z.shape[-1], num_proj))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    x_t = (Z @ A)[..., None] * t  # (T, B, P, K)
    err = (np.cos(x_t).mean(axis=1) - phi) ** 2 + np.sin(x_t).mean(axis=1) ** 2  # (T, P, K)
    stat = (err @ weights) * Z.shape[1]  # (T, P)
    return float(stat.mean())


rng = np.random.default_rng(1)
T, B, D = 4, 128, 192
gauss = rng.standard_normal((T, B, D))
collapsed = np.repeat(rng.standard_normal((T, 1, D)) * 0.01, B, axis=1) + 1e-3 * rng.standard_normal((T, B, D))
aniso = rng.standard_normal((T, B, D)) * np.linspace(1, 3, D)  # 분산이 차원마다 다름
res["V7_sigreg_check"] = {
    "setting": f"T={T}, B={B}, D={D}, 투영 1024, 매듭 17",
    "isotropic_gaussian": round(sigreg(gauss, rng=np.random.default_rng(2)), 4),
    "collapsed_near_constant": round(sigreg(collapsed, rng=np.random.default_rng(2)), 4),
    "anisotropic_gaussian": round(sigreg(aniso, rng=np.random.default_rng(2)), 4),
    "note": "논문 식 (EP)에는 배치 크기 N 곱셈이 없고 t 구간은 [0.2, 4]로 적혀 있으나, 공식 코드는 [0, 3]과 N 배 곱셈을 쓴다(04 D4).",
}

with OUT.open("w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])

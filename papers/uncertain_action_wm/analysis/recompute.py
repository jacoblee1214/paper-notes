"""논문 표 수치 재계산과 검정 (수 초, numpy·scipy).

1) 표 1 산술: ∆Frozen, ∆ACTUS가 SR 차이와 맞는지 (SR은 소수 셋째 자리)
2) 계수 자료의 Wilson / Clopper-Pearson 95% 구간 (τ=1 보정, 방향 판정, 경보)
3) Prop. 2: 정확 AUROC(비중심 χ² 대 중심 χ²)와 식 (8) 근사, λ0.75 비교
4) Thm. 2 분해: Fig. 5b 값과 표 1에서 Lbox, Llat, Lest 계산
5) 실로봇 r = 0.84의 Fisher z 구간 (n = 13, 26 두 경우)

실행: python3 analysis/recompute.py  → analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import brentq

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

# ---------- 1) 표 1 산술 ----------
# (이름, PushT SR, 표의 ∆Frozen, 표의 ∆ACTUS, Cube SR, 표의 ∆Frozen, 표의 ∆ACTUS)
# 표 1에 없는 ∆ 값은 None. SR 반올림 오차(±0.0005 × 2 → ±0.1 pt)를 허용 오차 0.15 pt로 둔다.
T1 = [
    ("Frozen", 0.650, None, None, 0.378, None, None),
    ("AdaJEPA", 0.627, -2.3, -23.5, 0.347, -3.1, -8.4),
    ("OpTTA", 0.600, -5.0, -26.2, 0.346, -3.2, -8.6),
    ("JEPA-TTT", 0.673, 2.3, -18.9, 0.391, 1.3, -4.1),
    ("Sandwich-Res.", 0.634, -1.6, -22.8, 0.368, -1.0, -6.3),
    ("ALRA", 0.634, -1.6, -22.8, 0.338, -4.0, -9.4),
    ("Robust planning", 0.526, -12.4, -33.6, 0.291, -8.7, -14.0),
    ("ActLin", 0.778, 12.8, -8.4, 0.364, -1.4, -6.7),
    ("Lie-SGD", 0.829, 17.9, -3.3, 0.388, 1.0, -4.4),
    ("ACTUS", 0.862, 21.2, 0.0, 0.432, 5.4, 0.0),
    ("Oracle@1", 0.864, 21.4, 0.2, 0.440, 6.2, 0.9),
    ("Oracle@0", 0.854, 20.4, -0.7, 0.467, 8.9, 3.5),
    ("JEPA-TTT+ACTUS", 0.877, 22.7, 1.5, 0.431, 5.3, -0.1),
    ("AdaJEPA+ACTUS", 0.877, 22.7, 1.6, 0.434, 5.6, 0.3),
]
SR_F = {"push": 0.650, "cube": 0.378}
SR_A = {"push": 0.862, "cube": 0.432}
rows, bad = [], []
for name, ps, pdf, pda, cs, cdf, cda in T1:
    r = {"method": name}
    for tag, sr, df, da, fro, act in (("push", ps, pdf, pda, SR_F["push"], SR_A["push"]),
                                       ("cube", cs, cdf, cda, SR_F["cube"], SR_A["cube"])):
        d_f = round((sr - fro) * 100, 1)
        d_a = round((sr - act) * 100, 1)
        r[tag] = {"sr": sr, "dFrozen_calc": d_f, "dFrozen_table": df,
                  "dACTUS_calc": d_a, "dACTUS_table": da}
        for lab, calc, tab in (("dFrozen", d_f, df), ("dACTUS", d_a, da)):
            if tab is not None and abs(calc - tab) > 0.15:
                bad.append((name, tag, lab, calc, tab))
    rows.append(r)
res["table1_arithmetic"] = {"rows": rows, "mismatches_beyond_0.15pt": bad}

# 표 1 밖의 지표: 명목 성공률 감소(어댑테이션 방법, PushT: Frozen 0.873)
nom_push = {"AdaJEPA": 0.847, "OpTTA": 0.807, "JEPA-TTT": 0.940, "Sandwich-Res.": 0.847, "ALRA": 0.860}
res["nominal_drop_push_points"] = {k: round((0.873 - v) * 100, 1) for k, v in nom_push.items()}

# 손실 규모: 표 값에서 계산한 Snom - Frozen (텍스트: 22.3, 14.2)
res["shift_cost_points"] = {"push": round((0.873 - 0.650) * 100, 1), "cube": round((0.520 - 0.378) * 100, 1)}
res["actus_recovery_share"] = {"push": round(21.2 / 22.3, 3), "cube": round(5.4 / 14.2, 3)}

# ---------- 2) 계수 구간 ----------
def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(c - h, 3), round(c + h, 3)

def clopper(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return round(float(lo), 3), round(float(hi), 3)

counts = {
    "τ=1 보정 (PushT fleet)": (86, 90),
    "τ=1 보정 (Cube fleet)": (43, 60),
    "방향 판정 (PushT generator)": (22, 24),
    "방향 판정 (TwoRoom)": (22, 24),
    "방향 판정 (Cube generator)": (15, 16),
    "DINO 경보, 첫 블록 (전체)": (104, 113),
    "DINO 경보 전체": (113, 113),
    "DINO PushT prior 경보": (30, 30),
    "DINO Cube prior 경보": (15, 15),
}
res["binomial_intervals"] = {
    k: {"k": a, "n": b, "rate": round(a / b, 3), "wilson95": wilson(a, b), "clopper95": clopper(a, b)}
    for k, (a, b) in counts.items()
}
# 거짓 경보 0%: 대상 정상 fleet 수가 논문에 없어 상한을 계산할 수 없다. 구간 상한 형식만 기록.
res["false_alarm_note"] = "held-out nominal fleet 수 미공개: 0% 주장의 상한을 계산할 수 없음"

# ---------- 3) Prop. 2 정확 AUROC ----------
def auc_exact(lam, K):
    # P(X1 > X0), X1 ~ ncχ²(K, λ), X0 ~ χ²(K)
    f = lambda x: stats.chi2.pdf(x, K) * stats.ncx2.sf(x, K, lam)
    val, _ = __import__("scipy.integrate", fromlist=["quad"]).quad(f, 0, np.inf, limit=400)
    return val

def auc_approx(lam, K):
    return stats.norm.cdf(lam / (2 * math.sqrt(K + lam)))  # 식 (8)

zq = stats.norm.ppf(0.75)
lam075 = {}
for K in (4, 6):
    ex = brentq(lambda l: auc_exact(l, K) - 0.75, 0.01, 30)
    ap = brentq(lambda l: auc_approx(l, K) - 0.75, 0.01, 60)
    lam075[K] = {"exact_ncx2": round(ex, 2), "eq8_approx": round(ap, 2), "paper": {4: 3.6, 6: 4.2}[K]}
res["lambda_0.75"] = lam075
grid = [0.5, 1, 2, 4, 6, 8, 10, 15]
res["auc_curve"] = {f"K{K}": {str(l): {"exact": round(auc_exact(l, K), 4), "eq8": round(auc_approx(l, K), 4)}
                              for l in grid} for K in (4, 6)}

# ---------- 4) Thm. 2 분해 (Fig. 5b 값과 대조) ----------
def decomp(snom, o0, o1, s_hat):
    return {"Lbox": round((snom - o0) * 100, 1), "Llat": round((o0 - o1) * 100, 1),
            "Lest": round((o1 - s_hat) * 100, 1), "total": round((snom - s_hat) * 100, 1)}
res["thm2_decomposition"] = {
    "cube (Snom 0.520, Oracle@0 0.467, Oracle@1 0.440, ACTUS 0.432)": decomp(0.520, 0.467, 0.440, 0.432),
    "push (Snom 0.873, Oracle@0 0.854, Oracle@1 0.864, ACTUS 0.862)": decomp(0.873, 0.854, 0.864, 0.862),
}

# ---------- 5) 실로봇 상관 r = 0.84 ----------
def fisher_ci(r, n):
    z = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    return round(math.tanh(z - 1.96 * se), 3), round(math.tanh(z + 1.96 * se), 3)
res["arm_corr"] = {f"n={n}": fisher_ci(0.84, n) for n in (13, 26)}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k not in ("table1_arithmetic", "auc_curve")},
                 ensure_ascii=False, indent=2))
print("table1 mismatches:", bad)

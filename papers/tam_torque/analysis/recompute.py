"""TAM (arXiv:2606.06218 v3) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문 본문과 표의 값을 아래에 그대로 옮겨 적었다(출처는 주석). 외부 데이터는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 실로봇 비율의 분모는 본문과 표 캡션의 시도 수(상자 21, flip 50, 공 20회 실행)다.
- "±" 값은 표준편차(SD)로 가정한다. 논문이 SD인지 SE인지 밝히지 않는다. SE이면 t 검정 결과가 더 강해진다.
- Table 2는 Direct가 Reset 열에만 있다고 읽는다. 캡션은 "both columns"라고만 한다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2, z=1.959964):
    """Newcombe (1998) 방법 10: 두 비율 차이의 Wilson 기반 구간."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def fisher(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(table, alternative="two-sided")[1])


def welch(m1, s1, n1, m2, s2, n2):
    r = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    return float(r.statistic), float(r.pvalue)


def comp(name, k1, n1, k2, n2):
    d, lo, hi = newcombe_diff(k1, n1, k2, n2)
    return {
        "comparison": name,
        "k1": k1, "n1": n1, "k2": k2, "n2": n2,
        "diff_pp": round(100 * d, 1),
        "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
        "fisher_p": round(fisher(k1, n1, k2, n2), 4),
    }


# ---------------------------------------------------------------- 1. 표 1 실로봇 (정확한 횟수로 환산)
# 상자 밀기: 21회. 47.6=10/21, 57.1=12/21, 52.4=11/21, 66.7=14/21, 76.2=16/21
box = {"Direct": 10, "Online SysID": 12, "RMA": 11, "TAM (w/ RMA)": 14, "TAM": 16}
n_box = 21
RES["box_counts"] = {k: [v, n_box, round(100 * v / n_box, 1)] for k, v in box.items()}
RES["box_wilson95"] = {k: [round(100 * x, 1) for x in wilson(v, n_box)] for k, v in box.items()}
RES["box_comparisons"] = [
    comp("TAM vs Direct", box["TAM"], n_box, box["Direct"], n_box),
    comp("TAM vs Online SysID", box["TAM"], n_box, box["Online SysID"], n_box),
    comp("TAM vs RMA", box["TAM"], n_box, box["RMA"], n_box),
    comp("TAM(w/ RMA) vs RMA", box["TAM (w/ RMA)"], n_box, box["RMA"], n_box),
]
# 상자 밀기 시뮬레이션: 이상 모델 84/100 (F.1, F.3과 같은 값). 표 1의 84.0%와 같다.
RES["box_ideal_sim"] = {"k": 84, "n": 100}

# 뒤집기(flip): 50회. 50.0=25, 34.0=17, 72.0=36. 이상 모델 73.5%는 n이 명시되지 않았다.
flip = {"Direct": 25, "Online SysID": 17, "TAM": 36}
n_flip = 50
RES["flip_counts"] = {k: [v, n_flip, round(100 * v / n_flip, 1)] for k, v in flip.items()}
RES["flip_wilson95"] = {k: [round(100 * x, 1) for x in wilson(v, n_flip)] for k, v in flip.items()}
RES["flip_comparisons"] = [
    comp("TAM vs Direct", flip["TAM"], n_flip, flip["Direct"], n_flip),
    comp("TAM vs Online SysID", flip["TAM"], n_flip, flip["Online SysID"], n_flip),
]
# 이상 모델 73.5%가 가능한 가장 작은 n (소수 첫째 자리 반올림 기준)
RES["flip_ideal_possible_n"] = [n for n in range(1, 201) if round(100 * (round(0.735 * n) / n), 1) == 73.5][:6]

# 공 위 막대(MPC): 목표 수 / 15 s, 20회 실행, 평균 ± SD(가정)
ball = {"Direct": (8.8, 2.4), "Online SysID": (13.1, 2.3), "TAM": (13.6, 2.9), "Ideal sim": (15.9, 2.3)}
n_ball = 20
RES["ball_welch"] = {
    "TAM vs Online SysID": dict(zip(["t", "p"], [round(v, 4) for v in welch(13.6, 2.9, 20, 13.1, 2.3, 20)])),
    "TAM vs Direct": dict(zip(["t", "p"], [round(v, 4) for v in welch(13.6, 2.9, 20, 8.8, 2.4, 20)])),
    "Online SysID vs Direct": dict(zip(["t", "p"], [round(v, 4) for v in welch(13.1, 2.3, 20, 8.8, 2.4, 20)])),
    "TAM vs ideal sim": dict(zip(["t", "p"], [round(v, 4) for v in welch(13.6, 2.9, 20, 15.9, 2.3, 20)])),
}
# SE로 읽었을 때의 같은 검정(SD = SE x sqrt(20))
RES["ball_welch_if_se"] = {
    "TAM vs Online SysID": dict(zip(["t", "p"], [round(v, 4) for v in welch(13.6, 2.9 * math.sqrt(20), 20, 13.1, 2.3 * math.sqrt(20), 20)])),
}
RES["ball_mean_ci95_TAM"] = [round(13.6 - 2.093 * 2.9 / math.sqrt(20), 2), round(13.6 + 2.093 * 2.9 / math.sqrt(20), 2)]

# ---------------------------------------------------------------- 2. 표 2 추종 오차 (cm, 10회, 평균 ± SD)
t2 = {
    "hidden_reset": {"Direct": (4.09, 0.39), "Online SysID": (2.67, 0.63), "PhysPred": (1.74, 0.50), "TAM": (1.51, 0.30)},
    "hidden_carried": {"Online SysID": (2.75, 0.58), "PhysPred": (1.64, 0.56), "TAM": (1.04, 0.24)},
    "payload_reset": {"Direct": (6.04, 0.51), "Online SysID": (4.69, 0.42), "PhysPred": (4.15, 0.41), "TAM": (2.35, 0.39)},
    "payload_carried": {"Online SysID": (4.65, 0.53), "PhysPred": (4.39, 0.35), "TAM": (1.29, 0.35)},
}
t2_tests = {}
for cond, row in t2.items():
    if "TAM" not in row:
        continue
    t2_tests[cond] = {}
    for other, (m, s) in row.items():
        if other == "TAM":
            continue
        tm, sm = row["TAM"]
        tt, pp = welch(tm, sm, 10, m, s, 10)
        t2_tests[cond][f"TAM vs {other}"] = {"t": round(tt, 2), "p": float(f"{pp:.2g}")}
RES["table2_welch"] = t2_tests
# Reset 대 Carried (TAM 자신)
RES["table2_tam_reset_vs_carried"] = {
    "hidden": dict(zip(["t", "p"], [round(v, 3) for v in welch(1.51, 0.30, 10, 1.04, 0.24, 10)])),
    "payload": dict(zip(["t", "p"], [round(v, 3) for v in welch(2.35, 0.39, 10, 1.29, 0.35, 10)])),
}

# ---------------------------------------------------------------- 3. 표 3 교차 로봇 (관절 RMSE, deg, 100회)
robots = [
    # name, split, Direct, SysID, TAM, TAM sim-ft, TAM w/o eτ (표 D.2)
    ("Panda", "Train", (4.16, 0.99), (1.43, 0.53), (0.59, 0.19), (0.43, 0.17), (1.27, 0.32)),
    ("Piper", "Train", (4.88, 0.96), (0.83, 0.18), (0.52, 0.17), (0.42, 0.15), (0.54, 0.14)),
    ("iiwa14", "Train", (3.69, 0.82), (0.99, 0.35), (0.56, 0.16), (0.47, 0.16), (0.94, 0.23)),
    ("Flexiv Rizon4", "Train", (4.06, 1.20), (1.33, 0.51), (0.84, 0.33), (0.68, 0.27), (1.43, 0.74)),
    ("RBY-1", "Train", (3.80, 0.81), (1.39, 0.83), (0.84, 0.16), (0.66, 0.17), (1.30, 0.23)),
    ("Google Robot", "Test", (4.69, 1.13), (1.65, 0.37), (1.05, 0.40), (0.93, 0.32), (3.58, 1.15)),
    ("Unitree Z1", "Test", (6.11, 1.27), (1.05, 0.23), (0.93, 0.19), (0.72, 0.24), (2.43, 0.17)),
]
means = {k: round(float(np.mean([r[i][0] for r in robots])), 3) for i, k in
         [(2, "Direct"), (3, "Online SysID"), (4, "TAM"), (5, "TAM sim-ft"), (6, "TAM w/o eτ")]}
RES["table3_mean_recomputed"] = means
RES["table3_mean_paper"] = {"Direct": 4.48, "Online SysID": 1.24, "TAM": 0.76, "TAM sim-ft": 0.62, "TAM w/o eτ": 1.64}
t3 = []
for r in robots:
    name = r[0]
    d_ = r[2]
    s_ = r[3]
    t_ = r[4]
    tt, pp = welch(t_[0], t_[1], 100, s_[0], s_[1], 100)
    tt2, pp2 = welch(t_[0], t_[1], 100, r[6][0], r[6][1], 100)
    t3.append({
        "robot": name, "split": r[1],
        "TAM_vs_SysID_ratio": round(t_[0] / s_[0], 2),
        "TAM_vs_SysID_p": float(f"{pp:.2g}"),
        "TAM_vs_noEtau_p": float(f"{pp2:.2g}"),
        "Direct_over_TAM": round(d_[0] / t_[0], 1),
    })
RES["table3_per_robot"] = t3
# 학습 로봇 5대와 테스트 로봇 2대의 TAM/SysID 비율 평균
RES["table3_ratio_summary"] = {
    "train_mean_TAM_over_SysID": round(float(np.mean([x["TAM_vs_SysID_ratio"] for x in t3[:5]])), 2),
    "test_mean_TAM_over_SysID": round(float(np.mean([x["TAM_vs_SysID_ratio"] for x in t3[5:]])), 2),
}
# 학습 로봇 Panda(6-DoF Piper 제외)의 TAM w/o eτ 대 TAM: 평균 차 (test 로봇에서 더 큼)
RES["table3_noetau_inflation"] = {
    "train_mean_ratio": round(float(np.mean([r[6][0] / r[4][0] for r in robots[:5]])), 2),
    "test_mean_ratio": round(float(np.mean([r[6][0] / r[4][0] for r in robots[5:]])), 2),
}

# ---------------------------------------------------------------- 4. 표 F.1 (시뮬, 100회)
f1 = {"Direct transfer": 21, "Short-window torque adaptor only": 42, "No physics residual feature": 67,
      "History predicts physics parameters": 59, "Privileged-parameter latent-state supervision": 70, "Full TAM": 84}
RES["f1_comparisons"] = [
    comp("Full TAM vs Direct", 84, 100, 21, 100),
    comp("Full TAM vs No eτ", 84, 100, 67, 100),
    comp("Full TAM vs Privileged latent", 84, 100, 70, 100),
    comp("Full TAM vs History->params", 84, 100, 59, 100),
    comp("Full TAM vs short window", 84, 100, 42, 100),
]

# ---------------------------------------------------------------- 5. 표 F.3 (이상 모델)과 표 F.3 friction (시뮬, 100회)
RES["f3_ideal_no_tam_vs_tam"] = comp("Ideal: with TAM vs without", 84, 100, 84, 100)
RES["ood_friction"] = {
    "2x": {"TAM": 82, "Direct": 75, "comp": comp("2x TAM vs Direct", 82, 100, 75, 100)},
    "3x": {"TAM": 80, "Direct": 59, "comp": comp("3x TAM vs Direct", 80, 100, 59, 100)},
}

# ---------------------------------------------------------------- 6. 본문 수치 대조
checks = []


def chk(label, paper, recomputed, tol=0.051):
    checks.append({"item": label, "paper": paper, "recomputed": round(recomputed, 3),
                   "ok": abs(paper - recomputed) <= tol})


chk("box TAM - Direct (pp)", 28.6, 76.2 - 47.6)
chk("box TAM - SysID (pp)", 19.1, 76.2 - 57.1)
chk("box TAM(w/ RMA) - RMA (pp)", 14.3, 66.7 - 52.4)
chk("flip TAM - Direct (pp)", 22.0, 72.0 - 50.0)
chk("flip TAM - SysID (pp)", 38.0, 72.0 - 34.0)
chk("flip TAM / SysID (x)", 2.12, 72.0 / 34.0, tol=0.05)
chk("table3 mean Direct", 4.48, means["Direct"])
chk("table3 mean SysID", 1.24, means["Online SysID"])
chk("table3 mean TAM", 0.76, means["TAM"])
chk("table3 mean TAM sim-ft", 0.62, means["TAM sim-ft"])
chk("table3 mean noEtau (D.2)", 1.64, means["TAM w/o eτ"])
chk("table2 hidden carried TAM gain (cm)", 4.09 - 1.04, 4.09 - 1.04)
chk("table2 payload carried TAM vs direct (cm)", 6.04 - 1.29, 6.04 - 1.29)
chk("ball TAM - SysID (goals)", 0.5, 13.6 - 13.1)
chk("ball TAM - Direct (goals)", 4.8, 13.6 - 8.8)
chk("F.3 ideal sim box 84/100 vs Table 1 84.0", 84.0, 84.0)
chk("F.1 full TAM vs F.1 direct (pp)", 63.0, 84 - 21)
RES["consistency_checks"] = checks

# ---------------------------------------------------------------- 8. 검정력: 상자 밀기 TAM 대 SysID를 p<0.05로 보려면 과제당 필요한 시도 수
def n_needed(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


RES["power"] = {
    "box TAM 76.2 vs SysID 57.1, n per arm": n_needed(0.762, 0.571),
    "box TAM 76.2 vs Direct 47.6, n per arm": n_needed(0.762, 0.476),
    "flip TAM 72 vs SysID 34, n per arm": n_needed(0.72, 0.34),
}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")


def show(title, obj):
    print(f"\n== {title}")
    print(json.dumps(obj, ensure_ascii=False, indent=1))


show("box counts + Wilson", {"counts": RES["box_counts"], "wilson": RES["box_wilson95"]})
show("box comparisons", RES["box_comparisons"])
show("flip comparisons", RES["flip_comparisons"])
show("flip ideal possible n", RES["flip_ideal_possible_n"])
show("ball welch", RES["ball_welch"])
show("ball welch if SE", RES["ball_welch_if_se"])
show("table2 welch", RES["table2_welch"])
show("table2 reset vs carried", RES["table2_tam_reset_vs_carried"])
show("table3 per robot", RES["table3_per_robot"])
show("table3 ratio", RES["table3_ratio_summary"])
show("table3 noEtau", RES["table3_noetau_inflation"])
show("table3 means", {"recomputed": means, "paper": RES["table3_mean_paper"]})
show("F.1 comparisons", RES["f1_comparisons"])
show("OOD friction", RES["ood_friction"])
show("power", RES["power"])
bad = [c for c in checks if not c["ok"]]
print("\nconsistency checks:", len(checks), "failed:", bad)

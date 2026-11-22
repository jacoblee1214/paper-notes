"""Phantom (arXiv:2503.00779 v2) 논문 표 수치를 다시 계산하고 비율 검정을 한다.

입력: 논문 Table 1, 2, 3, 5, 6, 10의 값을 아래에 그대로 옮겨 적었다(출처는 주석).
      모든 성공률은 시도 수 n으로 나눈 비율이며, n은 논문 캡션의 "25 rollouts" 또는 Table 6의 "25 rollouts"다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 성공률 0.xx는 k/25 로 환산한다. 0.75 와 같이 25의 배수가 아닌 값은 k/25 가 될 수 없으므로 V1로 따로 표시한다.
- Fisher 정확검정(양측)과 Wilson 95% 구간을 쓴다. 다중비교 보정은 하지 않는다.
"""
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N = 25  # 논문 캡션: "25 rollouts per evaluation"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")[1])


def comp(label, k1, n1, k2, n2, src):
    d = k1 / n1 - k2 / n2
    return {"비교": label, "k1/n1": f"{k1}/{n1}", "k2/n2": f"{k2}/{n2}",
            "차이_pp": round(100 * d, 1), "Fisher_p": round(fisher(k1, n1, k2, n2), 4), "출처": src}


def ci(k, n):
    lo, hi = wilson(k, n)
    return {"k/n": f"{k}/{n}", "p": round(k / n, 3), "wilson95": [round(lo, 3), round(hi, 3)]}


def min_detectable_gap(n, base, power=0.8, alpha=0.05):
    """두 군 n명씩, 기준 성공률 base에서 검출 가능한 최소 차이(근사, 대략적 탐색)."""
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    for d in [i / 1000 for i in range(1, 1000)]:
        p1, p2 = base, min(base + d, 0.999)
        pbar = (p1 + p2) / 2
        se = math.sqrt(2 * pbar * (1 - pbar) / n)
        if abs(p2 - p1) / se >= z_a + z_b:
            return round(100 * d, 1)
    return None


RES = {"n": N, "가정": "모든 비율은 k/25. 0.75는 k/25가 아니므로 V1로 표시"}

# Table 1 (In-distribution, 논문 p.6). 열: Pick/Place Book, Stack Cups, Tie Rope, Rotate Box, Grasp Brush, Sweep>0, Sweep>2, Sweep>4
T1_cols = ["Pick/Place Book", "Stack Cups", "Tie Rope", "Rotate Box", "Grasp Brush", "Sweep>0", "Sweep>2", "Sweep>4"]
T1 = {
    "Hand Inpaint": [0.92, 0.72, 0.64, 0.72, 0.88, 0.80, 0.72, 0.40],
    "Hand Mask":    [0.92, 0.52, 0.60, 0.76, 0.75, 0.75, 0.72, 0.68],
}
RES["T1_k25"] = {m: [round(v * N) for v in vals] for m, vals in T1.items()}
RES["T1_비정수"] = [(c, m, v) for m, vals in T1.items() for c, v in zip(T1_cols, vals) if abs(v * N - round(v * N)) > 1e-6]
RES["T1_inpaint_vs_mask"] = []
for i, c in enumerate(T1_cols):
    a, b = round(T1["Hand Inpaint"][i] * N), round(T1["Hand Mask"][i] * N)
    if (c, "Hand Mask", T1["Hand Mask"][i]) in RES["T1_비정수"]:
        # 0.75 -> 18/25 또는 19/25 두 경우 모두 계산 (V1 민감도)
        RES["T1_inpaint_vs_mask"].append(
            {**comp(f"{c}: Inpaint vs Mask (Mask=18/25)", a, N, 18, N, "Table 1"),
             "비고": "V1: 0.75는 25회에서 불가능. 18/25 가정"})
        RES["T1_inpaint_vs_mask"].append(
            {**comp(f"{c}: Inpaint vs Mask (Mask=19/25)", a, N, 19, N, "Table 1"),
             "비고": "V1: 19/25 가정"})
    else:
        RES["T1_inpaint_vs_mask"].append(comp(f"{c}: Inpaint vs Mask", a, N, b, N, "Table 1"))

# Table 2 (OOD, p.7). 열: Outdoor Lawn, Indoor Lounge, Indoor Lounge + OOD surface
RES["T2_k25"] = {"Hand Inpaint": [18, 21, 16], "Hand Mask": [13, 19, 17]}
RES["T2_inpaint_vs_mask"] = [
    comp("Outdoor Lawn: Inpaint 18 vs Mask 13", 18, N, 13, N, "Table 2"),
    comp("Indoor Lounge: Inpaint 21 vs Mask 19", 21, N, 19, N, "Table 2"),
    comp("OOD surface: Inpaint 16 vs Mask 17", 16, N, 17, N, "Table 2"),
]
RES["T2_inpaint_surface_drop_pp"] = round(100 * (21 - 16) / N, 1)  # 본문 "20% 하락" 대조

# Table 3 (inpainting, p.8), Indoor Lounge
RES["T3_k25"] = {"E2FGVI": 21, "OpenCV": 19, "Mask only": 15}
RES["T3_pairs"] = [
    comp("E2FGVI 21 vs OpenCV 19", 21, N, 19, N, "Table 3"),
    comp("E2FGVI 21 vs Mask only 15", 21, N, 15, N, "Table 3"),
    comp("OpenCV 19 vs Mask only 15", 19, N, 15, N, "Table 3"),
]

# Table 5 (co-training, p.17), Kinova sweep
RES["T5_pairs"] = [
    comp("Robot-only 22 (in-dist) vs Robot-only 0 (OOD)", 22, N, 0, N, "Table 5"),
    comp("Robot+Human 20 (OOD) vs Robot-only 0 (OOD)", 20, N, 0, N, "Table 5"),
]

# Table 6 (robot vs human, p.17-18), Kinova sweep, 25 rollouts
T6 = {50: (13, 11), 100: (22, 16)}  # (robot k, human k) 본문 p=0.778, 0.095 대조
RES["T6_pairs"] = [
    {**comp("50 demos: Robot 13 vs Human 11", 13, N, 11, N, "Table 6"), "논문_p": 0.778},
    {**comp("100 demos: Robot 22 vs Human 16", 22, N, 16, N, "Table 6"), "논문_p": 0.095},
    comp("Human 300 (21) vs Robot 100 (22)", 21, N, 22, N, "Table 6 (300 vs 100 robot, 참고)"),
    comp("Human 100 (16) vs Human 300 (21)", 16, N, 21, N, "Table 6 (human 스케일링, 참고)"),
]
RES["T6_mde_gap_pp"] = {
    "n25_per_arm_base50pct": min_detectable_gap(25, 0.50),
}

# Table 10 (augmentation, p.21)
RES["T10_pairs"] = [
    comp("Pick/Place Book: Mask 23 vs Mask no-aug 22", 23, N, 22, N, "Table 10"),
    comp("Stack Cups: Mask 13 vs Mask no-aug 10", 13, N, 10, N, "Table 10"),
]

# Table 1 Red Line(0/25) vs Hand Inpaint 는 자명하므로 생략. 처음 5열 평균(4과제 + Grasp Brush)
inp = [0.92, 0.72, 0.64, 0.72, 0.88]
msk = [0.92, 0.52, 0.60, 0.76, 0.75]
RES["T1_mean_first5cols_inpaint_vs_mask"] = {"inpaint_mean": round(sum(inp) / 5, 3), "mask_mean": round(sum(msk) / 5, 3)}

# 본문 주장 대조: Red Line 0/25 대 Inpaint, OOD 표면 하락 20 %p (21 대 16)
RES["C_extra_pairs"] = [
    comp("Pick/Place Book: Inpaint 23 vs Red Line 0", 23, N, 0, N, "Table 1"),
    comp("OOD: Indoor Lounge 21 vs OOD surface 16 (Inpaint)", 21, N, 16, N, "Table 2 (본문 '20 %p 하락')"),
    comp("Sweep>4: Inpaint 10 vs Mask 17", 10, N, 17, N, "Table 1 (v2 본문에 언급 없음)"),
]

# 본문 대조용 Wilson CI (대표 값)
RES["CI_examples"] = {"Inpaint Stack Cups": ci(18, N), "Mask Stack Cups": ci(13, N),
                      "Robot 50 demos": ci(13, N), "Human 50 demos": ci(11, N),
                      "Human 300 demos": ci(21, N)}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
if __name__ == "__main__":
    for k, v in RES.items():
        print(k, "=>", v if not isinstance(v, list) else "")
        if isinstance(v, list):
            for row in v:
                print("   ", row)

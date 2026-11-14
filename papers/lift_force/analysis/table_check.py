"""논문 본문 수치를 다시 계산한다 (CPU, 약 1분).

입력: 논문 본문(Section 5.2–5.5)과 Figure 1에 적힌 값. 원문 PDF의 그래프 값은 쓰지 않는다.
출력: analysis/table_check_results.json

가정(명시):
- Hanoi는 이진 성공이며 체크포인트당 30회(3 x 10)로 본다. v1 본문은 10회라고 했다(04 C3). 두 경우를 모두 적는다.
- towel·book은 단계 점수(0, 0.25, 0.5, 0.75, 1 또는 0, 0.5, 1)라서 표준편차를 알 수 없다. 그래서 SD를 0.2–0.5로 두고 민감도를 본다.
- 학습 시드는 논문에 없다. 아래 검정은 롤아웃 표본만 반영한다.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "table_check_results.json"
rng = np.random.default_rng(0)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fisher(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(table, alternative="two-sided")[1])


res = {"assumptions": {"binary_n": 30, "binary_n_v1": 10, "sd_grid": [0.2, 0.3, 0.4, 0.5]}}

# 1. Hanoi 이진 성공 (본문 5.2, 5.3). 값은 본문에 적힌 값을 n=30 횟수로 환산
hanoi = {
    "LIFT_peak_0.6": (18, 30),
    "LIFT_final_0.6_at_2.0K": (18, 30),
    "OnlineDAgger_peak_0.3": (9, 30),
    "OnlineDAgger_final_0.2": (6, 30),
    "LIFT_wo_RFI_peak_0.5": (15, 30),
    "LIFT_wo_RFI_final_0.1": (3, 30),
}
res["hanoi_counts"] = {}
for name, (k, n) in hanoi.items():
    lo, hi = wilson(k, n)
    lo10, hi10 = wilson(round(k / 3), 10)  # v1 가정(10회)으로 환산한 횟수는 정수가 아닐 수 있어 참고용
    res["hanoi_counts"][name] = {"k": k, "n": n, "rate": round(k / n, 3), "wilson95": [round(lo, 3), round(hi, 3)]}

pairs = [
    ("LIFT_peak_0.6", "OnlineDAgger_peak_0.3"),
    ("LIFT_final_0.6_at_2.0K", "OnlineDAgger_final_0.2"),
    ("LIFT_peak_0.6", "LIFT_wo_RFI_peak_0.5"),
    ("LIFT_final_0.6_at_2.0K", "LIFT_wo_RFI_final_0.1"),
]
res["hanoi_fisher"] = {}
for a, b in pairs:
    ka, na = hanoi[a]
    kb, nb = hanoi[b]
    res["hanoi_fisher"][f"{a} vs {b}"] = {"p_two_sided": round(fisher(ka, na, kb, nb), 4),
                                          "diff": round((ka - kb) / na, 3)}

# 2. 체크포인트 최대값(peak) 선택 편향: 두 정책이 같은 성공률 p0를 가질 때 최대값 차이가 0.3 이상 날 확률
#    체크포인트 K개, 각 30회, 몬테카를로 200,000회
def peak_null(p0, K, diff_counts=9, n=30, reps=200_000):
    x = rng.binomial(n, p0, size=(reps, K)).max(axis=1)
    y = rng.binomial(n, p0, size=(reps, K)).max(axis=1)
    return float(np.mean((x - y) >= diff_counts))


res["hanoi_peak_selection_null"] = {}
for p0 in (0.2, 0.3):
    for K in (4, 6):
        res["hanoi_peak_selection_null"][f"p0={p0},K={K}"] = round(peak_null(p0, K), 4)

# 3. 단계 점수 평균의 신뢰구간 폭 (n=30, t 분포)
t30 = stats.t.ppf(0.975, 29)
res["half_width_95_n30"] = {str(s): round(t30 * s / np.sqrt(30), 3) for s in (0.2, 0.3, 0.4, 0.5)}

# 4. towel·book 피크 차이의 z와 p (SD 민감도)
cases = {
    "towel_LIFT_0.825_vs_DAgger_0.725": (0.825, 0.725),
    "towel_LIFT_0.65_at_2.3K_vs_DAgger_0.725_at_3.1K": (0.65, 0.725),
    "book_LIFT_0.6_vs_DAgger_0.4": (0.6, 0.4),
    "hanoi_LIFT_0.6_vs_DAgger_0.3_means": (0.6, 0.3),
}
res["mean_diff_sensitivity"] = {}
for name, (m1, m2) in cases.items():
    row = {}
    for s in (0.2, 0.3, 0.4, 0.5):
        se = np.sqrt(2) * s / np.sqrt(30)
        z = (m1 - m2) / se
        row[f"SD={s}"] = {"z": round(float(z), 2), "p_two_sided": round(float(2 * stats.norm.sf(abs(z))), 4)}
    res["mean_diff_sensitivity"][name] = row

# 5. 검정력: 이진 n=30 대 n=30, Fisher 정확검정 (몬테카를로 4,000회)
def power_binary(p1, p2, n=30, reps=4000):
    hit = 0
    for _ in range(reps):
        k1 = rng.binomial(n, p1)
        k2 = rng.binomial(n, p2)
        if fisher(k1, n, k2, n) < 0.05:
            hit += 1
    return hit / reps


res["power_binary_n30"] = {
    "p1=0.6_vs_p2=0.3": round(power_binary(0.6, 0.3), 3),
    "p1=0.6_vs_p2=0.2": round(power_binary(0.6, 0.2), 3),
}

# 6. Figure 1 의 'average peak score' 와 본문 peak 값 비교 (정의는 논문에 없음)
fig1 = {"towel": (0.733, 0.842, 0.725, 0.825), "book": (0.367, 0.583, 0.4, 0.6), "hanoi": (0.267, 0.567, 0.3, 0.6)}
res["figure1_vs_text"] = {k: {"fig_vision": v[0], "fig_lift": v[1], "text_vision": v[2], "text_lift": v[3],
                              "diff_vision": round(v[0] - v[2], 3), "diff_lift": round(v[1] - v[3], 3)}
                          for k, v in fig1.items()}
# 3 그룹(각 10회)의 평균이라면 값은 1/30의 배수가 된다. 이진 결과에서 가능한 조합 검사
res["figure1_thirds_check"] = {k: {"x30": round(v * 30, 2)} for k, v in
                               {"hanoi_vision": 0.267, "hanoi_lift": 0.567, "book_vision": 0.367}.items()}


# 7. Figure 6 (일반화, 막대 위 숫자 라벨을 그대로 옮김). 원본 대비 차이의 SD 민감도와 이진 Fisher
gen = {
    "towel": {"Original": 0.8, "Another desktop": 0.825, "Blue towel": 0.925, "Purple light": 0.85},
    "book": {"Original": 0.5, "Another book": 0.35, "Another desktop": 0.1, "Purple light": 0.45},
    "hanoi": {"Original": 0.6, "Another desktop": 0.3, "Purple light": 0.5},
}
res["generalization_fig6"] = {}
for task, d in gen.items():
    base = d["Original"]
    rows = {}
    for cond, m in d.items():
        if cond == "Original":
            continue
        row = {"diff": round(m - base, 3)}
        for sd in (0.3, 0.5):
            se = np.sqrt(2) * sd / np.sqrt(30)
            z = (m - base) / se
            row[f"z_SD={sd}"] = round(float(z), 2)
            row[f"p_SD={sd}"] = round(float(2 * stats.norm.sf(abs(z))), 4)
        rows[cond] = row
    res["generalization_fig6"][task] = rows
res["generalization_fig6"]["hanoi_binary_fisher_original_18_of_30_vs_desktop_9_of_30"] = round(fisher(18, 30, 9, 30), 4)

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
print(json.dumps(res, ensure_ascii=False, indent=2))

"""TactSpace (arXiv:2606.18959v1) 표 I–IV 수치 재계산과 주장 대조. numpy 1스레드, 1초 미만.

실행: python3 analysis/table_check.py  -> analysis/table_check_results.json
표 값은 논문 PDF(p.6–7)를 렌더링해 눈으로 읽은 값이다. 원자료(샘플별 오차)는 공개되지 않았다.
"""
import itertools
import json
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
from math import erf, sqrt

OUT = Path(__file__).with_name("table_check_results.json")

# (Acc%, Pen um, Force mN) x (ID, OOD-Size, OOD-Shape); Pen/Force의 20-80 백분위 범위는 생략
# Table III: 행 이름은 (정렬 데이터, 복원 손실, 과제 데이터)
T3 = {
    "R1 정렬 Mc,Mp,Ms · 복원 없음 · 과제 Mp,Ms": [(56.35, 45, 60), (42.58, 79, 61), (41.70, 7, 65)],
    "R2 정렬 Mc,Mp · 복원 Mc,Mp · 과제 Mp": [(55.49, 58, 78), (47.44, 68, 75), (42.80, 13, 74)],
    "R3 정렬 Mc,Ms · 복원 Mc,Ms · 과제 Ms": [(54.94, 48, 63), (45.05, 66, 63), (29.10, 8, 51)],
    "R4 정렬 Mc,Mp,Ms · 복원 전부 · 과제 Mp,Ms (TactSpace)": [(61.09, 26, 65), (51.84, 39, 59), (45.25, 13, 63)],
    "R5 정렬 A∪D(OOD 포함) · 복원 전부 · 과제 Mp,Ms": [(61.91, 25, 62), (70.66, 49, 67), (56.05, 8, 72)],
    "R6 실측 지도학습 Mc (상한)": [(76.10, 18, 20), (83.03, 16, 18), (69.75, 3, 19)],
}
# Table II
T2 = {
    "No alignment": [(36.72, 37, 89), (39.69, 74, 86), (26.15, 10, 77)],
    "MSE": [(52.50, 35, 68), (49.81, 46, 64), (44.15, 14, 67)],
    "InfoNCE": [(61.09, 26, 65), (51.84, 39, 59), (45.25, 13, 63)],
}
# Table IV (RBS 스케일링)
T4 = {
    "RBS(1x)": [(58.78, 34, 76), (49.41, 80, 77), (38.90, 10, 72)],
    "RBS(5x)": [(65.32, 38, 75), (58.84, 48, 79), (40.50, 8, 73)],
    "RBS(15x)": [(70.81, 34, 78), (64.33, 47, 81), (46.80, 8, 75)],
    "RBS(30x)": [(70.85, 33, 77), (63.37, 49, 80), (45.45, 9, 73)],
    "RBS(30x)+FEM": [(69.00, 28, 59), (62.99, 37, 59), (41.05, 8, 49)],
}
# Table I (평가 모달리티별, 같은 네트워크)
T1 = {
    "RBS Mp (평가)": [(64.30, 16, 65), (71.75, 18, 67), (44.50, 20, 85)],
    "FEM Ms (평가)": [(55.96, 12, 43), (65.00, 14, 51), (25.00, 6, 27)],
    "Real Mc (평가)": [(61.09, 26, 65), (51.84, 39, 59), (45.25, 13, 63)],
}
SPLITS = ["ID", "OOD-Size", "OOD-Shape"]
METRICS = ["Acc", "Pen", "Force"]


def pct_reduction(base, new):
    return 100.0 * (base - new) / base


def phi(x):  # 표준정규 누적
    return 0.5 * (1 + erf(x / sqrt(2)))


def two_sided_p(z):
    return 2 * (1 - phi(abs(z)))


res = {"headline_check": {}, "pair_search": {}, "significance": {}, "claims": {}}

# 1) 헤드라인 수치(16.7%, 45.8%)가 어느 행 쌍에서 나오는지 전수 탐색
rows = {k: v for k, v in T3.items()}
hits = []
for (ka, va), (kb, vb) in itertools.permutations(rows.items(), 2):
    for si, split in enumerate(SPLITS):
        for mi, met in enumerate(METRICS):
            if met == "Acc":
                continue
            base, new = va[si][mi + 1 if False else (1 if met == "Pen" else 2)], vb[si][(1 if met == "Pen" else 2)]
            r = pct_reduction(base, new)
            if abs(r - 16.7) < 0.05 or abs(r - 45.8) < 0.05:
                hits.append({"base(행 A)": ka[:2], "new(행 B)": kb[:2], "split": split, "metric": met,
                             "base": base, "new": new, "reduction_%": round(r, 2)})
res["pair_search"] = hits
res["headline_check"] = {
    "force 16.7%": "(78-65)/78 = %.2f%%: R2(정렬 Mc,Mp; 과제 Mp) 대 R4 (ID 힘)" % pct_reduction(78, 65),
    "pen 45.8%": "(48-26)/48 = %.2f%%: R3(정렬 Mc,Ms; 과제 Ms) 대 R4 (ID 깊이)" % pct_reduction(48, 26),
    "no-alignment 대비 ID 깊이": "(37-26)/37 = %.1f%%" % pct_reduction(37, 26),
    "no-alignment 대비 ID 힘": "(89-65)/89 = %.1f%%" % pct_reduction(89, 65),
    "R1(복원 없음) 대비 ID 힘": "(60-65)/60 = %.1f%% (음수면 R4가 나쁨)" % pct_reduction(60, 65),
}

# 2) 주장 검사
res["claims"] = {
    "T2 'InfoNCE가 모든 지표·분할에서 우위'": {
        "반례": "OOD-Shape Pen: No alignment 10 µm < InfoNCE 13 µm",
        "값": T2["No alignment"][2][1] < T2["InfoNCE"][2][1],
    },
    "T1 '깊이 기하는 RBS가 강함(Pen)'": {
        "값": {split: {"RBS Pen": T1["RBS Mp (평가)"][i][1], "FEM Pen": T1["FEM Ms (평가)"][i][1]} for i, split in enumerate(SPLITS)},
        "해석": "세 분할 모두 FEM의 깊이 오차가 더 작다. '기하는 RBS' 주장은 Acc에만 맞는다",
    },
    "T1 '실측 잠재가 시뮬과 비슷(comparable)'": {
        "값": {split: {"RBS Acc/Pen/Force": T1["RBS Mp (평가)"][i], "Real Acc/Pen/Force": T1["Real Mc (평가)"][i]} for i, split in enumerate(SPLITS)},
        "해석": "OOD-Size Acc 51.8 대 71.8, 깊이 39 대 18 µm로 차이가 크다. ID Pen도 26 대 16 µm",
    },
    "T4 '규모 확대로 분류·깊이 개선'": {
        "깊이 ID (1x→30x)": [v[0][1] for k, v in T4.items() if k.startswith("RBS(") and "FEM" not in k],
        "Acc ID (1x→30x)": [v[0][0] for k, v in T4.items() if k.startswith("RBS(") and "FEM" not in k],
        "해석": "분류 Acc는 58.8→70.9로 오르지만 깊이는 34→33 µm로 변하지 않는다. 텍스트의 '깊이 추정 개선'과 어긋난다",
    },
    "T3 'full 모델이 최고 또는 차선'": {
        "R4가 최고 아닌 셀(R1–R4, 3분할×3지표)": None,
    },
}
# T3에서 R4가 최고 또는 차선이 아닌 셀 수 세기 (낮을수록 좋은 Pen/Force, 높을수록 좋은 Acc)
latent_rows = [k for k in T3 if not k.startswith("R6") and not k.startswith("R5")]
bad = []
for si, split in enumerate(SPLITS):
    for mi, met in enumerate(METRICS):
        vals = {k[:2]: T3[k][si][mi] for k in latent_rows}
        r4 = vals["R4"]
        better = [k for k, v in vals.items() if (v > r4 if met == "Acc" else v < r4)]
        ranks = sorted(vals.values(), reverse=(met == "Acc"))
        r4_rank = ranks.index(r4) + 1
        if r4_rank > 2:
            bad.append({"split": split, "metric": met, "R4": r4, "R4 순위(1=최고)": r4_rank, "앞선 행": better})
res["claims"]["T3 'full 모델이 최고 또는 차선'"]["R4가 차선 밖인 셀"] = bad
res["claims"]["T3 'full 모델이 최고 또는 차선'"]["셀 수"] = len(bad)

# 3) 유의성: 샘플 단위(과대 추정) 대 궤적 단위(보수적). 설계효과 DEFF = 1 + (m-1)ρ
N_TRAJ = {"ID": 0.25 * 231, "OOD-Size": 0.25 * 210, "OOD-Shape": 0.25 * 168}  # 테스트 궤적 수 (25% 분할)
M_PER_TRAJ = 32942 / 840  # 궤적당 샘플 수 (Mc, Mp 기준)
sig = {}


def acc_z(p1, p2, n):
    p1, p2 = p1 / 100, p2 / 100
    pbar = (p1 + p2) / 2
    se = sqrt(2 * pbar * (1 - pbar) / n)
    return (p1 - p2) / se


for rho in [0.0, 0.05, 0.2]:
    deff = 1 + (M_PER_TRAJ - 1) * rho
    for split in SPLITS:
        n_s = N_TRAJ[split] * M_PER_TRAJ
        n_eff = n_s / deff
        z = acc_z(T2["InfoNCE"][SPLITS.index(split)][0], T2["MSE"][SPLITS.index(split)][0], n_eff)
        sig[f"Acc InfoNCE vs MSE, {split}, rho={rho}"] = {"n_eff": round(n_eff, 1), "z": round(z, 2), "p": round(two_sided_p(z), 4)}
    # 같은 방식으로 RBS 1x vs 15x (ID Acc)
    n_eff = N_TRAJ["ID"] * M_PER_TRAJ / deff
    z = acc_z(T4["RBS(15x)"][0][0], T4["RBS(1x)"][0][0], n_eff)
    sig[f"Acc RBS 15x vs 1x, ID, rho={rho}"] = {"n_eff": round(n_eff, 1), "z": round(z, 2), "p": round(two_sided_p(z), 4)}
    # 논문 헤드라인 힘 비교는 Acc가 아니라 MAE이므로 건너뛴다
res["significance"] = sig
res["significance_note"] = ("샘플 단위 검정은 궤적 내 상관을 무시하므로 과대평가다. 궤적당 샘플 수 m≈%.1f, "
                            "ρ는 알 수 없으므로 0–0.2로 훑었다. 시드 반복 결과가 없어 학습 분산은 포함되지 않는다." % M_PER_TRAJ)

# 4) 로그 스케일 MAE 효과 크기: 20–80 백분위 범위에서 로그 표준편차 추정
def sigma_log(p20, p80):
    return (math.log(p80) - math.log(p20)) / (2 * 0.8416)

eff = {}
pairs_pen = [("InfoNCE ID Pen 26(9-59)", 26, (9, 59), "MSE ID Pen 35(21-66)", 35, (21, 66)),
             ("InfoNCE ID Pen 26(9-59)", 26, (9, 59), "No align ID Pen 37(12-82)", 37, (12, 82)),
             ("InfoNCE ID Force 65(37-118)", 65, (37, 118), "MSE ID Force 68(38-122)", 68, (38, 122))]
for a, va, ra, b, vb, rb in pairs_pen:
    s = 0.5 * (sigma_log(*ra) + sigma_log(*rb))
    d = (math.log(va) - math.log(vb)) / s
    eff[f"{a} vs {b}"] = {"d_log": round(d, 3)}
res["effect_size_log_mae"] = eff

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str))
print(json.dumps({"headline": res["headline_check"], "pair_hits": [h for h in hits][:12],
                  "bad_cells": bad, "sig_sample": {k: v for k, v in sig.items() if "rho=0.0" in k or "rho=0.2" in k},
                  "eff": eff}, ensure_ascii=False, indent=1))

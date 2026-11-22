"""The Imitator Game (arXiv:2608.22301 v1) 보고 수치 재계산과 일관성 검사.

논문 표 3, 4, 5, 9, 10, 11과 본문 수치를 옮겨 와 다시 계산한다(check 번호는 V1부터).
프로젝트 페이지가 공개한 셀 단위 시뮬레이션 결과와 Appendix C 수치(data/ 폴더)가 있으면 그것으로 표 3을 처음부터 다시 만든다.
data/ 가 없으면 그 부분은 건너뛴다. 이 스크립트는 무엇도 학습하거나 내려받지 않는다(수 초).

data/ 준비 (프로젝트 페이지의 공개 정적 파일, 실행하지 않고 텍스트로만 읽는다):
  mkdir -p data
  curl -sL https://imitator-game.github.io/assets/js/figdata.js -o data/site_figdata.js
  curl -sL https://imitator-game.github.io/assets/js/full-task-suite-data.js -o data/site_full_task_suite.js
  curl -sL https://imitator-game.github.io/assets/leaderboard/all_master_records.csv -o data/site_all_master_records.csv
  (site_sim_detail.json은 sim-detail-data.js의 객체 부분을 JSON으로 저장한 것)

실행: python3 analysis/recompute.py   (numpy, scipy)
"""
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = {}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def feasible(x, n, nd=2):
    """소수 nd자리로 반올림한 x가 k/n (정수 k)로 나올 수 있는가."""
    return any(abs(round(k / n, nd) - x) < 1e-9 for k in range(n + 1))


def two_prop(k1, n1, k2, n2):
    """두 비율 차이: Fisher 정확 검정(양측)과 z 검정."""
    _, p_f = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (k1 / n1 - k2 / n2) / se if se > 0 else float("nan")
    return {"fisher_p": p_f, "z": z, "z_p": 2 * (1 - stats.norm.cdf(abs(z)))}


# ---------------------------------------------------------------- 논문 표 (옮김)
# 표 3 (시뮬레이션): Seen SR, Sub / ZS SR, Sub / Scr SR, Sub / P+FT SR, Sub / Δ SR, Sub
T3 = {
    "OpenVLA": ("VLA", [.29, .60, .06, .20, .14, .34, .20, .44, .06, .10]),
    "RDT-1B": ("VLA", [.43, .72, .07, .19, .35, .55, .21, .43, -.14, -.11]),
    "GR00T-N1.6": ("VLA", [.51, .81, .03, .17, .21, .49, .67, .79, .46, .31]),
    "pi0.5": ("VLA", [.73, .89, .09, .22, .80, .86, .85, .91, .05, .04]),
    "UniSkill": ("Skill", [.75, .89, .07, .17, .27, .46, .59, .76, .32, .30]),
    "XSkill": ("Skill", [.79, .91, .10, .25, .35, .51, .73, .82, .38, .31]),
    "DP/DINOv2": ("VA", [.67, .84, .07, .21, .13, .36, .54, .73, .41, .37]),
    "DP/SigLIP2": ("VA", [.61, .84, .05, .20, .11, .35, .48, .69, .38, .34]),
    "DP/VideoMAE": ("VA", [.14, .44, .02, .15, .09, .27, .36, .58, .27, .31]),
    "VQ-BeT/DINOv2": ("VA", [.52, .76, .13, .27, .35, .53, .23, .43, -.12, -.10]),
    "VQ-BeT/SigLIP2": ("VA", [.57, .78, .08, .23, .26, .46, .17, .34, -.08, -.11]),
    "VQ-BeT/VideoMAE": ("VA", [.30, .51, .07, .21, .14, .36, .15, .33, .01, -.04]),
    "ACT/DINOv2": ("VA", [.81, .93, .02, .13, .76, .83, .84, .88, .09, .05]),
    "ACT/SigLIP2": ("VA", [.79, .93, .03, .12, .65, .78, .80, .86, .15, .08]),
    "ACT/VideoMAE": ("VA", [.72, .88, .04, .14, .63, .76, .82, .87, .19, .12]),
}
# 표 4 (실제 로봇): 모델 -> {설정: (SR, Q, WR)}, Δ = (SR, Q, WR)
T4 = {
    "pi0.5": {"seen": (.51, 6.87, .56), "zs": (.04, 2.34, .04), "scr": (.27, 5.52, .50), "pft": (.36, 5.89, .73), "d": (.09, .37, .23)},
    "XSkill": {"seen": (.63, 7.49, .89), "zs": (.29, 5.24, .50), "scr": (.28, 5.40, .52), "pft": (.49, 6.72, .95), "d": (.21, 1.32, .43)},
    "ACT/DINOv2": {"seen": (.49, 6.70, .41), "zs": (.26, 5.44, .42), "scr": (.22, 5.38, .24), "pft": (.35, 5.88, .70), "d": (.13, .50, .46)},
    "DP/DINOv2": {"seen": (.39, 6.19, .14), "zs": (.22, 4.96, .25), "scr": (.22, 5.32, .24), "pft": (.31, 5.77, .58), "d": (.09, .45, .34)},
}
# 표 5 (실제 로봇 P+FT, 레벨별): 모델 -> [(SR, Q) x L0..L3]
T5 = {
    "pi0.5": [(.44, 6.27), (.35, 5.77), (.37, 5.97), (.28, 5.57)],
    "XSkill": [(.53, 6.80), (.57, 7.20), (.56, 7.23), (.29, 5.63)],
    "ACT/DINOv2": [(.37, 6.00), (.37, 5.90), (.33, 5.77), (.33, 5.83)],
    "DP/DINOv2": [(.33, 5.97), (.37, 6.00), (.29, 5.70), (.25, 5.43)],
}
T5_AVG = [(.42, 6.26), (.42, 6.22), (.39, 6.17), (.29, 5.62)]
# 표 9: 레벨 x 규모(15/30/45): Sim ZS, Sim P+FT, Real ZS, Real P+FT
T9 = {
    "sim_zs": {"L0": [.08, .06, .06], "L1": [.14, .09, .07], "L2": [.04, .07, .06], "L3": [.03, .01, .04]},
    "sim_pft": {"L0": [.52, .59, .62], "L1": [.48, .59, .58], "L2": [.34, .36, .41], "L3": [.47, .55, .59]},
    "real_zs": {"L0": [.18, .23, .23], "L1": [.18, .20, .24], "L2": [.14, .16, .20], "L3": [.19, .23, .25]},
    "real_pft": {"L0": [.34, .44, .48], "L1": [.35, .45, .45], "L2": [.31, .37, .49], "L3": [.23, .28, .36]},
}
# 표 10 (데모 교체, 시뮬 seen, 45-task 체크포인트): 모델 -> {영상: [L0..L3, Avg]}
T10 = {
    "ACT/DINOv2": {"original": [.87, .77, .72, .88, .81], "similar": [.32, .29, .32, .32, .31], "unrelated": [.34, .28, .39, .19, .30]},
    "DP/DINOv2": {"original": [.72, .66, .59, .71, .67], "similar": [.02, .05, .01, .04, .03], "unrelated": [0, 0, 0, 0, 0]},
    "pi0.5": {"original": [.85, .67, .70, .69, .73], "similar": [.09, .18, .09, .23, .15], "unrelated": [.01, 0, 0, .02, .01]},
    "XSkill": {"original": [.87, .85, .77, .68, .79], "similar": [.47, .41, .05, .29, .31], "unrelated": [.39, .23, .03, .27, .23]},
}
# 표 11 (시뮬 전체 50과제, DP와 ACT, 10회씩): 과제 -> (DP[L0..L3], ACT[L0..L3], 평균DP, 평균ACT)
T11 = {
    "CleanCup": ([40, 50, 20, 100], [30, 30, 70, 90], 52.5, 55.0),
    "CleanDesk": ([10, 60, 40, 60], [80, 100, 100, 100], 42.5, 95.0),
    "CutFruit": ([100, 50, 0, 100], [100, 100, 0, 100], 62.5, 75.0),
    "FoldBox": ([100, 100, 100, 100], [100, 100, 100, 100], 100.0, 100.0),
    "FoldTowel": ([60, 40, 10, 40], [100, 100, 100, 80], 37.5, 95.0),
    "GrindFood": ([100, 100, 100, 100], [90, 100, 100, 70], 100.0, 90.0),
    "KnifeBowlFork": ([0, 60, 60, 90], [100, 100, 80, 50], 52.5, 82.5),
    "LiftLidFromSkillet": ([0, 80, 80, 100], [100, 100, 100, 100], 65.0, 100.0),
    "OpenBox": ([100, 90, 0, 100], [100, 100, 0, 100], 72.5, 75.0),
    "OpenLiquidCap": ([60, 50, 70, 50], [90, 60, 90, 0], 57.5, 60.0),
    "PickAppleBananaToBaskets": ([50, 100, 50, 100], [90, 90, 50, 50], 75.0, 70.0),
    "PickAppleBasket": ([70, 70, 100, 80], [100, 90, 100, 100], 80.0, 97.5),
    "PickAppleToScale": ([40, 90, 40, 80], [90, 10, 90, 80], 62.5, 67.5),
    "PickFood": ([40, 100, 10, 90], [100, 70, 30, 90], 60.0, 72.5),
    "PickFruitsToPlate": ([90, 90, 100, 50], [20, 40, 100, 0], 82.5, 40.0),
    "PickPillToRegions": ([40, 60, 50, 10], [90, 90, 70, 80], 40.0, 82.5),
    "PickRemoteControl": ([100, 100, 100, 40], [100, 100, 100, 80], 85.0, 95.0),
    "PickTennisBallGolfBall": ([90, 90, 90, 100], [60, 80, 70, 50], 92.5, 65.0),
    "PickWash": ([70, 100, 30, 80], [90, 100, 60, 90], 70.0, 85.0),
    "PlaceBookBookcase": ([100, 100, 90, 90], [100, 80, 100, 30], 95.0, 77.5),
    "PlaceBrushRest": ([90, 70, 100, 100], [80, 100, 100, 100], 90.0, 95.0),
    "PlaceBurgerTray": ([80, 90, 80, 70], [90, 80, 90, 70], 80.0, 82.5),
    "PlaceChipsRack": ([90, 80, 100, 80], [70, 70, 70, 100], 87.5, 77.5),
    "PlaceClothBasket": ([100, 100, 80, 10], [100, 100, 90, 90], 72.5, 95.0),
    "PlaceCommodityRack": ([40, 50, 30, 30], [100, 100, 90, 0], 37.5, 72.5),
    "PlaceCupPlate": ([40, 20, 70, 30], [50, 60, 70, 0], 40.0, 45.0),
    "PlaceFileFolder": ([80, 70, 100, 80], [80, 80, 60, 100], 82.5, 80.0),
    "PlaceFoodScale": ([40, 20, 90, 0], [60, 100, 100, 0], 37.5, 65.0),
    "PlaceFruitBox": ([80, 20, 90, 30], [70, 30, 90, 60], 55.0, 62.5),
    "PlaceMagazineFolder": ([80, 100, 0, 70], [100, 100, 0, 70], 62.5, 67.5),
    "PlaceMugRack": ([100, 70, 80, 90], [100, 100, 80, 100], 85.0, 95.0),
    "PlacePillBox": ([80, 70, 60, 30], [100, 60, 80, 70], 60.0, 77.5),
    "PlacePlateRack": ([90, 40, 30, 80], [100, 90, 50, 80], 60.0, 80.0),
    "PlaceScrewdriver": ([30, 20, 30, 30], [100, 80, 40, 100], 27.5, 80.0),
    "PlaceShoeBox": ([90, 90, 60, 100], [100, 100, 90, 90], 85.0, 95.0),
    "PourCup": ([10, 100, 100, 90], [100, 100, 100, 100], 75.0, 100.0),
    "PourKetchupFries": ([50, 50, 100, 30], [100, 100, 80, 60], 57.5, 85.0),
    "PourKettle": ([90, 80, 0, 80], [100, 90, 0, 100], 62.5, 72.5),
    "PourLiquidCup": ([10, 0, 0, 10], [10, 10, 0, 0], 5.0, 5.0),
    "PourLiquidFilter": ([60, 60, 20, 50], [100, 100, 90, 90], 47.5, 95.0),
    "PourLiquidMug": ([60, 10, 100, 20], [0, 90, 70, 0], 47.5, 40.0),
    "PressJuicer": ([60, 80, 0, 100], [20, 100, 0, 100], 60.0, 55.0),
    "PressStapler": ([100, 70, 100, 90], [50, 90, 40, 40], 90.0, 55.0),
    "PutBox": ([100, 90, 90, 10], [100, 90, 100, 100], 72.5, 97.5),
    "PutCubeOnScale": ([100, 90, 90, 60], [100, 100, 100, 70], 85.0, 92.5),
    "ScanMilkBox": ([100, 100, 100, 90], [100, 100, 100, 100], 97.5, 100.0),
    "ScanPillBottle": ([90, 50, 90, 10], [100, 100, 90, 0], 60.0, 72.5),
    "StirSpoon": ([20, 0, 40, 10], [50, 50, 40, 80], 17.5, 55.0),
    "TransFood": ([50, 90, 30, 30], [60, 90, 0, 20], 50.0, 42.5),
    "WipePot": ([10, 0, 0, 100], [20, 0, 0, 80], 27.5, 25.0),
}


def section(title):
    print("\n" + "=" * 8, title)


# ================================================================ V1 표 3 내부 일관성
section("V1 표 3: Δ = P+FT - Scr, 본문 수치")
bad = []
for m, (fam, v) in T3.items():
    for i, name in ((0, "SR"), (1, "Sub")):
        d = round(v[6 + i] - v[4 + i], 2)
        if abs(d - v[8 + i]) > 0.011:
            bad.append((m, name, d, v[8 + i]))
print("Δ 불일치(허용 0.01 반올림):", bad)
pft_wins = sum(1 for m, (f, v) in T3.items() if v[8] > 0)
zs_max = max((v[2], m) for m, (f, v) in T3.items())
fam_seen = {}
for m, (f, v) in T3.items():
    fam_seen.setdefault(f, []).append(v[0])
fam_mean = {f: round(float(np.mean(x)), 3) for f, x in fam_seen.items()}
print("P+FT가 Scr보다 큰 변형:", pft_wins, "/ 15 (본문: 12)")
print("ZS 최대:", zs_max, "(초록: 모든 모델 13% 미만)")
print("패밀리 평균 seen SR:", fam_mean, "(부록 C.1: 0.49, 0.77, 0.57)")
col = lambda j: float(np.mean([v[j] for f, v in T3.values()]))
means = {n: round(col(j), 4) for n, j in (("seen", 0), ("zs", 2), ("scr", 4), ("pft", 6))}
print("표 3 15개 변형 평균:", means, " (부록 C.3: Scr 0.35)")
OUT["V1"] = {"delta_mismatch": bad, "pft_wins": pft_wins, "zs_max": zs_max, "family_mean_seen": fam_mean, "col_means": means}

# ================================================================ V2 표 4 일관성
section("V2 표 4: Δ, 환산, WR 평균")
bad = []
for m, r in T4.items():
    for i, nm in enumerate(("SR", "Q", "WR")):
        d = round(r["pft"][i] - r["scr"][i], 2)
        if abs(d - r["d"][i]) > 0.011:
            bad.append((m, nm, d, r["d"][i]))
print("Δ 불일치:", bad)
wr_mean = {k: round(float(np.mean([r[k][2] for r in T4.values()])), 4) for k in ("seen", "zs", "scr", "pft")}
print("모델 평균 WR(설정별):", wr_mean, "-> 같은 묶음 안에서는 0.5여야 함(승점 합 보존, 참가 횟수 동일 가정)")
print("transfer(Scr+P+FT) 묶음 WR 평균:", round((wr_mean["scr"] + wr_mean["pft"]) / 2, 4))
scr_real = float(np.mean([r["scr"][0] for r in T4.values()]))
print("실제 Scr SR 평균:", round(scr_real, 4), "(부록 C.3: 0.25)")
OUT["V2"] = {"delta_mismatch": bad, "wr_mean": wr_mean, "scr_real_mean": scr_real}

# ================================================================ V3 표 5와 표 4/9 교차
section("V3 표 5: 평균 행, 표 4 P+FT, 표 9와의 일치")
avg_sr = [round(float(np.mean([T5[m][L][0] for m in T5])), 4) for L in range(4)]
avg_q = [round(float(np.mean([T5[m][L][1] for m in T5])), 4) for L in range(4)]
print("평균 SR:", avg_sr, "표기:", [a for a, _ in T5_AVG])
print("평균 Q :", avg_q, "표기:", [b for _, b in T5_AVG])
cross = {}
for m in T5:
    sr = round(float(np.mean([x[0] for x in T5[m]])), 4)
    q = round(float(np.mean([x[1] for x in T5[m]])), 4)
    cross[m] = (sr, T4[m]["pft"][0], q, T4[m]["pft"][1])
print("모델별 레벨 평균 vs 표 4 P+FT (SR, 표기SR, Q, 표기Q):", cross)
t9_real_pft = [round(float(np.mean(T9["real_pft"][f"L{L}"])), 3) for L in range(4)]
t9_sim_pft = [round(float(np.mean(T9["sim_pft"][f"L{L}"])), 3) for L in range(4)]
print("표 9 real P+FT 규모 평균:", t9_real_pft, "(표 5 평균 행과 같아야 함)")
print("표 9 sim P+FT 규모 평균:", t9_sim_pft, "(부록 C.5 패널 d: 0.58/0.55/0.37/0.54)")
zs_real_scale = [round(float(np.mean([T9["real_zs"][f"L{L}"][s] for L in range(4)])), 4) for s in range(3)]
print("표 9 real ZS 규모별 4레벨 평균:", zs_real_scale, " 표 4 ZS 평균:", round(float(np.mean([r['zs'][0] for r in T4.values()])), 4))
# 부록 C.3: 규모별 ZS (XSkill 0.24->0.33, ACT 0.22->0.30, DP 0.19->0.25, pi0.5 0.04)
c3 = {15: (.24 + .22 + .19 + .04) / 4, 45: (.33 + .30 + .25 + .04) / 4}
print("C.3 수치로 만든 15/45 규모 ZS 평균:", {k: round(v, 4) for k, v in c3.items()}, "표 9: ", zs_real_scale[0], zs_real_scale[2])
sim_zs_mean = float(np.mean([np.mean(v) for v in T9["sim_zs"].values()]))
print("표 9 sim ZS 전체 평균:", round(sim_zs_mean, 4), " 표 3 ZS 평균:", round(means["zs"], 4))
feas = {}
for m in T5:
    feas[m] = [feasible(T5[m][L][0], 75) for L in range(4)]
print("표 5 SR의 k/75 가능성(레벨당 5과제x5회x3규모=75회 가정):", feas)
print("평균 행 k/300 가능성:", [feasible(a, 300) for a, _ in T5_AVG])
OUT["V3"] = {"avg_sr": avg_sr, "avg_q": avg_q, "cross": cross, "t9_real_pft": t9_real_pft, "t9_sim_pft": t9_sim_pft,
             "zs_real_scale": zs_real_scale, "k75_feasible": feas}

# ================================================================ V4 표 10
section("V4 표 10: Avg 열, 표 3과의 비교")
bad = []
for m, d in T10.items():
    for k, v in d.items():
        a = round(float(np.mean(v[:4])), 2)
        if abs(a - v[4]) > 0.011:
            bad.append((m, k, a, v[4]))
print("Avg 불일치:", bad)
cmp = {m: (T10[m]["original"][4], T3[mm][1][0]) for m, mm in (("ACT/DINOv2", "ACT/DINOv2"), ("DP/DINOv2", "DP/DINOv2"), ("pi0.5", "pi0.5"), ("XSkill", "XSkill"))}
print("표 10 original Avg(45-task) vs 표 3 Seen SR(15/30/45 평균):", cmp)
ret = {m: {k: round(T10[m][k][4] / T10[m]["original"][4], 2) for k in ("similar", "unrelated")} for m in T10}
print("원본 대비 유지율:", ret)
OUT["V4"] = {"avg_mismatch": bad, "orig_vs_table3": cmp, "retention": ret}

# ================================================================ V5 표 11
section("V5 표 11 (시뮬 전체 50과제)")
bad_mean = []
not10 = []
for t, (dp, act, mdp, mact) in T11.items():
    if abs(np.mean(dp) - mdp) > 0.06 or abs(np.mean(act) - mact) > 0.06:
        bad_mean.append(t)
    if any(x % 10 for x in dp + act):
        not10.append(t)
print("과제 수:", len(T11), " 평균 열 불일치:", bad_mean, " 10의 배수 아님:", not10)
dp_all = np.array([v[0] for v in T11.values()])
act_all = np.array([v[1] for v in T11.values()])
lvl = {"DP": dp_all.mean(0).round(1).tolist(), "ACT": act_all.mean(0).round(1).tolist()}
print("레벨별 평균(%):", lvl)
both_zero_l2 = [t for t, (dp, act, a, b) in T11.items() if dp[2] == 0 and act[2] == 0]
l2_broken = [t for t in both_zero_l2 if min(T11[t][0][0], T11[t][1][0], T11[t][0][3], T11[t][1][3]) >= 50 or t in ("CutFruit", "OpenBox", "PlaceMagazineFolder", "PourKettle", "PressJuicer")]
zero_cells = {"DP": int((dp_all == 0).sum()), "ACT": int((act_all == 0).sum())}
print("L0, L3 에서 잘 되는데 L2만 DP와 ACT 모두 0%인 과제:", l2_broken)
print("DP와 ACT가 L2에서 모두 0%인 과제:", both_zero_l2, " 0% 셀 수:", zero_cells)
others = [(t, dp[0], dp[1], dp[3], act[0], act[1], act[3]) for t, (dp, act, a, b) in T11.items() if t in both_zero_l2]
print("그 과제들의 다른 레벨(DP L0 L1 L3, ACT L0 L1 L3):", others)
# 레벨 비교: L3 대 L0 (50과제 x 10회 x 2모델)
for nm, arr in (("DP", dp_all), ("ACT", act_all)):
    k = [int(arr[:, L].sum() / 10) for L in range(4)]
    n = 50 * 10
    print(nm, "레벨별 성공 횟수/500:", k, "L3 vs L0 Fisher p:", round(two_prop(k[3], n, k[0], n)["fisher_p"], 4),
          "L2 vs 나머지:", round(two_prop(k[2], n, sum(k[i] for i in (0, 1, 3)), 3 * n)["fisher_p"], 5))
w = stats.wilcoxon(dp_all[:, 3] - dp_all[:, 0]); w2 = stats.wilcoxon(act_all[:, 3] - act_all[:, 0])
print("과제 단위 L3-L0 Wilcoxon p: DP", round(w.pvalue, 3), " ACT", round(w2.pvalue, 3))
seen5 = ["StirSpoon", "FoldTowel", "PlaceMugRack", "PlaceFileFolder", "PlacePlateRack"]
unseen5 = ["PickRemoteControl", "ScanMilkBox", "PourKettle", "PickFood", "FoldBox"]
print("seen 5과제 ACT 평균:", np.mean([T11[t][3] for t in seen5]), "DP:", np.mean([T11[t][2] for t in seen5]))
print("unseen 5과제 DP 레벨별:", np.mean([T11[t][0] for t in unseen5], 0).tolist(), "ACT:", np.mean([T11[t][1] for t in unseen5], 0).tolist())
site_full = None
fp = DATA / "site_full_task_suite.js"
if fp.exists():
    s = fp.read_text()
    site_full = json.loads(s[s.index("{", s.index("window.FULL_TASK_SUITE")):].rstrip().rstrip(";"))
    diff = []
    for r in site_full["tasks"]:
        dp, act, a, b = T11[r["name"].replace(" ", "")]
        if r["dp"] != dp or r["act"] != act or r["meanDP"] != a or r["meanACT"] != b:
            diff.append(r["name"])
    print("프로젝트 페이지 50과제 표와 논문 표 11의 불일치:", diff)
OUT["V5"] = {"level_means_pct": lvl, "both_zero_L2": both_zero_l2, "L2_zero_only_L2": l2_broken, "zero_cells": zero_cells, "not_multiple_of_10": not10,
             "mean_col_mismatch": bad_mean}

# ================================================================ V6 개수 일관성
section("V6 롤아웃 수, 판정 횟수, 데이터셋 수")
n_var, scales = 15, 3
per_pair_sim = 10
pairs = 20  # 5과제 x 4레벨
seen_unseen_settings = {"seen": 1, "zs": 1, "pft": 1}
sim_per_variant = pairs * per_pair_sim * (3 * 3 + 3)  # seen/zs/pft 3규모 + scr
sim_scr_once = pairs * per_pair_sim * (3 * 3 + 1)
print("시뮬 롤아웃/변형: 3설정x3규모+scr(규모마다 1회)=", sim_per_variant, " scr 1회만=", sim_scr_once)
print("  x15 변형:", sim_per_variant * n_var, "(논문: 36,000), scr 1회만:", sim_scr_once * n_var)
real_per_model = pairs * 5 * (3 * 3 + 3)
print("실제 롤아웃/모델: ", real_per_model, " x4 =", real_per_model * 4, "(논문: 4,800)")
sim_screens, real_screens = 15000, 5000
print("판정 슬롯: 시뮬", sim_screens * 2, "/ 36,000 롤아웃 =", round(sim_screens * 2 / 36000, 2), "회/롤아웃;  실제", real_screens * 2, "/ 4,800 =", round(real_screens * 2 / 4800, 2))
print("화면 분할 합: 시뮬", 6000 + 6000 + 3000, "실제", 2500 + 2500, "(실제에는 zero-shot 묶음이 없음)")
dom = {"Household": 3550, "Supermarket": 1770, "Restaurant": 3300, "Logistics": 750, "Hospital": 1100, "Laboratory": 1250}
tasks = {"Household": 12, "Supermarket": 7, "Restaurant": 17, "Logistics": 4, "Hospital": 6, "Laboratory": 7}
omit = {"Logistics": 1, "Hospital": 2, "Laboratory": 3}
real_total = sum(dom.values())
variants = sum((4 * t - omit.get(d, 0)) for d, t in tasks.items())
print("도메인 합:", real_total, " 과제 합:", sum(tasks.values()), " 과제-레벨 변형:", variants, " 쌍당 평균 에피소드:", round(real_total / variants, 1), "(B.1: 평균 50)")
per_domain = {d: round(dom[d] / (4 * tasks[d] - omit.get(d, 0)), 1) for d in dom}
print("도메인별 쌍당 에피소드:", per_domain)
OUT["V6"] = {"sim_rollouts_per_variant": sim_per_variant, "sim_total": sim_per_variant * n_var, "real_total": real_per_model * 4,
             "judgments_per_rollout": {"sim": sim_screens * 2 / 36000, "real": real_screens * 2 / 4800},
             "real_episodes_total": real_total, "task_level_variants": variants, "episodes_per_pair": round(real_total / variants, 1),
             "per_domain": per_domain}

# ================================================================ V7 통계: 핵심 주장
section("V7 통계 검정 (실제 시도 수 가정)")
# 실제 로봇: 레벨당 모델당 5과제 x 5회 x 3규모 = 75회 (표 5). 모델 4개 합 300회.
st = {}
for m in T5:
    k3 = round(T5[m][3][0] * 75); k2 = round(T5[m][2][0] * 75)
    st[m] = {"L3": (k3, 75, [round(x, 3) for x in wilson(k3, 75)]), "L2": (k2, 75, [round(x, 3) for x in wilson(k2, 75)]),
             "L2_vs_L3_fisher": round(two_prop(k2, 75, k3, 75)["fisher_p"], 4)}
print("실제 P+FT, 모델별 L2 대 L3:", json.dumps(st, ensure_ascii=False))
l012 = round(np.mean([[T5_AVG[i][0]] for i in (0, 1, 2)]) * 300 * 3)
k_l3 = round(.29 * 300)
res = two_prop(k_l3, 300, l012, 900)
print("실제 4모델 평균 L3(0.29, n=300) 대 L0-L2(0.41, n=900): ", {k: round(v, 4) for k, v in res.items()},
      " L3 Wilson:", [round(x, 3) for x in wilson(k_l3, 300)], "(모델 간 상관과 같은 과제 반복 때문에 실제 p는 더 큼)")
# 평탄하다는 주장: L0-L2 사이 차이
res2 = two_prop(round(.42 * 300), 300, round(.39 * 300), 300)
print("실제 L0 대 L2 (0.42 vs 0.39):", {k: round(v, 3) for k, v in res2.items()})
# 모델 간 비교: 실제 seen SR (n=300)
real_seen = {m: round(T4[m]["seen"][0] * 300) for m in T4}
pairs_t = {}
for a, b in (("XSkill", "ACT/DINOv2"), ("XSkill", "pi0.5"), ("ACT/DINOv2", "DP/DINOv2")):
    pairs_t[f"{a} vs {b}"] = round(two_prop(real_seen[a], 300, real_seen[b], 300)["fisher_p"], 4)
print("실제 seen 모델 쌍 Fisher p (n=300 가정):", pairs_t)
# 초록: 13% 미만 주장 (표 3 최대 0.13) — 실제 로봇은 별도
real_zs = {m: (T4[m]["zs"][0], [round(x, 3) for x in wilson(round(T4[m]["zs"][0] * 300), 300)]) for m in T4}
print("실제 ZS SR와 Wilson(n=300 가정):", real_zs)
# 시뮬: Scr 대비 P+FT가 늘었다는 주장, 변형별 n=200 (5과제 x 4레벨 x 10회) x 3규모 합 가정이 아니라 규모 평균: 200회
sim_d = {}
for m, (f, v) in T3.items():
    kp, ks = round(v[6] * 200), round(v[4] * 200)
    sim_d[m] = {"d": v[8], "fisher_p": round(two_prop(kp, 200, ks, 200)["fisher_p"], 4)}
print("표 3 P+FT 대 Scr (n=200 회 가정):", {m: (d["d"], d["fisher_p"]) for m, d in sim_d.items()})
ns = sum(1 for d in sim_d.values() if d["fisher_p"] < 0.05)
print("p<0.05인 변형 수:", ns, "/ 15")
ns_pos = sum(1 for d_ in sim_d.values() if d_["fisher_p"] < 0.05 and d_["d"] > 0)
ns_neg = sum(1 for d_ in sim_d.values() if d_["fisher_p"] < 0.05 and d_["d"] < 0)
print("유의하게 좋아진 변형:", ns_pos, " 유의하게 나빠진 변형:", ns_neg, " 유의하지 않은 변형:", 15 - ns_pos - ns_neg)
# 실제 로봇: XSkill을 뺀 3개 모델에서 L3 대 L0-L2 (모델당 레벨 75회 가정)
oth = [m for m in T5 if m != "XSkill"]
k3o = sum(round(T5[m][3][0] * 75) for m in oth)
k012o = sum(round(T5[m][L][0] * 75) for m in oth for L in range(3))
res3 = two_prop(k3o, 75 * 3, k012o, 75 * 9)
print("XSkill 제외 3모델 L3", k3o, "/225 대 L0-L2", k012o, "/675: ", {k: round(v, 4) for k, v in res3.items()})
res4 = two_prop(round(.29 * 75), 75, round(.56 * 75), 75)
print("XSkill L3(0.29) 대 L2(0.56) n=75: ", {k: round(v, 4) for k, v in res4.items()})
OUT["V7b"] = {"sim_pft_sig_pos": ns_pos, "sim_pft_sig_neg": ns_neg, "real_without_xskill_L3_vs_L012": {k: round(v, 4) for k, v in res3.items()}}
# 추가 모델 간 비교
def _k(p, n):
    return round(p * n)
extra = {
    "실제 ZS XSkill 대 ACT (n=300)": two_prop(_k(.29, 300), 300, _k(.26, 300), 300)["fisher_p"],
    "실제 Scr XSkill 대 pi0.5 (n=100, 5과제x4레벨x5회)": two_prop(_k(.28, 100), 100, _k(.27, 100), 100)["fisher_p"],
    "실제 ZS XSkill 대 pi0.5 (n=300)": two_prop(_k(.29, 300), 300, _k(.04, 300), 300)["fisher_p"],
    "시뮬 Seen ACT/DINOv2 대 XSkill (n=600)": two_prop(_k(.81, 600), 600, _k(.79, 600), 600)["fisher_p"],
    "시뮬 Seen ACT/DINOv2 대 pi0.5 (n=600)": two_prop(_k(.81, 600), 600, _k(.73, 600), 600)["fisher_p"],
    "시뮬 P+FT pi0.5 대 ACT/DINOv2 (n=600)": two_prop(_k(.85, 600), 600, _k(.84, 600), 600)["fisher_p"],
}
extra = {k: round(v, 4) for k, v in extra.items()}
print("추가 비교 Fisher p:", extra)
OUT["V7c"] = extra
OUT["V7"] = {"real_models_L2_L3": st, "real_L3_vs_L012": {k: round(v, 5) for k, v in res.items()}, "real_L0_vs_L2": res2,
             "real_seen_pairs_fisher": pairs_t, "real_zs_wilson": real_zs, "sim_pft_vs_scr": sim_d, "sim_pft_vs_scr_sig": ns}

# ================================================================ V8 셀 단위(프로젝트 페이지)로 표 3 다시 만들기
section("V8 프로젝트 페이지 셀 데이터로 표 3과 레벨별 수치 재구성")
sp = DATA / "site_sim_detail.json"
if sp.exists():
    d = json.loads(sp.read_text())
    models = [m["short"] for m in d["models"]]
    keymap = {m["key"]: m["short"] for m in d["models"]}
    mk = [m["key"] for m in d["models"]]
    tasks_seen = [t["key"] for t in d["tasks"] if t["seen"]]
    tasks_unseen = [t["key"] for t in d["tasks"] if not t["seen"]]
    LV = d["levels"]

    def cell(block, task, L, key, idx=0):
        v = d[block][task][L][key]
        return None if v is None else v[idx]

    rec = {}
    for key in mk:
        short = keymap[key]
        row = {}
        for blk, tl in (("seen", tasks_seen), ("zs", tasks_unseen), ("pft", tasks_unseen)):
            for idx, nm in ((0, "SR"), (1, "Sub")):
                vals = [cell(blk, t, L, key, idx) for sc in ("15", "30", "45") for t in tl for L in LV
                        for _ in [0] if True for _v in [0]] if False else None
            # 규모별 블록
            sr, sub = [], []
            for sc in ("15", "30", "45"):
                for t in tl:
                    for L in LV:
                        v = d[blk][sc][t][L][key]
                        if v is not None:
                            sr.append(v[0]); sub.append(v[1])
            row[blk] = (round(float(np.mean(sr)), 4), round(float(np.mean(sub)), 4), len(sr))
        scr = [d["scr"][t][L][key] for t in tasks_unseen for L in LV]
        scr = [v for v in scr if v is not None]
        row["scr"] = (round(float(np.mean([v[0] for v in scr])), 4), round(float(np.mean([v[1] for v in scr])), 4), len(scr))
        rec[short] = row
    # 논문 표 3과 비교
    name_map = {"pi0.5": "π₀.₅"}
    mism = []
    for m, (fam, v) in T3.items():
        r = rec.get(name_map.get(m, m)) or rec.get(m)
        if r is None:
            mism.append((m, "없음")); continue
        sims = [r["seen"][0], r["seen"][1], r["zs"][0], r["zs"][1], r["scr"][0], r["scr"][1], r["pft"][0], r["pft"][1]]
        diffs = [round(abs(a - b), 3) for a, b in zip(sims, v[:8])]
        if max(diffs) > 0.0051:
            mism.append((m, diffs))
    print("셀 데이터로 만든 표 3 vs 논문 표 3 (|차이|>0.005):", mism if mism else "없음")
    print("모델 수:", len(rec), " seen 셀 수(모델당):", rec[models[0]]["seen"][2], " zs:", rec[models[0]]["zs"][2], " scr:", rec[models[0]]["scr"][2])
    # 규모 수준 셀이 k/10 인가
    nonk = 0; tot = 0
    for blk in ("seen", "zs", "pft"):
        for sc in ("15", "30", "45"):
            for t in d[blk][sc]:
                for L in LV:
                    for key in mk:
                        v = d[blk][sc][t][L][key]
                        if v is None:
                            continue
                        tot += 1
                        if abs(v[0] * 10 - round(v[0] * 10)) > 1e-6:
                            nonk += 1
    print("규모 단위 셀 중 k/10이 아닌 셀:", nonk, "/", tot)
    # 레벨별 평균 (표 9, 그림 12c, d 재구성)
    lvl_rec = {}
    for blk, tl in (("seen", tasks_seen), ("zs", tasks_unseen), ("pft", tasks_unseen)):
        per_scale = {}
        for sc in ("15", "30", "45"):
            per_scale[sc] = [round(float(np.mean([d[blk][sc][t][L][key][0] for key in mk for t in tl if d[blk][sc][t][L][key] is not None])), 3) for L in LV]
        lvl_rec[blk] = per_scale
    print("레벨별(L0..L3) 규모별 SR - pft:", lvl_rec["pft"], "\n  zs:", lvl_rec["zs"])
    print("seen 전 규모 평균:", [round(float(np.mean([lvl_rec['seen'][sc][i] for sc in ('15', '30', '45')])), 3) for i in range(4)], "(그림 12c: 0.62/0.57/0.48/0.63)")
    t9_sim_pft_site = {L: [lvl_rec["pft"][sc][i] for sc in ("15", "30", "45")] for i, L in enumerate(LV)}
    t9_sim_zs_site = {L: [lvl_rec["zs"][sc][i] for sc in ("15", "30", "45")] for i, L in enumerate(LV)}
    dev = max(abs(a - b) for L in LV for a, b in zip(t9_sim_pft_site[L], T9["sim_pft"][L]))
    dev2 = max(abs(a - b) for L in LV for a, b in zip(t9_sim_zs_site[L], T9["sim_zs"][L]))
    print("표 9 sim P+FT/ZS 대비 최대 편차:", round(dev, 3), round(dev2, 3))
    # 레벨 간 비교: seen 시뮬, (모델, 과제, 규모) 단위 짝 비교
    diffs = {}
    for a, b in (("L3", "L0"), ("L2", "L0"), ("L3", "L2")):
        x = []
        for key in mk:
            for t in tasks_seen:
                for sc in ("15", "30", "45"):
                    x.append(d["seen"][sc][t][a][key][0] - d["seen"][sc][t][b][key][0])
        x = np.array(x)
        diffs[f"{a}-{b}"] = {"mean": round(float(x.mean()), 3), "wilcoxon_p": float(stats.wilcoxon(x[x != 0]).pvalue), "n": len(x)}
    print("seen 시뮬 레벨 짝 비교(셀 225개, 모델-과제-규모):", diffs)
    # 과제 군집 부트스트랩 (과제 5개 재표집) : seen 레벨 평균의 L3-L0
    rng = np.random.default_rng(0)
    arr = np.zeros((len(tasks_seen), 4))
    for i, t in enumerate(tasks_seen):
        for j, L in enumerate(LV):
            arr[i, j] = np.mean([d["seen"][sc][t][L][key][0] for sc in ("15", "30", "45") for key in mk])
    print("seen 과제x레벨 평균 SR:", arr.round(3).tolist())
    boots = []
    for _ in range(5000):
        idx = rng.integers(0, len(tasks_seen), len(tasks_seen))
        m = arr[idx].mean(0)
        boots.append([m[3] - m[0], m[2] - m[0], m[3] - m[2]])
    boots = np.array(boots)
    print("과제 단위 부트스트랩 95% (L3-L0, L2-L0, L3-L2):", np.percentile(boots, [2.5, 97.5], axis=0).round(3).tolist())
    # P+FT unseen 과제별 L2
    unseen_l2 = {t.replace("TwoRobot", ""): [round(float(np.mean([d["pft"][sc][t][L][key][0] for sc in ("15", "30", "45") for key in mk])), 3) for L in LV] for t in tasks_unseen}
    print("unseen P+FT 과제별 L0..L3(15변형, 3규모 평균):", unseen_l2)
    # 표 10 original 행이 45-task 체크포인트인지 세 규모 평균인지
    t10map = {"ACT/DINOv2": "act / dinov2_vitl14", "DP/DINOv2": "dp / dinov2_vitl14", "pi0.5": "pi", "XSkill": "xskill"}
    t10 = {}
    for mname, key in t10map.items():
        a3 = [round(float(np.mean([d["seen"][sc][t][L][key][0] for t in tasks_seen for sc in ("15", "30", "45")])), 3) for L in LV]
        a45 = [round(float(np.mean([d["seen"]["45"][t][L][key][0] for t in tasks_seen])), 3) for L in LV]
        match3 = all(abs(x - y) < 0.0051 for x, y in zip(a3, T10[mname]["original"][:4]))
        match45 = all(abs(x - y) < 0.0051 for x, y in zip(a45, T10[mname]["original"][:4]))
        t10[mname] = {"avg3scale": a3, "only45": a45, "orig_matches_3scale": match3, "orig_matches_45": match45}
    print("표 10 original 행 대 셀 데이터(세 규모 평균 / 45만):", json.dumps(t10, ensure_ascii=False))
    OUT["V4b"] = t10
    # 부록 C.3: "45 과제에서는 모든 계열의 P+FT가 Scratch 평균(sim 0.35)보다 위"
    fam_of = {m["key"]: m["fam"] for m in d["models"]}
    fam_scale = {}
    for fam in ("VLA", "Skill", "VideoVA"):
        ks = [k for k in mk if fam_of[k] == fam]
        fam_scale[fam] = [round(float(np.mean([d["pft"][sc][t][L][k][0] for k in ks for t in tasks_unseen for L in LV])), 3) for sc in ("15", "30", "45")]
    scr_all = float(np.mean([d["scr"][t][L][k][0] for k in mk for t in tasks_unseen for L in LV]))
    print("계열별 P+FT SR (15/30/45):", fam_scale, " Scr 전체 평균:", round(scr_all, 3))
    OUT["V8b"] = {"family_pft_by_scale": fam_scale, "scr_mean": round(scr_all, 3)}
    # 모든 변형, 모든 규모에서 정확히 0인 (과제, 레벨) 셀: 환경 문제 가능성
    zero_cells = {}
    for blk, tl in (("seen", tasks_seen), ("zs", tasks_unseen), ("pft", tasks_unseen)):
        for t in tl:
            for L in LV:
                vals = [d[blk][sc][t][L][key][0] for sc in ("15", "30", "45") for key in mk if d[blk][sc][t][L][key] is not None]
                if vals and max(vals) == 0.0 and blk != "zs":
                    zero_cells[f"{blk}/{t.replace('TwoRobot', '')}/{L}"] = len(vals)
    scr_zero = {}
    for t in tasks_unseen:
        for L in LV:
            vals = [d["scr"][t][L][key][0] for key in mk if d["scr"][t][L][key] is not None]
            if vals and max(vals) == 0.0:
                scr_zero[f"scr/{t.replace('TwoRobot', '')}/{L}"] = len(vals)
    print("모든 변형과 규모에서 SR이 정확히 0인 셀(seen, pft):", zero_cells, " scr:", scr_zero)
    # PourKettle L2 를 빼면 unseen P+FT 레벨 평균
    base = {t: unseen_l2[t.replace("TwoRobot", "")] for t in tasks_unseen}
    allm = np.mean(list(base.values()), 0).round(3).tolist()
    wo = np.mean([v for t, v in base.items() if "PourKettle" not in t], 0).round(3).tolist()
    print("unseen P+FT 과제 평균 L0..L3:", allm, " PourKettle 제외:", wo)
    OUT["V11"] = {"all_zero_cells": zero_cells, "scr_zero_cells": scr_zero, "unseen_pft_levels_all": allm, "unseen_pft_levels_without_PourKettle": wo}
    OUT["V8"] = {"table3_rebuild_mismatch": mism, "non_k10_cells": [nonk, tot], "pairwise_levels": diffs,
                 "seen_task_level_mean": arr.round(3).tolist(), "bootstrap95_task": np.percentile(boots, [2.5, 97.5], axis=0).round(3).tolist(),
                 "unseen_pft_task_levels": unseen_l2, "t9_dev": [round(dev, 3), round(dev2, 3)],
                 "rebuild_rows": {m: {k: list(v) for k, v in r.items()} for m, r in rec.items()}}
else:
    print("data/site_sim_detail.json 없음: 건너뜀")

# ================================================================ V9 사람-자동 일치도 (부록 C.7)
section("V9 부록 C.7: 자동 지표와 사람 판정의 상관")
fp = DATA / "site_figdata.js"
if fp.exists():
    s = fp.read_text()
    pts = re.findall(r'\{ model: "([^"]+)", fam: "[^"]+", setting: "([^"]+)", x: ([\d.]+), y: ([\d.]+), subx: ([\d.]+), q: ([\d.]+) \}', s)
    P = [(m, st_, float(x), float(y), float(sx), float(q)) for m, st_, x, y, sx, q in pts]
    print("점 수:", len(P))
    x = np.array([p[2] for p in P]); y = np.array([p[3] for p in P]); sx = np.array([p[4] for p in P]); q = np.array([p[5] for p in P])
    r1 = stats.pearsonr(x, y)[0]; r2 = stats.pearsonr(sx, q)[0]
    print("Pearson r (SR vs SRhuman):", round(r1, 3), " (Sub-SR vs Q):", round(r2, 3), " 논문 본문: 0.858, 0.861;  사이트: 0.956, 0.952(변형x설정)")
    nz = np.array([p[1] != "ZS" for p in P])
    r1b = stats.pearsonr(x[nz], y[nz])[0]; r2b = stats.pearsonr(sx[nz], q[nz])[0]
    print("ZS 제외(45점) r:", round(r1b, 3), round(r2b, 3))
    sr = stats.spearmanr(x[nz], y[nz])[0]
    print("ZS 제외 Spearman:", round(sr, 3))
    # 사람 SR의 분모 추정: y = k/n 이 되는 가장 작은 n
    def smallest_n(v):
        for n in range(1, 201):
            k = round(v * n)
            if abs(k / n - v) < 5.1e-5:
                return n
        return None
    ns_ = [smallest_n(p[3]) for p in P]
    print("사람 SR 최소 가능 분모 n (4자리 반올림 일치 최소 n):", sorted(ns_))
    print("  분모 중앙값:", float(np.median([n for n in ns_ if n])), " 최대:", max(n for n in ns_ if n))
    bad_gap = [(p[0], p[1], p[2], p[3]) for p in P if abs(p[2] - p[3]) > 0.3]
    print("자동 SR과 사람 SR이 0.3 넘게 다른 점:", bad_gap)
    above = sum(1 for p in P if p[3] > p[2]); print("사람 SR > 자동 SR인 점:", above, "/ 60")
    zs_h = [(p[0], p[3]) for p in P if p[1] == "ZS"]
    print("ZS 사람 SR:", zs_h)
    OUT["V9"] = {"n_points": len(P), "r_sr": round(r1, 3), "r_sub_q": round(r2, 3), "r_sr_noZS": round(r1b, 3), "r_sub_q_noZS": round(r2b, 3),
                 "spearman_sr_noZS": round(sr, 3), "min_denominators_sorted": sorted(n for n in ns_ if n), "big_gaps": bad_gap}
    # 레벨별 사람 판정 (그림 13b)
    m_ = re.search(r"arenaSeenByLevel: \{ SRhuman: \[([^\]]+)\], Qbar: \[([^\]]+)\]", s)
    if m_:
        print("그림 13b 사람 SR 레벨별:", m_.group(1), " Q:", m_.group(2), "(자동 SR 0.62/0.57/0.48/0.63)")
else:
    print("data/site_figdata.js 없음: 건너뜀")

# ================================================================ V12 원 기록(프로젝트 페이지 all_master_records.csv, 3,440행)
section("V12 공개된 셀 단위 원 기록 (all_master_records.csv): Sub-SR 형태, Scr 실행 수, 최솟값 대 평균")
import csv
import collections
rp = DATA / "site_all_master_records.csv"
if rp.exists():
    rows = list(csv.DictReader(open(rp, encoding="utf-8")))
    n_ep = sum(int(r["episodes"]) for r in rows)
    print("행 수:", len(rows), " 에피소드 합:", n_ep, "(논문 B.3: 시뮬 고유 롤아웃 36,000)")
    per_split = collections.Counter(r["split"] for r in rows)
    print("split별 행:", dict(per_split), "(전부 채워진 격자는 변형 15 x 규모 3 x 20 환경 = 900행)")
    # Sub-SR이 이산(완료 단계 수 / M)인지, 연속(피크 보상 평균)인지
    def is_frac(v, M):
        return abs(v * 10 * M - round(v * 10 * M)) < 1e-6
    sub = [float(r["SubSR"]) for r in rows]
    frac_any = {M: sum(is_frac(v, M) for v in sub) / len(sub) for M in (2, 3, 4, 5, 6, 8, 10)}
    print("Sub-SR x 10 x M 이 정수인 행 비율 (M=단계 수 가정):", {k: round(v, 3) for k, v in frac_any.items()})
    print("서로 다른 Sub-SR 값 수:", len(set(sub)), " 예:", sorted(set(sub))[:4])
    # Scr 실행 수 (모델, 규모)
    scr = collections.defaultdict(set)
    for r in rows:
        if r["split"] == "unseen_scratch":
            scr[(r["policy"], r["backbone"])].add(r["num_tasks"])
    copies = {f"{k[0]}/{k[1]}": sorted(v) for k, v in scr.items()}
    print("Scr 기록이 있는 규모 표지:", copies)
    n_combo = sum(len(v) for v in scr.values())
    print("Scr (모델, 규모) 조합 수:", n_combo, "/ 45 (변형 15 x 3)")
    # 최솟값 대 평균
    cell = collections.defaultdict(dict)
    for r in rows:
        if r["split"] == "unseen_scratch":
            cell[(r["policy"], r["backbone"], r["env_id"])][r["num_tasks"]] = float(r["SR"])
    per_model = collections.defaultdict(lambda: {"min": [], "mean": [], "n": []})
    for (p_, b_, e_), v in cell.items():
        per_model[(p_, b_)]["min"].append(min(v.values()))
        per_model[(p_, b_)]["mean"].append(float(np.mean(list(v.values()))))
        per_model[(p_, b_)]["n"].append(len(v))
    pft = collections.defaultdict(list)
    for r in rows:
        if r["split"] == "unseen_finetune":
            pft[(r["policy"], r["backbone"])].append(float(r["SR"]))
    res12 = {}
    for k, v in sorted(per_model.items()):
        mn, me = float(np.mean(v["min"])), float(np.mean(v["mean"]))
        pf = float(np.mean(pft[k])) if k in pft else float("nan")
        res12[f"{k[0]}/{k[1]}"] = {"scr_min": round(mn, 3), "scr_mean": round(me, 3), "copies_mean": round(float(np.mean(v["n"])), 2),
                                  "delta_min": round(pf - mn, 3), "delta_mean": round(pf - me, 3)}
    print("모델별 Scr: 최솟값 대 평균, Δ(P+FT - Scr):")
    for k, v in res12.items():
        print("  ", k, v)
    # PourKettle L2
    pk = collections.defaultdict(list)
    for r in rows:
        if r["env_id"] == "L2_TwoRobotPourKettle-v1":
            pk[r["split"]].append(float(r["SR"]))
    print("L2 PourKettle 행의 SR 최댓값/행 수:", {k: (max(v), len(v)) for k, v in pk.items()})
    OUT["V12"] = {"rows": len(rows), "episodes": n_ep, "split_rows": dict(per_split), "subsr_integer_fraction": {str(k): round(v, 3) for k, v in frac_any.items()},
                  "distinct_subsr": len(set(sub)), "scr_scales": copies, "scr_combos": n_combo, "scr_min_vs_mean": res12,
                  "pourkettle_L2": {k: [max(v), len(v)] for k, v in pk.items()}}
else:
    print("data/site_all_master_records.csv 없음: 건너뜀")

# ================================================================ V10 Scr 민감도
section("V10 Scr가 3회 중 최솟값이면 Δ가 얼마나 커지나 (모형 가정)")
rng = np.random.default_rng(1)
rows = []
for p in (0.1, 0.3, 0.5, 0.8):
    n_trials = 10
    draws = rng.binomial(n_trials, p, size=(200000, 3)) / n_trials
    rows.append((p, round(float(draws.mean()), 3), round(float(draws.min(1).mean()), 3)))
print("(참 성공률 p, 3회 평균, 3회 최솟값의 기댓값) [셀당 10회]:", rows)
OUT["V10"] = {"binomial_min_of_3": rows, "note": "사이트 sim-detail-data.js 머리말이 scr를 '규모별 값 중 최솟값'이라고 적은 것을 전제로 한 모형 계산"}

(HERE / "recompute_results.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("\nwrote recompute_results.json")

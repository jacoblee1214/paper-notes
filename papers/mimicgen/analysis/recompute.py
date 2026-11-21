"""논문 표 수치를 옮겨 적고 (1) 표끼리의 일치, (2) 본문 문장과 표의 일치, (3) 검정과 구간을 다시 계산한다.

입력: 논문 PDF에서 옮긴 표 값(아래 상수). 원문은 paper/paper.txt와 렌더링 이미지로 대조했다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 1초 미만)

가정(논문에 명시되지 않음):
  - "평균 ± 값"의 ± 는 3개 시드의 표본 표준편차로 본다(논문이 정의하지 않음). n = 3.
  - 시뮬 성공률은 한 평가당 롤아웃 50회(부록 O). 체크포인트 수와 시도 수는 알 수 없음.
  - 실로봇 표본 수는 본문에 있음(Stack 243 시도, Coffee 192 시도, 평가 50회).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"
R = {}

# ---------------------------------------------------------------- 1. 표 값 (평균, 표준편차)
# Fig. 4 (왼쪽): 이미지 기반 에이전트, 시드 3개, 최대 체크포인트 성공률.  (src, D0, D1, D2)
FIG4 = {
    "Stack": ((26.0, 1.6), (100.0, 0.0), (99.3, 0.9), None),
    "Stack Three": ((0.7, 0.9), (92.7, 1.9), (86.7, 3.4), None),
    "Square": ((11.3, 0.9), (90.7, 1.9), (73.3, 3.4), (49.3, 2.5)),
    "Threading": ((19.3, 3.4), (98.0, 1.6), (60.7, 2.5), (38.0, 3.3)),
    "Coffee": ((74.0, 4.3), (100.0, 0.0), (90.7, 2.5), (77.3, 0.9)),
    "Three Pc. Assembly": ((1.3, 0.9), (82.0, 1.6), (62.7, 2.5), (13.3, 3.8)),
    "Hammer Cleanup": ((59.3, 5.7), (100.0, 0.0), (62.7, 4.7), None),
    "Mug Cleanup": ((12.7, 2.5), (80.0, 4.9), (64.0, 3.3), None),
    "Kitchen": ((54.7, 8.4), (100.0, 0.0), (76.0, 4.3), None),
    "Nut Assembly": ((0.0, 0.0), (53.3, 1.9), None, None),
    "Pick Place": ((0.0, 0.0), (50.7, 6.6), None, None),
    "Coffee Preparation": ((12.7, 3.4), (97.3, 0.9), (42.0, 0.0), None),
    "Mobile Kitchen": ((2.0, 0.0), (46.7, 18.4), None, None),
    "Nut-and-Bolt Assembly": ((8.7, 2.5), (92.7, 2.5), (81.3, 8.2), (72.7, 4.1)),
    "Gear Assembly": ((14.7, 5.2), (98.7, 1.9), (74.0, 2.8), (56.7, 1.9)),
    "Frame Assembly": ((10.7, 6.8), (82.0, 4.3), (68.7, 3.4), (36.7, 2.5)),
}
# 표 P.1: 데이터 생성 성공률(DGR, %). 시도 수는 주지 않음.  (D0, D1, D2)
DGR_TAB = {
    "Stack": (94.3, 90.0, None), "Stack Three": (71.3, 68.9, None), "Square": (73.7, 48.9, 31.8),
    "Threading": (51.0, 39.2, 21.6), "Coffee": (78.2, 63.5, 27.7), "Three Pc. Assembly": (35.6, 35.5, 31.3),
    "Hammer Cleanup": (47.6, 20.4, None), "Mug Cleanup": (29.5, 17.0, None), "Kitchen": (100.0, 42.7, None),
    "Nut Assembly": (50.0, None, None), "Pick Place": (32.7, None, None),
    "Coffee Preparation": (53.2, 36.1, None), "Mobile Kitchen": (20.7, None, None),
    "Nut-and-Bolt Assembly": (66.0, 59.4, 47.6), "Gear Assembly": (46.9, 8.2, 7.1), "Frame Assembly": (45.3, 32.7, 28.9),
}
# 표 F.1 / F.2 (이미지): 로봇 팔별. 순서: Panda, Sawyer, IIWA, UR5e
F1_DGR = {"Square D0": (73.7, 55.8, 37.7, 64.7), "Square D1": (48.9, 38.8, 26.5, 34.1),
          "Threading D0": (51.0, 28.8, 20.4, 21.4), "Threading D1": (39.2, 23.7, 11.5, 18.5)}
F2_SR = {"Square D0": ((90.7, 1.9), (86.0, 1.6), (80.0, 4.3), (84.7, 0.9)),
         "Square D1": ((73.3, 3.4), (60.7, 2.5), (48.0, 3.3), (56.0, 4.3)),
         "Threading D0": ((98.0, 1.6), (88.7, 7.5), (94.0, 3.3), (91.3, 0.9)),
         "Threading D1": ((60.7, 2.5), (50.7, 3.8), (49.3, 4.1), (60.7, 2.5))}
# 표 Q.1 (저차원 관측, 시드 3개). 과제별 (src, D0, D1, D2)
Q1 = {
    "Stack": ((38.7, 4.1), (100.0, 0.0), (100.0, 0.0), None),
    "Stack Three": ((2.7, 0.9), (88.0, 1.6), (90.7, 0.9), None),
    "Square": ((18.7, 0.9), (98.0, 1.6), (80.7, 3.4), (58.7, 1.9)),
    "Threading": ((9.3, 2.5), (97.3, 0.9), (72.0, 1.6), (60.7, 6.2)),
    "Coffee": ((42.7, 4.1), (100.0, 0.0), (93.3, 2.5), (76.7, 0.9)),
    "Three Pc. Assembly": ((2.7, 0.9), (74.7, 3.8), (61.3, 1.9), (38.7, 4.1)),
    "Hammer Cleanup": ((64.7, 4.1), (100.0, 0.0), (74.0, 1.6), None),
    "Mug Cleanup": ((8.0, 1.6), (82.0, 2.8), (54.7, 5.0), None),
    "Kitchen": ((43.3, 3.4), (100.0, 0.0), (78.0, 2.8), None),
    "Nut Assembly": ((0.0, 0.0), (76.0, 1.6), None, None),
    "Pick Place": ((0.0, 0.0), (58.7, 7.5), None, None),
    "Coffee Preparation": ((2.0, 0.0), (76.0, 5.7), (59.3, 3.4), None),
    "Mobile Kitchen": ((6.7, 3.8), (76.7, 10.5), None, None),
    "Nut-and-Bolt Assembly": ((2.0, 0.0), (98.0, 1.6), (96.0, 1.6), (81.3, 3.8)),
    "Gear Assembly": ((12.0, 1.6), (92.7, 1.9), (76.0, 4.9), (64.0, 3.3)),
    "Frame Assembly": ((9.3, 3.4), (87.3, 2.5), (70.7, 1.9), (58.0, 5.7)),
}

# ---------------------------------------------------------------- 2. 도움 함수
def welch(a, b):
    """두 (평균, 표준편차, n=3) 비교. 양측 Welch t-test."""
    t, p = stats.ttest_ind_from_stats(a[0], a[1], 3, b[0], b[1], 3, equal_var=False)
    diff = a[0] - b[0]
    return {"diff_pp": round(diff, 2), "t": round(float(t), 2), "p": float(p)}


def wilson(k, n, z=1.959964):
    """이항 비율의 Wilson 95% 구간 (%)."""
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def binom_se(p_pct, n=50):
    """평가 한 번(롤아웃 n회)의 이항 표준오차 (%p)."""
    p = p_pct / 100
    return round(100 * math.sqrt(p * (1 - p) / n), 2)


# ---------------------------------------------------------------- 3. 표끼리의 일치 검사
checks = []

def chk(name, a, b):
    checks.append({"check": name, "a": a, "b": b, "match": bool(abs(a - b) < 1e-9)})

# Fig. 4 의 D0 와 Table F.2 Panda, Table N.1 normal, Table U.1 None, Table I.1 Op A, Table G.1 D0
for task, d0, d1 in [("Square", 90.7, 73.3), ("Threading", 98.0, 60.7)]:
    chk(f"Fig4 {task} D0 = F.2 Panda D0", FIG4[task][1][0], d0)
    chk(f"Fig4 {task} D1 = F.2 Panda D1", FIG4[task][2][0], d1)
chk("Fig4 Mug D0 = G.1 D0 (image)", FIG4["Mug Cleanup"][1][0], 80.0)
chk("Fig4 Stack D1 = H.1 text 99.3", FIG4["Stack"][2][0], 99.3)
chk("Fig4 Stack Three D1 = U.1 None (86.7)", FIG4["Stack Three"][2][0], 86.7)
chk("Fig4 Square D2 = U.1 Square D2 None (49.3)", FIG4["Square"][3][0], 49.3)
chk("Fig4 Coffee D1 = U.1 Coffee D1 None (90.7)", FIG4["Coffee"][2][0], 90.7)
chk("Fig4 Coffee D0 = U.1 Coffee D0 None (100.0)", FIG4["Coffee"][1][0], 100.0)
chk("Fig4 Stack Three D0 = I.1 OpA image (92.7)", FIG4["Stack Three"][1][0], 92.7)
chk("Fig4 TPA D0 = I.1 OpA image (82.0)", FIG4["Three Pc. Assembly"][1][0], 82.0)
# 저차원 Q.1 과 저차원 부록 표들
chk("Q1 Square D0 = F.3 Panda D0", Q1["Square"][1][0], 98.0)
chk("Q1 Threading D0 = F.3 Panda D0", Q1["Threading"][1][0], 97.3)
chk("Q1 Threading D1 = F.3 Panda D1", Q1["Threading"][2][0], 72.0)
chk("Q1 Square D1 = F.3 Panda D1", Q1["Square"][2][0], 80.7)
chk("Q1 Stack D0 = N.2 normal (100.0)", Q1["Stack"][1][0], 100.0)
chk("Q1 Stack Three D0 = N.2 normal (88.0)", Q1["Stack Three"][1][0], 88.0)
chk("Q1 Pick Place D0 = N.2 normal (58.7)", Q1["Pick Place"][1][0], 58.7)
chk("Q1 Nut Assembly D0 = N.2 normal (76.0)", Q1["Nut Assembly"][1][0], 76.0)
chk("Q1 Stack Three D1 = I.1 OpA low-dim (90.7)", Q1["Stack Three"][2][0], 90.7)
chk("Q1 Mobile Kitchen source vs text 2.7 (본문: 2.7%)", Q1["Mobile Kitchen"][0][0], 2.7)
# DGR: P.1 와 F.1, N.1, N.2, T.1, U.1
chk("표 P.1 Square D0 = F.1 Panda", DGR_TAB["Square"][0], F1_DGR["Square D0"][0])
chk("표 P.1 Square D1 = F.1 Panda", DGR_TAB["Square"][1], F1_DGR["Square D1"][0])
chk("표 P.1 Threading D0 = F.1 Panda", DGR_TAB["Threading"][0], F1_DGR["Threading D0"][0])
chk("표 P.1 Threading D1 = F.1 Panda", DGR_TAB["Threading"][1], F1_DGR["Threading D1"][0])
chk("표 P.1 Square D2 = U.1 None (31.8)", DGR_TAB["Square"][2], 31.8)
chk("표 P.1 Coffee D0 = U.1 None (78.2)", DGR_TAB["Coffee"][0], 78.2)
chk("표 P.1 Mug D0 = G.1 DGR (29.5)", DGR_TAB["Mug Cleanup"][0], 29.5)
chk("표 P.1 Stack D0 = N.2 normal (94.3)", DGR_TAB["Stack"][0], 94.3)
chk("표 P.1 Stack Three D1 = U.1 None (68.9)", DGR_TAB["Stack Three"][1], 68.9)
R["table_checks"] = checks
R["table_checks_mismatch"] = [c for c in checks if not c["match"]]

# ---------------------------------------------------------------- 4. 본문 문장 검사
text = {}
text["6.1 Square 11.3 -> 90.7"] = (FIG4["Square"][0][0], FIG4["Square"][1][0])
text["6.1 Threading 19.3 -> 98.0"] = (FIG4["Threading"][0][0], FIG4["Threading"][1][0])
text["6.1 TPA 1.3 -> 82.0"] = (FIG4["Three Pc. Assembly"][0][0], FIG4["Three Pc. Assembly"][1][0])
text["6.1 Mobile 2.0 -> 46.7 (image)"] = (FIG4["Mobile Kitchen"][0][0], FIG4["Mobile Kitchen"][1][0])
text["6.1 Mobile 2.7 -> 76.7 (low-dim, 본문)"] = (2.7, Q1["Mobile Kitchen"][1][0])
text["6.1 Mobile 표 Q.1 source 6.7 -> 76.7"] = (Q1["Mobile Kitchen"][0][0], Q1["Mobile Kitchen"][1][0])
R["text_checks"] = {k: {"source": v[0], "agent": v[1]} for k, v in text.items()}

# 6.1 Factory: D0 82-99, source 9-15, D1/D2 37-81
fac = ["Nut-and-Bolt Assembly", "Gear Assembly", "Frame Assembly"]
R["factory_ranges"] = {
    "D0_image": [min(FIG4[t][1][0] for t in fac), max(FIG4[t][1][0] for t in fac)],
    "src_image": [min(FIG4[t][0][0] for t in fac), max(FIG4[t][0][0] for t in fac)],
    "D1D2_image": [min(v[0] for t in fac for v in (FIG4[t][2], FIG4[t][3]) if v), max(v[0] for t in fac for v in (FIG4[t][2], FIG4[t][3]) if v)],
}
# 6.1 D1 범위 "42% to 99%"
d1 = [v[2][0] for v in FIG4.values() if v[2]]
R["D1_range_image"] = [min(d1), max(d1)]

# 6.3 과 부록 P 에서 든 "생성률 대 정책 성능" 예시: 본문은 과제 이름과 저차원/이미지 기준을 섞어 씀
ex = [("Mug Cleanup (본문 'Object Cleanup') D0", DGR_TAB["Mug Cleanup"][0], Q1["Mug Cleanup"][1][0], FIG4["Mug Cleanup"][1][0]),
      ("Three Pc. Assembly D0", DGR_TAB["Three Pc. Assembly"][0], Q1["Three Pc. Assembly"][1][0], FIG4["Three Pc. Assembly"][1][0]),
      ("Coffee D2", DGR_TAB["Coffee"][2], Q1["Coffee"][3][0], FIG4["Coffee"][3][0]),
      ("Gear Assembly D1", DGR_TAB["Gear Assembly"][1], Q1["Gear Assembly"][2][0], FIG4["Gear Assembly"][2][0])]
R["sec63_examples"] = [{"case": c, "DGR": g, "text_agent_value": t, "image_value": i} for c, g, t, i in ex]

# 부록 P: "Gear D0 46.9, D1 8.2, D2 7.1 / 정책 92.7, 76.0, 64.0" -> 저차원 값
R["appP_gear_lowdim_vs_image"] = {"text": [92.7, 76.0, 64.0], "lowdim_Q1": [Q1["Gear Assembly"][1][0], Q1["Gear Assembly"][2][0], Q1["Gear Assembly"][3][0]],
                                  "image_Fig4": [FIG4["Gear Assembly"][1][0], FIG4["Gear Assembly"][2][0], FIG4["Gear Assembly"][3][0]]}

# ---------------------------------------------------------------- 5. 부록 T: 여러 시드 DGR 편차 주장 "0.6% 이하"
T1_DGR = {"Stack Three D0": (71.7, 71.3), "Stack Three D1": (69.3, 68.9), "Square D0": (74.4, 73.7),
          "Square D1": (48.5, 48.9), "Square D2": (32.0, 31.8)}
T1_SR = {"Stack Three D0": (89.6, 88.0), "Stack Three D1": (92.4, 90.7), "Square D0": (96.7, 98.0),
         "Square D1": (81.6, 80.7), "Square D2": (58.0, 58.7)}
R["appT_dgr_dev"] = {k: round(a - b, 2) for k, (a, b) in T1_DGR.items()}
R["appT_sr_dev"] = {k: round(a - b, 2) for k, (a, b) in T1_SR.items()}
R["appT_claim"] = {"text_dgr_max_dev_pp": 0.6, "text_sr_max_dev_pp": 2.0,
                   "dgr_violations": [k for k, (a, b) in T1_DGR.items() if abs(a - b) > 0.6],
                   "sr_violations": [k for k, (a, b) in T1_SR.items() if abs(a - b) >= 2.0]}

# ---------------------------------------------------------------- 6. 부록 I, I.2 의 "편차" 주장
# 표 I.1: (작업자 A 값, 작업자 B 값). 이미지 기반과 저차원
I1_IMG = {"Stack Three D0": (92.7, 86.0), "Stack Three D1": (86.7, 69.3),
          "Threading D0": (98.0, 98.0), "Threading D1": (60.7, 58.0), "Threading D2": (38.0, 38.0),
          "TPA D0": (82.0, 76.0), "TPA D1": (62.7, 54.7), "TPA D2": (13.3, 5.3)}
I1_LOW = {"Stack Three D0": (88.0, 82.7), "Stack Three D1": (90.7, 84.0),
          "Threading D0": (97.3, 97.3), "Threading D1": (72.0, 76.0), "Threading D2": (60.7, 70.0),
          "TPA D0": (74.7, 77.3), "TPA D1": (61.3, 65.3), "TPA D2": (38.7, 46.0)}
R["appI1_max_dev_image"] = round(max(abs(a - b) for a, b in I1_IMG.values()), 1)
R["appI1_max_dev_lowdim"] = round(max(abs(a - b) for a, b in I1_LOW.values()), 1)
R["appI1_max_dev_image_cell"] = max(I1_IMG, key=lambda k: abs(I1_IMG[k][0] - I1_IMG[k][1]))
I2_sq = {"Square D0 image": (90.7, 90.0, 90.7), "Square D1 image": (73.3, 64.0, 59.3), "Square D2 image": (49.3, 50.0, 45.3),
         "Square D1 low": (80.7, 82.0, 76.7), "Square D2 low": (58.7, 60.7, 52.7), "Square D0 low": (98.0, 95.3, 95.3)}
R["appI2_max_dev"] = round(max(max(abs(v[0] - v[1]), abs(v[0] - v[2])) for v in I2_sq.values()), 1)
# 통계: 같은 과제에서 최선 대 최악 데모 제공자 (D1 image) 차이가 유의한지
R["appI2_D1_image_better_vs_worse"] = welch((73.3, 3.4), (59.3, 2.5))
R["appI2_D1_image_better_vs_okay"] = welch((73.3, 3.4), (64.0, 7.1))

# ---------------------------------------------------------------- 7. 로봇 전이(부록 F): "매우 비슷하다" 주장 검정
rt = {}
arms = ["Sawyer", "IIWA", "UR5e"]
for cell, vals in F2_SR.items():
    panda = vals[0]
    rt[cell] = {arm: welch(panda, vals[i + 1]) for i, arm in enumerate(arms)}
    rt[cell]["range_all_arms"] = [min(v[0] for v in vals), max(v[0] for v in vals)]
R["robot_transfer_welch_vs_panda"] = rt
R["robot_transfer_DGR_range_D0"] = [min(F1_DGR["Square D0"] + F1_DGR["Threading D0"]), max(F1_DGR["Square D0"] + F1_DGR["Threading D0"])]

# ---------------------------------------------------------------- 8. 부록 N.1 "노이즈 제거" 와 "재생 + 노이즈"
R["appN1_noise_vs_nonoise_square_D0"] = welch((90.7, 1.9), (72.0, 3.3))
R["appN1_noise_vs_nonoise_threading_D0_lowdim"] = welch((97.3, 0.9), (69.3, 0.9))
R["appN1_threading_dgr_increase_pp"] = round(84.5 - 51.0, 1)
R["appN1_replay_vs_mg_square_D0"] = welch((90.7, 1.9), (42.0, 1.6))
R["appN1_replay_vs_mg_threading_D0"] = welch((98.0, 1.6), (74.0, 3.3))

# ---------------------------------------------------------------- 9. 부록 S: 더 다양한 원천 데모(D2 원천) 비교
R["appS_tpa_src_D2_vs_D0_lowdim_D0"] = welch((74.7, 3.8), (62.0, 4.9))
R["appS_square_src_D2_vs_D0_lowdim_D1"] = welch((80.7, 3.4), (84.7, 1.9))

# 표 N.2: 최근접 이웃 선택 제거 (저차원 SR, Square D0)
R["appN2_square_D0_nn_vs_random_lowdim"] = welch((98.0, 1.6), (94.7, 2.5))
R["appN2_square_D0_dgr_nn_vs_random"] = {"normal": 73.7, "no_nn": 36.7}

# ---------------------------------------------------------------- 10. 부록 G: O1 대 D0
R["appG_O1_vs_D0_image"] = welch((90.7, 1.9), (80.0, 4.9))
R["appG_O2_vs_D0_image"] = welch((75.3, 5.2), (80.0, 4.9))

# ---------------------------------------------------------------- 11. 실로봇 (본문 6.4, 부록 H)
real = {}
real["Stack DGR 200/243"] = {"rate": round(100 * 200 / 243, 1), "wilson95": wilson(200, 243), "attempts": 243}
real["Coffee DGR 100/192"] = {"rate": round(100 * 100 / 192, 1), "wilson95": wilson(100, 192), "attempts": 192}
real["Stack BC-RNN 18/50"] = {"rate": 36.0, "wilson95": wilson(18, 50)}
real["Stack DiffPolicy 38/50"] = {"rate": 76.0, "wilson95": wilson(38, 50)}
real["Stack source 0/50"] = {"rate": 0.0, "wilson95": wilson(0, 50)}
real["Coffee policy 7/50"] = {"rate": 14.0, "wilson95": wilson(7, 50)}
real["Coffee pod insertion 10/50"] = {"rate": 20.0, "wilson95": wilson(10, 50)}
real["Coffee pod grasp 30/50"] = {"rate": 60.0, "wilson95": wilson(30, 50)}
R["real_robot"] = real
R["real_fisher_stack_bc_vs_dp"] = {"p": float(stats.fisher_exact([[18, 32], [38, 12]])[1])}
R["real_fisher_stack_bc_vs_source"] = {"p": float(stats.fisher_exact([[18, 32], [0, 50]])[1])}
R["real_fisher_coffee_vs_source"] = {"p": float(stats.fisher_exact([[7, 43], [0, 50]])[1])}
# 본문 "DP 76% 대 BC 36%" 의 검정력: 시도 50회에서 차이를 80% 검정력으로 보려면 대략 필요한 평가 수
def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)
R["real_power_n_per_arm_for_36_vs_76"] = n_per_arm(0.36, 0.76)
R["real_power_n_per_arm_for_14_vs_0"] = n_per_arm(0.14, 0.02)
# 실로봇 DGR 과 원 시뮬의 같은 과제 DGR 비교 없음(시뮬 DGR 시도 수 미공개)

# ---------------------------------------------------------------- 12. 평가 한 번의 이항 표준오차 대 보고된 ±
bin_rows = {}
for task, vals in FIG4.items():
    for lab, v in zip(["D0", "D1", "D2"], vals[1:]):
        if v:
            bin_rows[f"{task} {lab}"] = {"mean": v[0], "reported_sd_3seeds": v[1], "binomial_se_one_eval_n50": binom_se(v[0])}
R["binomial_vs_reported_sd"] = bin_rows
sds = [v["reported_sd_3seeds"] for v in bin_rows.values()]
bse = [v["binomial_se_one_eval_n50"] for v in bin_rows.values()]
R["binomial_vs_reported_sd_summary"] = {
    "median_reported_sd": float(np.median(sds)),
    "median_binomial_se_one_eval": float(np.median(bse)),
    "share_cells_reported_sd_below_binomial_se": round(float(np.mean([s < b for s, b in zip(sds, bse)])), 2),
    "n_cells": len(sds),
}

# ---------------------------------------------------------------- 13. DGR 과 정책 성공률의 상관 (본문 6.3 "꼭 같지 않다")
tasks = [t for t in DGR_TAB if FIG4[t][1] is not None and DGR_TAB[t][0] is not None]
dgr0 = [DGR_TAB[t][0] for t in tasks]
sr0 = [FIG4[t][1][0] for t in tasks]
rho0, p0 = stats.spearmanr(dgr0, sr0)
pairs = []
for t in DGR_TAB:
    for i, lab in enumerate(["D0", "D1", "D2"]):
        if DGR_TAB[t][i] is not None and FIG4[t][i + 1] is not None:
            pairs.append((DGR_TAB[t][i], FIG4[t][i + 1][0]))
rho_all, p_all = stats.spearmanr([a for a, _ in pairs], [b for _, b in pairs])
R["dgr_vs_sr_spearman"] = {"D0_n_tasks": len(tasks), "D0_rho": round(float(rho0), 3), "D0_p": float(p0),
                           "all_variants_n": len(pairs), "all_rho": round(float(rho_all), 3), "all_p": float(p_all)}

# ---------------------------------------------------------------- 14. 데이터 규모: 1000개 목표와 시도 수 추정
main_variants = {t: sum(1 for x in DGR_TAB[t] if x is not None) for t in DGR_TAB}
n_main = sum(main_variants.values())
attempts = {}
for t in DGR_TAB:
    for i, lab in enumerate(["D0", "D1", "D2"]):
        if DGR_TAB[t][i]:
            attempts[f"{t} {lab}"] = round(1000 / (DGR_TAB[t][i] / 100))
R["scale"] = {"main_task_variants_in_dgr_table": n_main, "demos_at_1000_each": n_main * 1000,
              "variants_in_G1_extra": 2, "total_expected_attempts_main": int(sum(attempts.values())),
              "max_attempts_variant": max(attempts, key=attempts.get), "max_attempts": max(attempts.values()),
              "tasks_in_dgr_table": len(DGR_TAB), "tasks_claimed_in_abstract": 18, "tasks_in_repo_configs": 12}

# 그림용 값 (plot_figures.py가 읽음)
R["plot_dgr_sr_D0"] = {t: [DGR_TAB[t][0], FIG4[t][1][0]] for t in tasks}
R["plot_robot_transfer"] = {cell: {arm: list(v) for arm, v in zip(["Panda", "Sawyer", "IIWA", "UR5e"], vals)} for cell, vals in F2_SR.items()}

# ---------------------------------------------------------------- 저장과 요약
OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print("표 일치 검사:", len(checks), "개, 불일치:", [c["check"] for c in R["table_checks_mismatch"]])
print("부록 T 주장 위반:", R["appT_claim"]["dgr_violations"], R["appT_claim"]["sr_violations"])
print("부록 I.1 최대 편차 image/low:", R["appI1_max_dev_image"], R["appI1_max_dev_lowdim"], "I.2 최대:", R["appI2_max_dev"])
print("실로봇 BC 대 DP Fisher p:", R["real_fisher_stack_bc_vs_dp"]["p"])
print("Spearman D0:", R["dgr_vs_sr_spearman"])
print("규모:", R["scale"])
print("기록:", OUT)

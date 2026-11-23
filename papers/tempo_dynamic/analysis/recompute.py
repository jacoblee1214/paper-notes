"""TEMPO (arXiv:2609.16864 v1) 논문 표 수치를 다시 계산하고 검정한다.

입력: 논문 본문, 표, 그림의 막대 숫자를 아래에 그대로 옮겨 적었다(출처는 주석). 외부 데이터는 쓰지 않는다.
      TempoBench 수치는 공개 저장소 data/manifest.json과 각 과제 meta/stats.json의 값을 옮겨 적었다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, statsmodels, 수 초)

가정(논문에 없는 것은 표시한다)
- 성공률 분모는 과제당 50회다(본문 5.2절 "50 trials", 표 4 캡션, 부록 C 그림과 본문 그림 5의 막대 합이 모두 50).
- Wine Pour 열은 비율이 아니라 비즈 질량 보존율이라 분모가 없다. 표준편차도 보고되지 않아 검정하지 않는다.
- 그림 5의 막대에서 TEMPO와 RTC는 범주 4개 중 3개 숫자만 적혀 있다. 빠진 범주는 0으로 읽는다.
"""
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}
N = 50


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def cmp(name, k1, k2, n1=N, n2=N):
    return {"비교": name, "k1": k1, "k2": k2, "n": n1,
            "차이_pp": round(100 * (k1 / n1 - k2 / n2), 1), "fisher_p": round(fisher(k1, n1, k2, n2), 5)}


# ---------------------------------------------------------------- 표 4 (그림 5와 부록 C 막대의 성공 수와 맞춤)
T4 = {  # 과제: {방법: 성공률 %}  (Table 4)
    "Bottle Handover": {"RTC": 44, "VLASH": 38, "TEMPO": 74, "TEMPO+UVT": 88},
    "Drop Catch": {"RTC": 80, "VLASH": 38, "TEMPO": 66, "TEMPO+UVT": 88},
    "Flick Catch": {"RTC": 48, "VLASH": 20, "TEMPO": 66, "TEMPO+UVT": 76},
    "Wine Pour": {"RTC": 80.8, "VLASH": 94.0, "TEMPO": 98.1, "TEMPO+UVT": 96.7},
}
STATED_AVG = {"RTC": 63.2, "VLASH": 47.5, "TEMPO": 76.0, "TEMPO+UVT": 87.2}
# 그림 5 (Handover), 부록 C의 Drop 그림, Flick 그림: 방법별 막대 숫자(왼쪽부터). 본문·프로젝트 페이지로 범주를 확정한 것만 이름을 붙인다.
BARS = {
    "Bottle Handover": {"RTC": [22, 4, 24], "VLASH": [19, 5, 8, 18], "TEMPO": [37, 3, 10]},
    "Drop Catch": {"RTC": [40, 6, 4], "VLASH": [19, 2, 6, 23], "TEMPO": [33, 4, 7, 6]},
    "Flick Catch": {"RTC": [24, 19, 7], "VLASH": [10, 37, 3], "TEMPO": [33, 4, 9, 4]},
}

V = {}

# V1: 표 4 평균 열
rows = []
for m in ["RTC", "VLASH", "TEMPO", "TEMPO+UVT"]:
    vals = [T4[t][m] for t in T4]
    avg = sum(vals) / 4
    rows.append({"방법": m, "재계산_평균": round(avg, 3), "논문_평균": STATED_AVG[m],
                 "일치": abs(round(avg, 1) - STATED_AVG[m]) < 0.051 or abs(avg - STATED_AVG[m]) < 0.06})
V["V1_표4_평균"] = rows

# V2: 정수 가능성과 막대 합
feas = []
for t in ["Bottle Handover", "Drop Catch", "Flick Catch"]:
    for m in ["RTC", "VLASH", "TEMPO", "TEMPO+UVT"]:
        pct = T4[t][m]
        k = pct * N / 100
        feas.append({"과제": t, "방법": m, "값": pct, "n=50 성공 수": k, "정수": float(k).is_integer()})
V["V2_정수_가능성"] = feas
bar_rows = []
for t, d in BARS.items():
    for m, b in d.items():
        bar_rows.append({"과제": t, "방법": m, "막대": b, "합": sum(b), "첫 막대(성공)": b[0],
                         "성공률 %": 100 * b[0] / N, "표 4": T4[t][m], "표와 일치": 100 * b[0] / N == T4[t][m]})
V["V2b_막대_합과_표4"] = bar_rows

# V3: Wilson 95% 구간 (n=50 가정)
wil = []
for t in ["Bottle Handover", "Drop Catch", "Flick Catch"]:
    for m in ["RTC", "VLASH", "TEMPO", "TEMPO+UVT"]:
        k = round(T4[t][m] * N / 100)
        lo, hi = wilson(k, N)
        wil.append({"과제": t, "방법": m, "k": k, "wilson95_%": [round(100 * lo, 1), round(100 * hi, 1)]})
V["V3_wilson"] = wil

# V4: 핵심 비교 Fisher + Holm
core = []
for t in ["Bottle Handover", "Drop Catch", "Flick Catch"]:
    kt = round(T4[t]["TEMPO"] * N / 100)
    for b in ["RTC", "VLASH"]:
        core.append(cmp(f"{t}: TEMPO 대 {b}", kt, round(T4[t][b] * N / 100)))
ps = [c["fisher_p"] for c in core]
holm = multipletests(ps, method="holm")[1]
for c, h in zip(core, holm):
    c["holm_p"] = round(float(h), 5)
V["V4_핵심비교_6개"] = core

uvt = []
for t in ["Bottle Handover", "Drop Catch", "Flick Catch"]:
    ku = round(T4[t]["TEMPO+UVT"] * N / 100)
    uvt.append(cmp(f"{t}: TEMPO+UVT 대 TEMPO", ku, round(T4[t]["TEMPO"] * N / 100)))
    uvt.append(cmp(f"{t}: TEMPO+UVT 대 RTC", ku, round(T4[t]["RTC"] * N / 100)))
V["V4b_UVT_비교"] = uvt

# V5: 세 과제 합산(CMH) — TEMPO 대 각 기준선
cmh = {}
for b in ["RTC", "VLASH"]:
    tabs = []
    for t in ["Bottle Handover", "Drop Catch", "Flick Catch"]:
        kt = round(T4[t]["TEMPO"] * N / 100)
        kb = round(T4[t][b] * N / 100)
        tabs.append(np.array([[kt, N - kt], [kb, N - kb]]))
    st = StratifiedTable(tabs)
    t = st.test_null_odds(correction=False)
    cmh[b] = {"합산 성공": [int(sum(x[0, 0] for x in tabs)), int(sum(x[1, 0] for x in tabs)), 150],
              "MH_odds": round(float(st.oddsratio_pooled), 2), "CMH_p": float(f"{t.pvalue:.3g}")}
V["V5_세과제_CMH"] = cmh

# V6: 본문 5.2절의 "+36%, +4.1%, +46%"가 어느 기준선에 대한 값인지
V["V6_본문_차이의_기준선"] = {
    "+36% Bottle Handover": {"대 VLASH": 74 - 38, "대 RTC": 74 - 44},
    "+4.1% Wine Pour": {"대 VLASH": round(98.1 - 94.0, 1), "대 RTC": round(98.1 - 80.8, 1)},
    "+46% Flick Catch": {"대 VLASH": 66 - 20, "대 RTC": 66 - 48},
    "Drop Catch 대 RTC": 66 - 80,
}
# TEMPO+UVT "3 of 4 tasks outperform baselines"
cnt = {}
for b in ["RTC", "VLASH"]:
    cnt[b] = [t for t in T4 if T4[t]["TEMPO+UVT"] > T4[t][b]]
V["V6b_UVT가_기준선_위인_과제"] = {"대 RTC": cnt["RTC"], "대 VLASH": cnt["VLASH"],
                              "TEMPO보다 높은 과제": [t for t in T4 if T4[t]["TEMPO+UVT"] > T4[t]["TEMPO"]]}

# V7: 실패 유형 막대 비교 (Fisher)
V["V7_실패유형"] = [
    cmp("Handover 완전 실패: TEMPO 10 대 VLASH 18", 10, 18),
    cmp("Handover 완전 실패: TEMPO 10 대 RTC 24", 10, 24),
    cmp("Drop 조기 이동: TEMPO 6 대 VLASH 23", 6, 23),
    cmp("Flick 손 추적 실패: TEMPO 9 대 VLASH 37", 9, 37),
    cmp("Flick 공 잡기 실패: TEMPO 4 대 RTC 19", 4, 19),
]
V["V7b_Drop_TEMPO_대_RTC"] = cmp("Drop Catch 성공: TEMPO 33 대 RTC 40", 33, 40)

# V8: 표 1(미절단) — Flick 34/50 대 0/50
lo, hi = wilson(34, 50)
V["V8_표1_미절단"] = {
    "Flick Catch TEMPO 34/50": {"wilson95_%": [round(100 * lo, 1), round(100 * hi, 1)],
                                "대 0/50 Fisher p": float(f"{fisher(34, 50, 0, 50):.3g}")},
    "트림 대 미절단 TEMPO Flick 33/50 대 34/50": round(fisher(33, 50, 34, 50), 3),
    "Wine Pour 트림 98.1 대 미절단 97.6 (질량 %)": round(98.1 - 97.6, 1),
    "참고": "0/50은 성공 수. 0% 두 칸의 Wilson 상한",
    "0/50 Wilson 상한 %": round(100 * wilson(0, 50)[1], 1),
}

# V9: 표 5(백본), 표 6(절제), 표 2(인코더) — 핸들오버 50회 가정
bb = [("π0.5", 26, 56), ("RTC", 44, 80), ("VLASH", 38, 74)]
t5 = []
for name, b, w in bb:
    kb, kw = round(b / 2), round(w / 2)
    t5.append(cmp(f"{name}: +TEMPO {kw}/50 대 기준 {kb}/50", kw, kb))
V["V9_표5_백본"] = t5
V["V9b_표5_RTC+TEMPO_대_VLASH+TEMPO"] = cmp("Handover: RTC+TEMPO 40/50 대 VLASH+TEMPO 37/50", 40, 37)
abl = [cmp("표 6: TEMPO 전체 37/50 대 ACT만 26/50", 37, 26),
       cmp("표 6: TEMPO 전체 37/50 대 MOT만 28/50", 37, 28),
       cmp("표 6: ACT만 26/50 대 VLASH 19/50", 26, 19),
       cmp("표 6: MOT만 28/50 대 VLASH 19/50", 28, 19)]
V["V10_표6_절제"] = abl
enc = [("SAM 2.1-tiny", 74), ("Reducio", 76), ("VidTwin", 60), ("VideoLaVIT", 52)]
V["V11_표2_인코더"] = [cmp(f"{n} {p}% 대 SAM 74%", round(p / 2), 37) for n, p in enc[1:]]
V["V11b_표2_인코더_대_VLASH"] = [cmp(f"{n} {p}% 대 VLASH 38%", round(p / 2), 19) for n, p in enc]

# V12: 표 7 평균과 인코더 프로브(부록 E 그림) 순위 상관
t7 = {"RTC": [0.46, 0.31, -0.07, 0.06], "VLASH": [0.43, 0.28, 0.03, 0.09], "TEMPO": [0.52, 0.33, 0.57, 0.44]}
st7 = {"RTC": 0.19, "VLASH": 0.21, "TEMPO": 0.47}
V["V12_표7_평균"] = [{"방법": m, "재계산": round(float(np.mean(v)), 4), "논문": st7[m]} for m, v in t7.items()]
a5 = {"Object Position": [0.24, 0.03, 0.03, 0.02], "VideoLaVIT": [0.45, 0.25, 0.28, 0.36],
      "Reducio": [0.50, 0.36, 0.38, 0.32], "SAM 2": [0.58, 0.37, 0.50, 0.39], "VidTwin": [0.59, 0.44, 0.43, 0.45]}
st_a5 = {"Object Position": 0.08, "VideoLaVIT": 0.34, "Reducio": 0.39, "SAM 2": 0.46, "VidTwin": 0.48}
V["V13_그림A5_평균"] = [{"인코더": m, "재계산": round(float(np.mean(v)), 4), "논문": st_a5[m]} for m, v in a5.items()]
succ = {"VideoLaVIT": 52, "Reducio": 76, "SAM 2": 74, "VidTwin": 60}  # 표 2
names = list(succ)
probe_avg = [float(np.mean(a5[n])) for n in names]
probe_h = [a5[n][0] for n in names]
sv = [succ[n] for n in names]
V["V14_프로브_대_과제성공"] = {
    "인코더": names, "표 2 Handover 성공률": sv,
    "부록E 평균 r2": [round(x, 3) for x in probe_avg], "부록E Handover r2": probe_h,
    "스피어먼 rho(평균 r2, 성공률)": round(float(stats.spearmanr(probe_avg, sv)[0]), 2),
    "스피어먼 rho(Handover r2, 성공률)": round(float(stats.spearmanr(probe_h, sv)[0]), 2),
    "해석": "인코더 4개뿐이라 p 값은 의미 없음. 순위 일치가 약하다는 관찰만 남긴다.",
}
# TEMPO가 표 7에서 가장 높은 행: 전 행에서 최대인가
V["V15_표7_행별_최대"] = {t: ("TEMPO" if t7["TEMPO"][i] == max(t7[m][i] for m in t7) else "아님")
                       for i, t in enumerate(["Handover", "Drop", "Flick", "Wine"])}
V["V15b_표7_Drop_격차"] = {"TEMPO-RTC": round(0.33 - 0.31, 2), "TEMPO-VLASH": round(0.33 - 0.28, 2)}

# V16: 검출 가능 최소 차이 (Fisher 양측 α = 0.05, 검정력 80%, n = 50 대 50)
def fisher_p_matrix(n):
    P = np.zeros((n + 1, n + 1))
    for a in range(n + 1):
        for b in range(n + 1):
            P[a, b] = fisher(a, n, b, n)
    return P


P = fisher_p_matrix(N)
ks = np.arange(N + 1)
sig = P < 0.05


def power(p1, p2):
    w1 = stats.binom.pmf(ks, N, p1)
    w2 = stats.binom.pmf(ks, N, p2)
    return float((w1[:, None] * w2[None, :] * sig).sum())


mde = []
for base in (0.2, 0.4, 0.5, 0.8):
    d = next((round(x, 2) for x in np.arange(0.02, 1 - base + 1e-9, 0.01) if power(base + x, base) >= 0.8), None)
    mde.append({"기준 성공률": base, "검정력 80% 최소 차이": d})
V["V16_검출가능_차이_n50"] = mde

# V17: Wine Pour 질량 보존의 손실 환산
V["V17_WinePour_손실"] = {"RTC": round(100 - 80.8, 1), "VLASH": round(100 - 94.0, 1), "TEMPO": round(100 - 98.1, 1),
                          "TEMPO+UVT": round(100 - 96.7, 1), "TEMPO 미절단": round(100 - 97.6, 1),
                          "VLASH 손실 / TEMPO 손실": round(6.0 / 1.9, 2),
                          "해석": "보존율 4.1p 차이는 손실로 보면 6.0% 대 1.9%. 표준편차와 시행 수 보고가 없어 검정 불가."}

# V18: 비용·학습 수치 (논문 4.4절, 부록 B.3, 서문)
V["V18_비용_일관성"] = {
    "지연 차이 ms": round(35.3 - 33.8, 1), "상대 증가 %": round(100 * (35.3 - 33.8) / 33.8, 1),
    "RTC 대비 TEMPO 지연 비": round(127.7 / 35.3, 2),
    "제어 주기 30 Hz의 한 스텝 ms": round(1000 / 30, 1),
    "2M, 0.08%가 뜻하는 기반 파라미터 수(억)": round(2e6 / 0.0008 / 1e8, 1),
    "학습 시간 2 h, 0.50 s/iter에서의 반복 수": round(2 * 3600 / 0.50),
    "공개 설정 10k+5k 스텝의 시간(h, 0.50 s/iter)": round(15000 * 0.50 / 3600, 2),
    "TEMPO 이력 창 H = K*L (프레임, 초 @30Hz)": [10 * 30, 10 * 30 / 30],
}

# V19: 공개 코드 파라미터 수 산술 (정적 읽기, 코드를 실행하지 않고 형태에서 계산)
w_vlm, w_exp, sam_dim, ntok, hist_k, hist_d = 2048, 1024, 256, 64, 10, 14  # gemma_2b, gemma_300m 폭(gemma.py)
fusion = (2 * w_vlm) + (sam_dim * w_vlm + w_vlm) + (ntok * w_vlm) + (2 * w_vlm) + 4 * (w_vlm * w_vlm + w_vlm) + 1
cond = (hist_k * hist_d * w_exp + w_exp) + 2 * (w_exp * w_exp + w_exp)
toks = (hist_d * w_vlm + w_vlm) + hist_k * w_vlm
V["V19_공개코드_추가_파라미터"] = {
    "SAM2 융합 블록": fusion, "ACT adaRMS 조건 MLP": cond, "ACT 접두 토큰 투영": toks,
    "ACT 합": cond + toks, "전체 합": fusion + cond + toks,
    "논문 2M 대비 배": round((fusion + cond + toks) / 2e6, 1),
    "ACT 합이 2M과 같은 규모인가": abs((cond + toks) - 2e6) / 2e6 < 0.2,
    "2.5B 기준 비율 %": round(100 * (fusion + cond + toks) / 2.5e9, 2),
}

# V20: TempoBench 표
TB = {  # 과제: (에피소드, 주석 에피소드, 프레임, 검출 프레임, 주석 프레임 or None)
    "catch_ball": (59, 54, 10881, 3823), "ball_drop": (31, 29, 29882, 24875, 28260),
    "dynamic_pour": (23, 23, 18432, 15258, 18432), "spartan": (54, 52, 19534, 8624),
    "dynamic_handover": (90, 24, 18536, 2902),
}
tot_ep = sum(v[0] for v in TB.values())
tot_ann = sum(v[1] for v in TB.values())
tot_fr = sum(v[2] for v in TB.values())
tot_det = sum(v[3] for v in TB.values())
V["V20_TempoBench"] = {
    "에피소드 합": [tot_ep, 257], "주석 에피소드 합": [tot_ann, 182], "프레임 합": [tot_fr, 97265],
    "검출 프레임 합": [tot_det, 55482], "총 길이 분(30fps)": round(tot_fr / 30 / 60, 1),
    "검출 프레임 / 전체 %": round(100 * tot_det / tot_fr, 1),
    "과제별 프레임/에피소드": {k: round(v[2] / v[0]) for k, v in TB.items()},
    "과제별 검출 프레임 / 프레임 %": {k: round(100 * v[3] / v[2], 1) for k, v in TB.items()},
    "해석": "논문의 '50k annotated frames'는 검출 프레임 55,482와 같은 규모. 전체 프레임 97,265의 57%.",
}

RES["가정"] = "과제당 50회(본문 5.2절), Wine Pour는 질량 %로 검정 안 함"
RES["결과"] = V
OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")

# 표준출력 요약
print("V1 평균:", [(r["방법"], r["재계산_평균"], r["논문_평균"]) for r in V["V1_표4_평균"]])
print("V2 정수 불가능:", [f for f in feas if not f["정수"]])
print("V2b 막대:", [(r["과제"], r["방법"], r["합"], r["표와 일치"]) for r in bar_rows])
for c in core:
    print("V4", c)
print("V5", cmh)
print("V6", V["V6_본문_차이의_기준선"], V["V6b_UVT가_기준선_위인_과제"])
for c in V["V7_실패유형"] + [V["V7b_Drop_TEMPO_대_RTC"]]:
    print("V7", c)
print("V8", V["V8_표1_미절단"])
for c in t5 + [V["V9b_표5_RTC+TEMPO_대_VLASH+TEMPO"]] + abl + V["V11_표2_인코더"]:
    print("V9-11", c)
print("V12", V["V12_표7_평균"], V["V13_그림A5_평균"])
print("V14", V["V14_프로브_대_과제성공"])
print("V16", mde)
print("V18", V["V18_비용_일관성"])
print("V19", V["V19_공개코드_추가_파라미터"])
print("V20", V["V20_TempoBench"])

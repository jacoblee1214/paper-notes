"""ActionCodec (arXiv:2602.15397 v2) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문 표와 본문의 값을 아래에 그대로 옮겨 적었다(출처는 주석). 외부 데이터는 쓰지 않는다.
      그림 3과 그림 7의 값은 PDF 그림에 인쇄된 숫자를 읽어 옮긴 것이다(그림 판독 값).
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- LIBERO는 스위트당 10과제 x 50회 = 500회, 전체 2,000회다(부록 B.1, B.3). 시행은 독립이라고 가정한 근사다.
- 그림 3(LIBERO-Goal)은 한 번의 학습 결과이고 시행 수는 500회로 가정한다. 시드 수는 논문에 없다.
- 그림 7(SO100)은 과제 10개 x 20회 = 200회(부록 B.1)로 읽는다.
- 표 3(xArm) 과제(채소) 수는 논문에 없다. 정수 격자 검사로 가능한 값을 찾는다.
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
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def comp(name, k1, n1, k2, n2):
    d, lo, hi = newcombe_diff(k1, n1, k2, n2)
    return {
        "comparison": name, "k1": k1, "n1": n1, "k2": k2, "n2": n2,
        "diff_pp": round(100 * d, 1),
        "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
        "fisher_p": float(f"{fisher(k1, n1, k2, n2):.3g}"),
    }


def pct_to_k(p, n):
    return round(p / 100 * n)


def feasible(p, n, tol=0.0501):
    """퍼센트 p(소수 1자리)가 n회 시행의 k/n으로 표현되는가."""
    return any(abs(100 * k / n - p) <= tol for k in range(n + 1))


# ---------------------------------------------------------------- 표 1 (논문 값 그대로)
# 열: Goal, Spatial, Object, Long, Avg, AvgRank (None은 논문의 '-')
T1A = {
    "Octo-Base": [84.6, 78.9, 85.7, 51.1, 75.1, 10.5], "SpatialVLA": [78.6, 88.2, 89.9, 55.5, 78.1, 9.5],
    "GR00T-N1": [93.0, 94.4, 97.6, 90.6, 93.9, 6.6], "UniVLA": [95.6, 96.5, 96.8, 92.0, 95.2, 5.4],
    "VQ-VLA": [75.2, None, None, 60.0, None, None], "OpenVLA": [79.2, 84.7, 88.4, 53.7, 76.5, 10.0],
    "OpenVLA-OFT": [97.9, 97.6, 98.4, 94.5, 97.1, 2.5], "pi0": [95.8, 96.8, 98.8, 85.2, 94.2, 4.5],
    "pi0.5": [98.0, 98.8, 98.2, 92.4, 96.8, 2.5], "pi0 FAST": [88.6, 96.4, 96.8, 60.2, 85.5, 7.4],
    "ActionCodec (2.2B)": [95.4, 96.2, 99.6, 90.6, 95.5, 5.1], "ActionCodec-BAR": [97.6, 97.2, 99.8, 94.8, 97.4, 2.0],
}
T1B = {
    "Diffusion Policy": [68.3, 78.3, 92.5, 50.5, 72.4, 13.8], "MiniVLA": [None, None, None, 77.0, None, None],
    "OpenVLA-OFT (scratch)": [91.7, 94.3, 95.2, 86.5, 91.9, 7.8], "SmolVLA 2.2B": [91.0, 93.0, 94.0, 77.0, 88.8, 10.2],
    "pi0 FAST (scratch)": [89.0, 87.0, 63.0, 48.0, 71.8, 14.2], "pi0.5 (scratch)": [94.6, 96.6, 97.2, 85.8, 93.3, 6.4],
    "VLA-0": [96.2, 97.0, 97.8, 87.6, 94.7, 3.8],
}
T1C = {
    "Binning": [27.4, 64.2, 71.0, 50.8, 53.4, 15.5], "String": [30.8, 69.6, 71.6, 26.4, 49.6, 15.5],
    "VQVLA's": [60.5, 56.0, 69.8, 51.4, 60.5, 15.2], "MiniVLA's": [82.1, 82.8, 90.2, 75.2, 82.6, 12.8],
    "FAST": [89.2, 90.2, 97.2, 85.8, 90.6, 9.4], "ActionCodec (256M)": [91.2, 94.2, 97.0, 86.8, 92.3, 7.5],
    "ActionCodec (500M)": [96.4, 91.6, 96.0, 85.8, 92.5, 7.8], "ActionCodec (2.2B)": [95.4, 96.2, 99.6, 90.6, 95.5, 3.8],
    "ActionCodec-PD": [97.0, 95.2, 98.4, 91.6, 95.6, 3.0], "ActionCodec-KI": [95.0, 93.6, 97.6, 90.8, 94.3, 5.5],
    "ActionCodec-BAR": [97.6, 97.2, 99.8, 94.8, 97.4, 1.0],
}
# 표 1(d): 열 Spoon, Carrot, Block, Eggplant, Avg, AvgRank. 첫 블록(Pt=Y 모델 + BAR)과 둘째 블록(MiniVLA + BAR)
T1D1 = {
    "Octo-Base": [12.5, 8.3, 0.0, 43.1, 16.0, 7.3], "SpatialVLA": [16.7, 25.0, 29.2, 100.0, 42.7, 4.0],
    "pi0": [66.7, 58.3, 58.3, 88.3, 67.9, 2.0], "OpenVLA-OFT": [12.5, 4.2, 8.3, 0.0, 6.3, 8.1],
    "UniVLA": [54.2, 66.7, 50.0, 4.2, 43.8, 3.5], "OpenVLA": [32.0, 30.0, 18.0, 38.0, 29.5, 4.8],
    "VQ-VLA": [12.5, 8.0, 6.0, 0.0, 6.6, 8.1], "pi0 FAST": [29.1, 21.9, 10.8, 66.6, 32.1, 5.3],
    "ActionCodec-BAR": [71.7, 64.2, 55.0, 70.0, 65.2, 2.0],
}
T1D2 = {"MiniVLA": [68.0, 44.0, 70.0, 14.0, 49.0, 1.8], "ActionCodec-BAR": [71.7, 64.2, 55.0, 70.0, 65.2, 1.3]}


def avg_rank(table, cols):
    names = list(table)
    ranks = {n: [] for n in names}
    for c in cols:
        idx = [i for i, n in enumerate(names) if table[n][c] is not None]
        vals = np.array([table[names[i]][c] for i in idx])
        r = stats.rankdata(-vals)
        for i, rr in zip(idx, r):
            ranks[names[i]].append(float(rr))
    return {n: (float(np.mean(v)) if len(v) == len(cols) else None) for n, v in ranks.items()}


def rhalf(x):
    """소수 첫째 자리 반올림(0.05 경계는 짝수 쪽, 논문이 어느 쪽인지 모르므로 두 값 모두 허용)."""
    return round(x + 1e-9, 1), round(x - 1e-9, 1)


# ---------------------------------------------------------------- V1 평균 재계산
v1 = []
for pname, tab, ncol in (("1(a)", T1A, 4), ("1(b)", T1B, 4), ("1(c)", T1C, 4), ("1(d)-1", T1D1, 4), ("1(d)-2", T1D2, 4)):
    for name, row in tab.items():
        vals = row[:ncol]
        if any(v is None for v in vals) or row[4] is None:
            continue
        m = float(np.mean(vals))
        ok = row[4] in rhalf(m)
        v1.append({"panel": pname, "row": name, "recomputed": round(m, 3), "reported": row[4], "match": bool(ok)})
RES["V1_row_averages"] = {"n_rows": len(v1), "n_match": sum(r["match"] for r in v1),
                          "mismatch": [r for r in v1 if not r["match"]]}

# ---------------------------------------------------------------- V2 평균 순위 재현
rk = {}
a_no_vq = {k: v for k, v in T1A.items() if v[1] is not None}
rk["1(a) 패널 안(결측 행 VQ-VLA 제외)"] = avg_rank(a_no_vq, [0, 1, 2, 3])
bc = {}
for k, v in T1B.items():
    if v[1] is not None:
        bc["b:" + k] = v
for k, v in T1C.items():
    bc["c:" + k] = v
rk["1(b)+1(c) 합쳐서(결측 행 제외)"] = avg_rank(bc, [0, 1, 2, 3])
rk["1(c) 패널 안만"] = avg_rank({"c:" + k: v for k, v in T1C.items()}, [0, 1, 2, 3])
d1 = avg_rank(T1D1, [0, 1, 2, 3])
d2 = avg_rank(T1D2, [0, 1, 2, 3])
rk["1(d) 첫 블록"] = d1
rk["1(d) 둘째 블록"] = d2


def compare_rank(ranks, rep_table, prefix=""):
    out = []
    for n, r in ranks.items():
        key = n.split(":", 1)[1] if ":" in n else n
        rep = rep_table.get(key, [None] * 6)[5]
        if r is None or rep is None:
            continue
        ok = rep in rhalf(r) or abs(r - rep) < 0.06
        out.append({"row": n, "recomputed": round(r, 3), "reported": rep, "match": bool(ok)})
    return out


rep_a = {k: v for k, v in T1A.items()}
rep_b = {**T1B}
rep_c = {**T1C}
cmp_a = compare_rank(rk["1(a) 패널 안(결측 행 VQ-VLA 제외)"], rep_a)
cmp_bc = []
for n, r in rk["1(b)+1(c) 합쳐서(결측 행 제외)"].items():
    pre, key = n.split(":", 1)
    rep = (rep_b if pre == "b" else rep_c)[key][5]
    if r is None or rep is None:
        continue
    cmp_bc.append({"row": n, "recomputed": round(r, 3), "reported": rep, "match": bool(abs(r - rep) < 0.06)})
cmp_c_only = []
for n, r in rk["1(c) 패널 안만"].items():
    key = n.split(":", 1)[1]
    cmp_c_only.append({"row": n, "recomputed": round(r, 3), "reported": rep_c[key][5], "match": bool(abs(r - rep_c[key][5]) < 0.06)})
cmp_d1 = compare_rank(d1, T1D1)
cmp_d2 = compare_rank(d2, T1D2)
RES["V2_avg_rank"] = {
    "note": "Avg. Rank는 4개 스위트 열의 순위를 평균. 결측('-')이 있는 행은 순위 계산에서 빠짐. (b)와 (c)는 한 풀로 합쳐 순위를 매겨야 보고 값이 나옴",
    "1(a)": {"n_match": sum(x["match"] for x in cmp_a), "n": len(cmp_a)},
    "1(b)+1(c) 합친 풀": {"n_match": sum(x["match"] for x in cmp_bc), "n": len(cmp_bc)},
    "1(c) 패널 안만": {"n_match": sum(x["match"] for x in cmp_c_only), "n": len(cmp_c_only),
                      "example": [x for x in cmp_c_only if x["row"] in ("c:Binning", "c:ActionCodec-BAR", "c:ActionCodec (2.2B)")]},
    "1(d) 첫 블록": {"n_match": sum(x["match"] for x in cmp_d1), "n": len(cmp_d1), "rows": cmp_d1},
    "1(d) 둘째 블록": {"n_match": sum(x["match"] for x in cmp_d2), "n": len(cmp_d2), "rows": cmp_d2},
    "1(a)에서 ActionCodec (2.2B) 순위 5.1 대 1(c)에서 3.8": "같은 행 값이 풀에 따라 다름",
}

# ---------------------------------------------------------------- V3 격자 검사 (500회/스위트)
lat = []
for pname, tab in (("1(c) 저자 실행", T1C),):
    for name, row in tab.items():
        for j, s in enumerate(["Goal", "Spatial", "Object", "Long"]):
            if row[j] is not None and not feasible(row[j], 500):
                lat.append({"panel": pname, "row": name, "suite": s, "value": row[j]})
lat_other = []
for pname, tab in (("1(a)", T1A), ("1(b)", T1B)):
    for name, row in tab.items():
        for j, s in enumerate(["Goal", "Spatial", "Object", "Long"]):
            if row[j] is not None and not feasible(row[j], 500):
                lat_other.append({"panel": pname, "row": name, "suite": s, "value": row[j]})
RES["V3_lattice_500"] = {"author_rows_not_on_lattice": lat, "imported_rows_not_on_lattice": lat_other,
                         "note": "스위트당 500회이면 값은 0.2의 배수. 1(a), 1(b)는 다른 논문에서 옮겼을 가능성이 있어 시행 수를 알 수 없음"}

fig3 = {  # 그림 3 (LIBERO-Goal, 256M). 0.5k, 1k, 20k 스텝 값 (그림 판독 값). acc는 L1 오차
    "n8_S2048_OR40": (18.2, 36.8, 73.4, 0.047), "n16_S2048_OR40": (12.4, 37.2, 67.2, 0.029),
    "n32_S2048_OR40": (12.8, 37.2, 60.0, 0.020), "n64_S2048_OR40": (5.2, 18.4, 29.2, 0.016),
    "n16_S512_OR40": (12.6, 37.8, 77.0, 0.031), "n16_S1024_OR40": (12.6, 38.0, 75.6, 0.024),
    "n16_S4096_OR40": (10.8, 35.2, 65.8, 0.029),
    "n16_S2048_OR26": (8.2, 22.4, 37.4, 0.020), "n16_S2048_OR70": (33.4, 67.6, 86.2, 0.028),
    "n16_S2048_OR70_SA": (22.8, 44.8, 77.2, 0.028), "n16_S2048_OR70_Causal": (26.0, 48.2, 83.8, 0.028),
    "n16_S2048_OR70_SP-L": (34.2, 67.8, 87.8, 0.027), "n16_S2048_OR70_SP-L-B-D": (34.0, 68.8, 87.4, 0.031),
    "FAST_S2048_OR26": (2.0, 16.4, 52.0, 0.016),
}
bad = [(k, v) for k, vals in fig3.items() for v in vals[:3] if not feasible(v, 500)]
RES["V3b_fig3_lattice"] = {"values_off_lattice": bad, "n_values": 3 * len(fig3)}

# ---------------------------------------------------------------- V4 표 2 내부 일관성
T2 = [  # 이름, 토큰 예산, OR, horizon, latency, throughput, SR
    ("Binning T=20", 140.0, 39, 20, 4.9, 4.1, 3.5), ("Binning T=8", 56.0, 38, 8, 2.6, 3.1, 53.4),
    ("String T=20", 779.2, 28, 20, 33.8, 0.6, 20.4), ("String T=8", 311.8, 34, 8, 12.7, 0.6, 49.6),
    ("VQVLA's", 4.0, 45, 5, 0.7, 7.7, 60.5), ("MiniVLA's", 7.0, 40, 8, 0.7, 11.1, 82.6),
    ("FAST", 24.3, 19, 20, 1.5, 13.1, 90.6), ("ActionCodec", 16.0, 72, 20, 0.9, 22.0, 95.5),
]
t2 = []
for name, tok, orr, hz, lat_s, thr, sr in T2:
    lo, hi = hz / (lat_s + 0.05), hz / (lat_s - 0.05)
    t2.append({"row": name, "throughput_reported": thr, "range_from_latency_rounding": [round(lo, 2), round(hi, 2)],
               "consistent": bool(lo - 0.05 <= thr <= hi + 0.05)})
RES["V4_table2_throughput"] = t2
dims7 = [("Binning T=20", 7 * 20), ("Binning T=8", 7 * 8)]
RES["V4b_table2_binning_tokens"] = [{"row": n, "expected": e, "reported": next(x[1] for x in T2 if x[0] == n)} for n, e in dims7]
t1c_sr = {"Binning T=8": 53.4, "String T=8": 49.6, "VQVLA's": 60.5, "MiniVLA's": 82.6, "FAST": 90.6, "ActionCodec": 95.5}
t1c_avg = {"Binning T=8": T1C["Binning"][4], "String T=8": T1C["String"][4], "VQVLA's": T1C["VQVLA's"][4],
           "MiniVLA's": T1C["MiniVLA's"][4], "FAST": T1C["FAST"][4], "ActionCodec": T1C["ActionCodec (2.2B)"][4]}
RES["V4c_table2_SR_vs_table1c"] = {k: {"table2": t1c_sr[k], "table1c": t1c_avg[k], "match": t1c_sr[k] == t1c_avg[k]} for k in t1c_sr}
RES["V4d_binning_T20_vs_T8"] = {"note": "Binning과 String의 T=20 행(3.5, 20.4)은 표 1에 없음. 표 1의 Binning 53.4와 String 49.6은 T=8 행",
                                 "table1c_horizons": "Binning, String = 8; VQVLA's = 5; MiniVLA's = 8; FAST, ActionCodec = 20 (표 2)"}

# ---------------------------------------------------------------- V5 표 1(c)/(a) 유의성 (n=2000, 독립 가정)
N = 2000


def k_of(p):
    return pct_to_k(p, N)


c = {n: v[4] for n, v in T1C.items()}
V5 = []
V5.append(comp("ActionCodec(2.2B) 95.5 대 FAST 90.6", k_of(95.5), N, k_of(90.6), N))
V5.append(comp("ActionCodec(256M) 92.3 대 FAST(2.2B) 90.6", k_of(92.3), N, k_of(90.6), N))
V5.append(comp("ActionCodec(500M) 92.5 대 FAST(2.2B) 90.6", k_of(92.5), N, k_of(90.6), N))
V5.append(comp("ActionCodec(2.2B) 95.5 대 MiniVLA's 82.6", k_of(95.5), N, k_of(82.6), N))
V5.append(comp("PD 95.6 대 AR(ActionCodec 2.2B) 95.5", k_of(95.6), N, k_of(95.5), N))
V5.append(comp("BAR 97.4 대 AR 95.5", k_of(97.4), N, k_of(95.5), N))
V5.append(comp("KI 94.3 대 AR 95.5", k_of(94.3), N, k_of(95.5), N))
V5.append(comp("BAR 97.4 대 OpenVLA-OFT(Pt=Y) 97.1", k_of(97.4), N, k_of(97.1), N))
V5.append(comp("BAR 97.4 대 pi0.5(Pt=Y) 96.8", k_of(97.4), N, k_of(96.8), N))
V5.append(comp("BAR 97.4 대 VLA-0(scratch) 94.7", k_of(97.4), N, k_of(94.7), N))
V5.append(comp("BAR 97.4 대 pi0.5(scratch) 93.3", k_of(97.4), N, k_of(93.3), N))
RES["V5_libero_comparisons"] = V5
suites = ["Goal", "Spatial", "Object", "Long"]
RES["V5b_PD_KI_BAR_vs_AR_by_suite"] = {
    v: {s: round(T1C[v][i] - T1C["ActionCodec (2.2B)"][i], 1) for i, s in enumerate(suites)}
    for v in ("ActionCodec-PD", "ActionCodec-KI", "ActionCodec-BAR")}

# ---------------------------------------------------------------- V6 표 4 절제 (평균 2,000회)
T4 = {0: 92.0, 1: 92.7, 2: 95.2, 3: 95.5, 4: 95.1, 5: 94.6}
V6 = [comp(f"ID {a} ({T4[a]}) 대 ID {b} ({T4[b]})", k_of(T4[a]), N, k_of(T4[b]), N)
      for a, b in ((3, 2), (3, 1), (1, 0), (3, 4), (3, 5), (4, 5))]
RES["V6_table4"] = V6
RES["V6b_table4_pp_label"] = "표 4의 괄호 '-3.5%' 등은 95.5 기준 퍼센트 포인트 차이(92.0 = 95.5 - 3.5)이며 상대 비율(3.7%)이 아님"

# ---------------------------------------------------------------- V7 그림 3 대비 (LIBERO-Goal, 500회, 한 번의 학습)
M = 500


def f3(key, j=2):
    return pct_to_k(fig3[key][j], M)


V7 = [
    comp("OR 70% 대 OR 40% (20k, n=16 S=2048)", f3("n16_S2048_OR70"), M, f3("n16_S2048_OR40"), M),
    comp("OR 70% 대 OR 26% (20k)", f3("n16_S2048_OR70"), M, f3("n16_S2048_OR26"), M),
    comp("OR 70% 대 OR 26% (500스텝) 33.4 대 8.2", f3("n16_S2048_OR70", 0), M, f3("n16_S2048_OR26", 0), M),
    comp("OR 40% 대 OR 26% (500스텝) 12.4 대 8.2", f3("n16_S2048_OR40", 0), M, f3("n16_S2048_OR26", 0), M),
    comp("n=8 대 n=16 (20k, OR 40%) 73.4 대 67.2", f3("n8_S2048_OR40"), M, f3("n16_S2048_OR40"), M),
    comp("S=512 대 S=2048 (20k, n=16) 77.0 대 67.2", f3("n16_S512_OR40"), M, f3("n16_S2048_OR40"), M),
    comp("독립(70%) 86.2 대 인과 Causal 83.8", f3("n16_S2048_OR70"), M, f3("n16_S2048_OR70_Causal"), M),
    comp("독립(70%) 86.2 대 SA 77.2", f3("n16_S2048_OR70"), M, f3("n16_S2048_OR70_SA"), M),
    comp("독립(70%) 86.2 대 SP-L 87.8", f3("n16_S2048_OR70"), M, f3("n16_S2048_OR70_SP-L"), M),
    comp("ActionCodec식 70%OR 86.2 대 FAST 52.0 (20k, 256M)", f3("n16_S2048_OR70"), M, f3("FAST_S2048_OR26"), M),
]
RES["V7_fig3_contrasts"] = V7
RES["V7b_fig3_n_S_monotone"] = {
    "n sweep (S=2048, OR 40%) 20k": {"n8": 73.4, "n16": 67.2, "n32": 60.0, "n64": 29.2},
    "S sweep (n=16, OR 40%) 20k": {"S512": 77.0, "S1024": 75.6, "S2048": 67.2, "S4096": 65.8},
    "L1 by S": {"S512": 0.031, "S1024": 0.024, "S2048": 0.029, "S4096": 0.029},
    "note": "선택된 (n=16, S=2048)은 이 스윕에서 20k 성공률이 가장 높은 설정이 아님. 선택 근거는 재구성 오차와 FAST 어휘 크기(부록 E.2)"}

# 표 5와 그림 3 일치
T5 = {26: (100, 2016, 0.020, 8.2), 40: (100, 1583, 0.029, 12.4), 70: (100, 1290, 0.028, 33.4)}
RES["V8_table5_vs_fig3"] = {
    "SR at 500 steps": {"table5": {k: v[3] for k, v in T5.items()},
                         "fig3": {26: fig3["n16_S2048_OR26"][0], 40: fig3["n16_S2048_OR40"][0], 70: fig3["n16_S2048_OR70"][0]}},
    "L1": {"table5": {k: v[2] for k, v in T5.items()},
           "fig3": {26: fig3["n16_S2048_OR26"][3], 40: fig3["n16_S2048_OR40"][3], 70: fig3["n16_S2048_OR70"][3]}},
    "PPL_upper_bound_S2048": 2048, "PPL_all_below_bound": all(v[1] <= 2048 for v in T5.values()),
    "entropy_nats_from_PPL": {k: round(math.log(v[1]), 3) for k, v in T5.items()},
    "text_8.2_is_naive_VQ_VAE": "본문 3.1은 naive VQ-VAE 8.2%, 표 5에서 OR 26% 행의 값도 8.2%",
}

# ---------------------------------------------------------------- V9 Fig.1과 본문 수치
RES["V9_fig1_text"] = {
    "ActionCodec 5k 89.5 대 FAST 38.6": comp("5k 스텝, LIBERO-ALL(n=2000 가정)", k_of(89.5), N, k_of(38.6), N),
    "Fig1 20k 값": {"ActionCodec": 95.5, "FAST": 90.6, "표 1(c)과 같은 값": True},
    "note": "그림 1의 20k 스텝 점(95.5, 90.6)이 표 1의 최종 값과 같음. 부록 B.3은 30k 스텝까지 수렴을 관찰했다고 적음(그림 판독 값)"}

# ---------------------------------------------------------------- V10 SimplerEnv (d)
cand_n = [24, 48, 50, 72, 96, 100, 120, 144, 240]
d_rows = []
for name, row in {**{"A:" + k: v for k, v in T1D1.items()}, **{"B:" + k: v for k, v in T1D2.items()}}.items():
    ns = {}
    for j in range(4):
        ns[j] = [n for n in cand_n if feasible(row[j], n)]
    common = sorted(set.intersection(*[set(v) for v in ns.values()]))
    d_rows.append({"row": name, "feasible_n_per_task": ns, "common_n_from_candidates": common})
RES["V10_simpler_lattice"] = {"candidate_n": cand_n, "rows": d_rows,
                              "note": "BAR는 120회(24x5)와 일치. 행마다 가능한 시행 수가 다르므로 한 표 안에 여러 평가 횟수가 섞였을 가능성"}
bar_k = [pct_to_k(p, 120) for p in T1D1["ActionCodec-BAR"][:4]]
pi0_k = [pct_to_k(p, 120) for p in T1D1["pi0"][:4]]
RES["V10b_bar_vs_pi0"] = {
    "BAR counts (n=120/task)": bar_k, "sum": sum(bar_k), "avg_pct": round(100 * sum(bar_k) / 480, 2),
    "pi0 counts (n=120/task 가정)": pi0_k, "sum_pi0": sum(pi0_k), "avg_pct_pi0": round(100 * sum(pi0_k) / 480, 2),
    "test": comp("BAR 대 pi0 (4과제 합, n=480 가정)", sum(bar_k), 480, sum(pi0_k), 480),
}
mini_k = [pct_to_k(p, 50) for p in T1D2["MiniVLA"][:4]]
RES["V10c_bar_vs_minivla"] = {
    "MiniVLA counts (n=50/task 가정)": mini_k,
    "test_total": comp("BAR(480) 대 MiniVLA(200, 과제당 50회 가정)", sum(bar_k), 480, sum(mini_k), 200),
    "per_task_BAR_minus_MiniVLA_pp": [round(a - b, 1) for a, b in zip(T1D1["ActionCodec-BAR"][:4], T1D2["MiniVLA"][:4])],
    "per_task_fisher": [float(f"{fisher(b, 120, m, 50):.3g}") for b, m in zip(bar_k, mini_k)]}

# ---------------------------------------------------------------- V11 표 3 xArm 격자
T3 = {"w/ PT": 82.5, "w/o PT": 74.1, "pi0": 75.0, "pi0 FAST": 72.5}
Nn = []
for n_items in range(1, 41):
    tot = 20 * n_items
    if all(feasible(v, tot) for v in T3.values()):
        Nn.append(n_items)
RES["V11_xarm_lattice"] = {
    "trials_per_item": 20, "values": T3,
    "n_items_consistent_with_all_four": Nn,
    "n_items_for_each": {k: [n for n in range(1, 41) if feasible(v, 20 * n)] for k, v in T3.items()},
    "note": "논문은 채소 수를 적지 않음. 4개 값을 모두 만드는 최소 채소 수는 아래 목록의 첫 값"}
if Nn:
    tot = 20 * Nn[0]
    RES["V11b_xarm_tests_if_min_items"] = {
        "assumed_items": Nn[0], "n_total": tot,
        "w/PT vs w/oPT": comp("82.5 대 74.1", pct_to_k(82.5, tot), tot, pct_to_k(74.1, tot), tot),
        "w/PT vs pi0": comp("82.5 대 75.0", pct_to_k(82.5, tot), tot, pct_to_k(75.0, tot), tot)}

# ---------------------------------------------------------------- V12 SO100 그림 7 (그림 판독 값)
so = {  # 과제 1..10: (pick, place) 순서 ActionCodec, w/o CT, FAST, pi0
    "ActionCodec": [(18, 5), (14, 6), (13, 6), (10, 5), (14, 7), (15, 8), (19, 12), (16, 7), (13, 5), (16, 10)],
    "w/o CT": [(18, 2), (13, 4), (15, 3), (12, 6), (12, 7), (13, 7), (16, 13), (16, 7), (13, 5), (14, 7)],
    "FAST": [(12, 2), (7, 1), (9, 2), (10, 5), (11, 5), (14, 8), (17, 10), (14, 6), (14, 6), (10, 4)],
    "pi0": [(10, 1), (8, 1), (14, 3), (12, 5), (9, 6), (16, 8), (16, 9), (15, 5), (9, 3), (12, 5)],
}
rep_pick = {"ActionCodec": 74, "w/o CT": 71, "FAST": 59, "pi0": 61}
rep_place = {"ActionCodec": 36, "w/o CT": 31, "FAST": 25, "pi0": 23}
so_sum = {}
for m, v in so.items():
    pk = sum(a for a, _ in v)
    pl = sum(b for _, b in v)
    so_sum[m] = {"pick": pk, "place": pl, "pick_pct": round(100 * pk / 200, 1), "place_pct": round(100 * pl / 200, 1),
                 "reported_pick": rep_pick[m], "reported_place": rep_place[m],
                 "pick_matches": round(100 * pk / 200 + 1e-9) == rep_pick[m], "place_matches": round(100 * pl / 200 + 1e-9) == rep_place[m],
                 "place_given_pick_pct": round(100 * pl / pk, 1), "place_le_pick_all_tasks": all(b <= a for a, b in v)}
RES["V12_so100_sums"] = so_sum
RES["V12b_so100_tests"] = [
    comp("Place: ActionCodec 71 대 w/o CT 61 (n=200)", so_sum["ActionCodec"]["place"], 200, so_sum["w/o CT"]["place"], 200),
    comp("Place: ActionCodec 71 대 FAST 49", so_sum["ActionCodec"]["place"], 200, so_sum["FAST"]["place"], 200),
    comp("Place: ActionCodec 71 대 pi0 46", so_sum["ActionCodec"]["place"], 200, so_sum["pi0"]["place"], 200),
    comp("Pick: ActionCodec 148 대 w/o CT 142", so_sum["ActionCodec"]["pick"], 200, so_sum["w/o CT"]["pick"], 200),
    comp("Pick: ActionCodec 148 대 FAST 118", so_sum["ActionCodec"]["pick"], 200, so_sum["FAST"]["pick"], 200),
    comp("Pick: ActionCodec 148 대 pi0 121", so_sum["ActionCodec"]["pick"], 200, so_sum["pi0"]["pick"], 200),
]
paired = []
for i in range(10):
    paired.append(so["ActionCodec"][i][1] - so["w/o CT"][i][1])
RES["V12c_so100_place_paired"] = {
    "task_diffs_place_AC_minus_woCT": paired,
    "wins_losses_ties": [sum(d > 0 for d in paired), sum(d < 0 for d in paired), sum(d == 0 for d in paired)],
    "sign_test_p": float(stats.binomtest(sum(d > 0 for d in paired), sum(d != 0 for d in paired), 0.5).pvalue),
    "wilcoxon_p": float(stats.wilcoxon(paired).pvalue) if any(d != 0 for d in paired) else None,
    "note": "과제는 10개뿐이고 과제 효과와 시행 간 상관이 있어 Fisher 값은 낙관적"}

# ---------------------------------------------------------------- V13 토크나이저 매개변수 수 (공개 가중치 메타데이터와 구성 비교)
dim, L, ff = 384, 12, 4 * 384
attn = 4 * dim * dim
mlp = dim * ff + ff + ff * dim + dim
norms = 2 * 2 * dim
scale = 2 * dim
block = attn + mlp + norms + scale
enc = L * block + 7 * dim + dim + dim * 512 + 512 + 3 * dim * 16
dec = L * block + 512 * dim + dim + dim * 7 + 7 + 3 * dim * 1
codebook = 2048 * 512 * 2 + 2048  # EMA 코드북: embed, embed_avg, cluster_size
est = enc + dec + codebook
file_bytes = 180387949
RES["V13_param_estimate"] = {
    "estimate_params": est, "file_bytes_fp32": file_bytes, "file_params_if_fp32": file_bytes // 4,
    "ratio": round(est / (file_bytes / 4), 3),
    "note": "공개 ActionCodec-Base config(dim 384, 12층, z_dim 512, 코드북 2048)로 센 추정치와 safetensors 크기(fp32 가정)의 비. 논문은 토크나이저 매개변수 수를 적지 않음"}
expert = 24 * (4 * 768 * 2048 + 3 * 768 * 2048) + 0
RES["V13b_action_expert_params"] = {
    "estimate_params_no_embeddings": expert, "file_bytes": 1063429688, "file_params_if_fp32": 1063429688 // 4,
    "paper_text": "300M-parameter action expert (부록 B.4)",
    "note": "config(hidden 768, 24층, 중간 2048, 헤드 32 x 64)로 센 값. 추정이 파일 크기와 대략 같음(임베딩과 헤드 제외)"}

# ---------------------------------------------------------------- V14 표 10
T10 = {"pi0-FAST": (32.1, 85.5, None), "pi0.5": (57.1, 96.8, 57.5), "MolmoAct2-DROID": (None, None, 52.0),
       "ActionCodec (Qwen3.5-2B)": (87.7, 98.2, 82.5)}
RES["V14_table10"] = {
    "pi0-FAST Simpler 32.1 = 표 1(d) pi0 FAST 평균": T10["pi0-FAST"][0] == T1D1["pi0 FAST"][4],
    "pi0-FAST LIBERO 85.5 = 표 1(a)": T10["pi0-FAST"][1] == T1A["pi0 FAST"][4],
    "pi0.5 LIBERO 96.8 = 표 1(a)": T10["pi0.5"][1] == T1A["pi0.5"][4],
    "pi0.5 Simpler 57.1은 표 1(d)에 없음": True,
    "LIBERO 98.2 대 pi0.5 96.8 (n=2000 가정)": comp("98.2 대 96.8", k_of(98.2), N, k_of(96.8), N),
    "LIBERO 98.2 대 BAR 97.4": comp("98.2 대 97.4", k_of(98.2), N, k_of(97.4), N),
    "note": "DROID 제로샷 평가의 장소, 과제, 시행 수는 논문에 없음. 표 10의 비교 대상 값의 출처도 적혀 있지 않음"}

# ---------------------------------------------------------------- V15 버전 비교 (수치)
RES["V15_v1_v2_numbers"] = {
    "표 1(원래 표 1과 2) 수치": "v1과 v2에서 같음 (v1의 모든 소수 값이 v2에도 있음)",
    "v2에서 새로 생긴 값": ["표 5 (OR 26/40/70%의 PPL, L1, 500스텝 성공률 8.2/12.4/33.4)", "표 10 (대규모 사전학습 결과 57.1, 57.5, 87.7)",
                         "표 11 (인코딩 시간 0.11, 0.09 ms)", "그림 3 L1 값(0.020, 0.028, 0.029)"],
    "문구 변경": ["'robotics pre-training' -> 'VLA pre-training' (초록, 표 1 캡션에서 Pt.를 정의)", "'enhanced multimodal mutual information' -> 'improved multimodal alignment'",
                 "'token independence' -> 'reduced architectural token coupling'"],
}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------------------------------------------------------------- 요약 출력
print("V1 평균 재계산:", RES["V1_row_averages"]["n_match"], "/", RES["V1_row_averages"]["n_rows"], "일치",
      RES["V1_row_averages"]["mismatch"])
for k in ("1(a)", "1(b)+1(c) 합친 풀", "1(c) 패널 안만", "1(d) 첫 블록", "1(d) 둘째 블록"):
    print("V2 순위", k, RES["V2_avg_rank"][k]["n_match"], "/", RES["V2_avg_rank"][k]["n"])
print("V3 격자 밖(저자 행):", lat)
print("V3 격자 밖(가져온 행):", [(x["row"], x["suite"], x["value"]) for x in lat_other])
print("V3b 그림 3 격자 밖:", bad)
print("V4 처리량:", [(x["row"], x["consistent"]) for x in t2])
for x in V5:
    print("V5", x["comparison"], x["diff_pp"], x["newcombe95_pp"], "p =", x["fisher_p"])
print("V5b", RES["V5b_PD_KI_BAR_vs_AR_by_suite"])
for x in V6:
    print("V6", x["comparison"], x["diff_pp"], "p =", x["fisher_p"])
for x in V7:
    print("V7", x["comparison"], x["diff_pp"], "p =", x["fisher_p"])
print("V10 격자", [(r["row"], r["common_n_from_candidates"]) for r in d_rows])
print("V10b", RES["V10b_bar_vs_pi0"])
print("V10c", RES["V10c_bar_vs_minivla"])
print("V11", RES["V11_xarm_lattice"]["n_items_consistent_with_all_four"][:6], RES.get("V11b_xarm_tests_if_min_items"))
print("V12", {m: (v["pick"], v["place"], v["pick_matches"], v["place_matches"]) for m, v in so_sum.items()})
for x in RES["V12b_so100_tests"]:
    print("V12b", x["comparison"], x["diff_pp"], x["newcombe95_pp"], "p =", x["fisher_p"])
print("V12c", RES["V12c_so100_place_paired"])
print("V13", RES["V13_param_estimate"]["ratio"], RES["V13b_action_expert_params"])
print("V14", {k: v for k, v in RES["V14_table10"].items() if k != "note"})

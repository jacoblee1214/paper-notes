"""논문(arXiv:2606.13825 v3)의 표와 본문 수치를 다시 계산한다 (numpy, scipy, 수 초).

검사 항목은 V1–V14로 번호를 붙이고 결과를 recompute_results.json에 저장한다.
표 값은 논문에서 옮겨 적었다(표 1, 2, 3, 5, 6, 그림 1 본문 수치). 시행 수나 원자료가 없는 곳은
"가정"이라고 적고 가정한 값을 그대로 쓴다. 그림에서 읽은 값은 "그림 판독 값"이다.

실행: python3 analysis/recompute.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ---------------------------------------------------------------- 표 1 (표 1, 100개 시험 인스턴스)
# 클래스: (성공 수 Constant, Adaptive, Learned), (중앙값, Q1, Q3) Constant, Adaptive, Learned
T1 = {
    "Lyapunov LMI": ((100, 47, 100), (590, 397, 998), (788, 384, 1904), (235, 182, 342)),
    "Max Cut": ((100, 100, 100), (212, 200, 224), (550, 404, 1034), (69, 65, 79)),
    "Lovasz theta": ((0, 77, 100), None, (1449, 134, 7848), (74, 69, 81)),
    "Robust KF": ((100, 100, 100), (314, 312, 317), (314, 312, 317), (26, 24, 32)),
    "Robust MPC": ((99, 100, 100), (521, 380, 708), (81, 59, 102), (16, 14, 19)),
    "Double Integrator": ((100, 49, 100), (59, 37, 87), (684, 367, 1534), (18, 10, 31)),
    "Unicycle": ((90, 0, 99), (3954, 2672, 5306), None, (63, 60, 68)),
}


def v1_table1():
    rows = {}
    for name, (succ, con, ada, lea) in T1.items():
        rec = {"success": dict(zip(["constant", "adaptive", "learned"], succ))}
        ok = True
        for label, v in (("constant", con), ("adaptive", ada), ("learned", lea)):
            if v is not None:
                ok &= v[1] <= v[0] <= v[2]
        rec["median_inside_iqr"] = bool(ok)
        ratios = {}
        for label, v in (("constant", con), ("adaptive", ada)):
            if v is not None:
                ratios[label] = v[0] / lea[0]
        rec["iter_ratio_of_medians"] = ratios
        rec["best_baseline_ratio"] = min(ratios.values())
        # 절대 IQR 폭과 상대 IQR 폭 (IQR / 중앙값)
        w = {}
        for label, v in (("constant", con), ("adaptive", ada), ("learned", lea)):
            if v is not None:
                w[label] = {"abs": v[2] - v[1], "rel": (v[2] - v[1]) / v[0]}
        rec["iqr_width"] = w
        base_abs = min(w[k]["abs"] for k in ("constant", "adaptive") if k in w)
        base_rel = min(w[k]["rel"] for k in ("constant", "adaptive") if k in w)
        rec["learned_narrower_abs_than_best_baseline"] = w["learned"]["abs"] < base_abs
        rec["learned_narrower_rel_than_best_baseline"] = w["learned"]["rel"] < base_rel
        # 성공 수 검정 (학습 대 각 기준선, Fisher 정확 검정, 100 대 100)
        f = {}
        for i, label in ((0, "constant"), (1, "adaptive")):
            a, b = succ[2], succ[i]
            p = stats.fisher_exact([[a, 100 - a], [b, 100 - b]])[1]
            f[label] = {"learned": a, "baseline": b, "fisher_p": p}
        rec["fisher"] = f
        rec["wilson_learned"] = wilson(succ[2], 100)
        rows[name] = rec
    best = max(r["iter_ratio_of_medians"].get("constant", 0) for r in rows.values())
    R["V1_table1"] = rows
    R["V1_max_ratio_vs_constant"] = best
    # V2: 성공 수가 50 미만인 열의 중앙값은 실패를 상한으로 계산한 중앙값일 수 없다
    cap = []
    for name, (succ, con, ada, lea) in T1.items():
        for label, s, v in (("constant", succ[0], con), ("adaptive", succ[1], ada)):
            if v is not None and s < 50:
                cap.append({"class": name, "arm": label, "successes": s, "reported_median": v[0],
                            "median_if_failures_charged_10000": 10000})
    R["V2_success_only_median"] = cap


# ---------------------------------------------------------------- 표 6 (시간, 100개 시험 인스턴스) 과 표 1 대조
T6 = {  # 중앙값 시간(s): constant, adaptive, learned, cosmo.jl, clarabel
    "Lyapunov LMI": (1.58, 2.42, 1.09, 0.462, 0.593),
    "Max Cut": (0.396, 1.08, 0.159, 0.0411, 0.229),
    "Lovasz theta": (None, 1.01, 0.359, 0.599, 4.27),
    "Robust KF": (4.62, 4.64, 0.884, 0.0169, 0.00208),
    "Robust MPC": (1.02, 0.821, 0.216, 0.0137, 0.00193),
    "Double Integrator": (0.971, 4.81, 0.633, 2.73, 14.1),
    "Unicycle": (31.4, None, 0.911, 1.48, 3.82),
}


def v3_table6():
    rows = {}
    for name, (c, a, l, j, cl) in T6.items():
        succ, con, ada, lea = T1[name]
        rec = {}
        if c is not None:
            rec["time_ratio_constant_over_learned"] = c / l
            rec["ms_per_iter_constant"] = 1000 * c / con[0]
        if a is not None:
            rec["time_ratio_adaptive_over_learned"] = a / l
            rec["ms_per_iter_adaptive"] = 1000 * a / ada[0]
        rec["ms_per_iter_learned"] = 1000 * l / lea[0]
        rec["cosmojl_over_learned"] = j / l
        rec["clarabel_over_learned"] = cl / l
        rec["learned_fastest"] = bool(l < min(j, cl))
        rows[name] = rec
    R["V3_table6"] = rows


# ---------------------------------------------------------------- 표 2 (배치, 각 크기 10 배치)
T2 = {  # (B=16, B=64) 중앙값, [Q1, Q3]
    "Max Cut": {"learned": ((0.248, 0.232, 0.264), (0.372, 0.361, 0.397)),
                "cosmojl": ((0.721, 0.625, 0.757), (3.02, 2.74, 3.28)),
                "clarabel": ((3.70, 3.60, 3.76), (14.9, 14.7, 17.4))},
    "Lovasz theta": {"learned": ((1.53, 1.21, 2.12), (5.14, 3.65, 8.61)),
                     "cosmojl": ((9.63, 9.13, 10.3), (72.3, 39.8, 98.2)),
                     "clarabel": ((68.6, 68.4, 69.0), (276, 275, 277))},
}


def v4_table2():
    rows = {}
    for cls, d in T2.items():
        rec = {}
        for arm, (b16, b64) in d.items():
            rec[arm] = {"B64_over_B16": b64[0] / b16[0],
                        "per_instance_B16": b16[0] / 16, "per_instance_B64": b64[0] / 64}
        rec["cosmojl_over_learned_B16"] = d["cosmojl"][0][0] / d["learned"][0][0]
        rec["cosmojl_over_learned_B64"] = d["cosmojl"][1][0] / d["learned"][1][0]
        # 표 6의 단일 인스턴스 중앙값 x B 와의 대조
        j6 = T6[cls][3]
        c6 = T6[cls][4]
        rec["table6_cosmojl_per_instance"] = j6
        rec["table6_clarabel_per_instance"] = c6
        rec["cosmojl_table6_implied_B16"] = 16 * j6
        rec["cosmojl_table6_implied_B64"] = 64 * j6
        rec["clarabel_table6_implied_B16"] = 16 * c6
        rec["clarabel_table6_implied_B64"] = 64 * c6
        rec["cosmojl_B64_observed_over_implied"] = d["cosmojl"][1][0] / (64 * j6)
        rec["cosmojl_B16_observed_over_implied"] = d["cosmojl"][0][0] / (16 * j6)
        rec["clarabel_B64_observed_over_implied"] = d["clarabel"][1][0] / (64 * c6)
        # 표 6 값으로 환산한 COSMO.jl 대비 속도비 (B=64)
        rec["speedup_B64_with_table6_implied_cosmojl"] = 64 * j6 / d["learned"][1][0]
        rows[cls] = rec
    # 본문 수치 ("1.5x, 3.4x", "4x 이상")
    R["V4_table2"] = rows
    # 단일 인스턴스 학습 solver 대비 배치 처리량
    R["V4_learned_throughput_gain_B16"] = {
        "Max Cut": T6["Max Cut"][2] / (T2["Max Cut"]["learned"][0][0] / 16),
        "Lovasz theta": T6["Lovasz theta"][2] / (T2["Lovasz theta"]["learned"][0][0] / 16),
    }
    # 부록 I의 배치 반복 수와 표 1 단일 인스턴스 분포: 배치 반복 수 = 배치 내 최대
    # 배치 8개 중 최대의 중앙값 = 154(Lovasz), 110(Max Cut). 분포함수 F 에서 F(x)^B = 0.5 를 풀면 꼬리 비율
    tails = {}
    for cls, (b8, b64) in {"Lovasz theta": (154, 434), "Max Cut": (110, 122)}.items():
        tails[cls] = {"B8_frac_above": 1 - 0.5 ** (1 / 8), "B8_value": b8,
                      "B64_frac_above": 1 - 0.5 ** (1 / 64), "B64_value": b64}
    R["V4_batch_iteration_tail"] = tails


# ---------------------------------------------------------------- 표 3 (OOD, 20 인스턴스)
T3 = [  # 클래스, 크기, 기준 반복(med,q1,q3), 학습 반복, 기준 시간, 학습 시간, 보고 속도비 반복, 시간
    ("Max Cut", "n=50", (216, 194, 232), (86, 78, 90), (0.41, 0.36, 0.44), (0.20, 0.18, 0.21), 2.53, 2.02),
    ("Max Cut", "n=250", (432, 420, 437), (204, 192, 214), (2.65, 2.58, 2.68), (1.35, 1.27, 1.42), 2.14, 1.96),
    ("Max Cut", "n=500", (593, 581, 627), (316, 308, 328), (6.66, 6.52, 7.04), (3.85, 3.75, 3.99), 1.91, 1.76),
    ("Robust KF", "N=100", (318, 314, 320), (25, 24, 37), (4.72, 4.59, 4.80), (0.84, 0.81, 0.94), 12.5, 5.49),
    ("Robust KF", "N=500", (320, 317, 321), (48, 39, 60), (6.50, 6.37, 6.55), (1.42, 1.33, 1.55), 6.69, 4.58),
    ("Robust KF", "N=1000", (320, 319, 322), (66, 62, 80), (7.06, 7.03, 7.12), (1.91, 1.82, 2.17), 4.82, 3.67),
    ("Robust MPC", "N=50", (66, 52, 91), (16, 14, 17), (0.690, 0.477, 0.881), (0.217, 0.207, 0.233), 4.16, 3.16),
    ("Robust MPC", "N=250", (46, 39, 51), (14, 13, 16), (0.536, 0.486, 0.642), (0.214, 0.191, 0.230), 3.31, 2.59),
    ("Robust MPC", "N=500", (43, 33, 56), (14, 14, 17), (0.584, 0.387, 0.633), (0.224, 0.208, 0.253), 2.64, 2.16),
]


def v5_table3():
    rows = []
    for cls, size, bi, li, bt, lt, si, st in T3:
        ri, rt = bi[0] / li[0], bt[0] / lt[0]
        rows.append({"class": cls, "size": size, "iter_ratio_of_medians": ri, "iter_reported": si,
                     "iter_diff_pct": 100 * (si - ri) / ri, "time_ratio_of_medians": rt, "time_reported": st,
                     "time_diff_pct": 100 * (st - rt) / rt,
                     "median_inside_iqr": all(x[1] <= x[0] <= x[2] for x in (bi, li, bt, lt)),
                     "iqr_disjoint_iter": bool(li[2] < bi[1]), "iqr_disjoint_time": bool(lt[2] < bt[1])})
    R["V5_table3"] = rows
    big = [r for r in T3 if r[1] not in ("n=50", "N=100", "N=50")]
    R["V5_ood_range_5x_10x"] = {"iter": [min(r[6] for r in big), max(r[6] for r in big)],
                                "time": [min(r[7] for r in big), max(r[7] for r in big)]}
    R["V5_all_range"] = {"iter": [min(r[6] for r in T3), max(r[6] for r in T3)],
                         "time": [min(r[7] for r in T3), max(r[7] for r in T3)]}
    # 크기에 따라 속도비가 줄어드는가
    mono = {}
    for cls in ("Max Cut", "Robust KF", "Robust MPC"):
        a = [r for r in T3 if r[0] == cls]
        mono[cls] = {"iter": all(a[i][6] > a[i + 1][6] for i in range(2)),
                     "time": all(a[i][7] > a[i + 1][7] for i in range(2))}
    R["V5_monotone_decrease"] = mono


# ---------------------------------------------------------------- 그림 1 (SCP 20 시나리오)
def v6_fig1():
    n = 20
    out = {}
    for name, k in (("learned", 16), ("clarabel", 12), ("cosmo", 4)):
        out[name] = {"k": k, "wilson95": wilson(k, n)}
    out["fisher_learned_vs_clarabel"] = stats.fisher_exact([[16, 4], [12, 8]])[1]
    out["fisher_learned_vs_cosmo"] = stats.fisher_exact([[16, 4], [4, 16]])[1]
    out["fisher_clarabel_vs_cosmo"] = stats.fisher_exact([[12, 8], [4, 16]])[1]
    # 부록 H: Clarabel이 실패한 시나리오 중 학습 solver가 푼 것: 학습 실패 4개는 모두 Clarabel도 실패 -> 불일치 쌍 4 대 0
    b, c = 4, 0
    out["mcnemar_exact_two_sided_learned_vs_clarabel"] = min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5))
    out["sign_test_12_of_12_faster_two_sided"] = 2 * 0.5 ** 12
    out["median_time_ratio_clarabel_over_learned"] = 76.7 / 27.9
    # 표 6 단일 서브문제 시간과 대조: 중앙값 시간 / 서브문제 시간 = 같은 시간 안에 쓴 서브문제 수 환산
    out["implied_subproblems_learned"] = 27.9 / T6["Unicycle"][2]
    out["implied_subproblems_clarabel"] = 76.7 / T6["Unicycle"][4]
    out["scp_iteration_budget"] = 20
    out["budget_time_s"] = 20 * 20
    # 실패 서술의 개수 일관성 (부록 H)
    out["learned_failures_described"] = 1 + 1 + 2
    out["clarabel_failures_described"] = 5 + 3
    R["V6_fig1"] = out


# ---------------------------------------------------------------- 표 5 구조 검산
def v7_table5():
    out = {}
    # 최대 컷 n=50: 변수 n(n+1)/2, 제약 n + n(n+1)/2, nnz(A) = 제약 수 (단위 행렬 + 대각 n)
    for n, label in ((50, "Max Cut n=50"), (250, "Max Cut n=250"), (500, "Max Cut n=500")):
        nv = n * (n + 1) // 2
        out[label] = {"n_vars": nv, "m": n + nv, "nnz_A": n + nv}
    # Lyapunov n=40: 변수 820, 제약 2 x 820. 첫 블록은 A^T P + P A 의 svec 사상, 둘째는 항등
    n = 40
    nv = n * (n + 1) // 2
    # 대각 열 e_i e_i^T: 아래삼각 항 중 i를 포함하는 쌍 수 / 비대각 열: i 또는 j를 포함하는 쌍 수 (A가 조밀하다고 가정)
    pairs = nv
    diag_col = pairs - (n - 1) * n // 2
    off_col = pairs - (n - 2) * (n - 1) // 2
    nnz = n * diag_col + (nv - n) * off_col + nv
    out["Lyapunov n=40"] = {"n_vars": nv, "m": 2 * nv, "nnz_A_dense_A_assumed": nnz, "paper_nnz_A": 64040,
                            "per_diag_col": diag_col, "per_offdiag_col": off_col}
    # Lovasz n=100: 변수 5050, 제약 1 + E + 5050, nnz(A) = 100 + E + 5050 = m + 99
    n = 100
    nv = n * (n + 1) // 2
    out["Lovasz n=100"] = {"n_vars": nv, "m_formula": "5051 + E", "nnz_A_formula": "5150 + E",
                           "paper_m_range": [7444, 7567], "paper_nnz_range": [7543, 7666],
                           "nnz_minus_m": [7543 - 7444, 7666 - 7567]}
    E_lo, E_hi = 7444 - 5051, 7567 - 5051
    npairs = n * (n - 1) // 2
    p_in = stats.binom.cdf(E_hi, npairs, 0.5) - stats.binom.cdf(E_lo - 1, npairs, 0.5)
    rng = np.random.default_rng(0)
    sim = {}
    for ninst in (20, 100, 120):
        E = rng.binomial(npairs, 0.5, size=(20000, ninst))
        span = E.max(1) - E.min(1)
        sim[str(ninst)] = {"p_all_inside_exact": float(p_in ** ninst),
                           "mean_range": float(span.mean()), "p_range_le_123_sim": float((span <= E_hi - E_lo).mean())}
    out["Lovasz_E_range"] = {"E_range_paper": [E_lo, E_hi], "range_width": E_hi - E_lo, "sd_E": math.sqrt(npairs * .25),
                             "mean_E": npairs * .5, "p_in_single": float(p_in), "by_n_instances": sim}
    # 강건 KF N=100: n = N(n_x + n_w + n_y + 4) + n_x, m = N(n_x+n_y+1) + 2N + N (n_y + 1)
    for N in (100, 500, 1000):
        n_x, n_w, n_y = 4, 2, 2
        nv = N * (n_x + n_w + n_y + 4) + n_x
        m = N * (n_x + n_y + 1) + 2 * N + N * (n_y + 1)
        nnzA = N * (4 + 6 + 4 + 2 + 2 + 4) + 2 * N + N * 3
        nnzP = N * (n_w + 1)
        out[f"Robust KF N={N}"] = {"n_vars": nv, "m": m, "nnz_A": nnzA, "nnz_P": nnzP}
    out["Robust KF N=100"].update({"paper": [1204, 1200, 300, 2700]})
    # 강건 MPC N=50: x 6(N+1), u 3N, tau N+1
    for N in (50, 250, 500):
        nv = 6 * (N + 1) + 3 * N + N + 1
        m = 6 + 6 * N + 12 * (N + 1) + 6 * N + 4 * N + 7
        nnzP = 6 * (N + 1) + 3 * N
        out[f"Robust MPC N={N}"] = {"n_vars": nv, "m": m, "nnz_P": nnzP}
    out["Robust MPC N=50"].update({"paper": [507, 1425, 456, 4125]})
    R["V7_table5"] = out


# ---------------------------------------------------------------- 그림 2 메모리 산술 (float64)
def v8_fig2():
    out = []
    for n in (1000, 2000, 5000, 10000, 20000, 50000):
        m = n // 2
        dense_kkt = 8 * (n + m) ** 2 / 2 ** 20
        dense_M = 8 * n * n / 2 ** 20
        dense_A = 8 * m * n / 2 ** 20
        nnzA = 0.01 * m * n
        csr = nnzA * (8 + 8) / 2 ** 20  # float64 값 + int64 열 색인
        out.append({"n": n, "m": m, "nnz_A": nnzA, "dense_KKT_MiB": dense_kkt, "dense_M_MiB": dense_M,
                    "dense_A_MiB": dense_A, "one_CSR_copy_MiB": csr, "three_CSR_copies_MiB": 3 * csr})
    R["V8_fig2_memory_arithmetic"] = out
    # 그림 판독 값 (피크 메모리, MB): 우리 방법
    read = {1000: 2.2, 2000: 4.5, 5000: 12, 10000: 30, 20000: 80, 50000: 300}
    ns = np.array(list(read))
    ys = np.array(list(read.values()))
    slope_all = np.polyfit(np.log(ns), np.log(ys), 1)[0]
    slope_tail = np.polyfit(np.log(ns[-3:]), np.log(ys[-3:]), 1)[0]
    R["V8_fig2_ours_loglog_slope_read"] = {"read_MB": read, "slope_all": slope_all, "slope_last3": slope_tail,
                                           "note": "그림 판독 값. nnz(A)=0.005 n^2 이면 선형은 1, nnz 지배는 2"}


# ---------------------------------------------------------------- 학습 규모
def v9_training():
    ep = {"Lyapunov LMI": (30, 400), "Max Cut": (60, 300), "Lovasz theta": (50, 100), "Robust KF": (100, 100),
          "Robust MPC": (60, 300), "Double Integrator": (60, 150), "Unicycle": (60, 150)}
    out = {}
    for k, (e, K) in ep.items():
        steps = e * 100 // 5
        out[k] = {"epochs": e, "K": K, "adam_steps": steps, "unrolled_iters_per_step_x_batch": K * 5}
    R["V9_training_scale"] = out
    R["V9_unicycle_epoch_29s"] = {"epoch_s": 29, "epochs": 60, "total_min": 29 * 60 / 60}


# ---------------------------------------------------------------- v1 대 v3 표 1
def v10_versions():
    v1 = {"Lyapunov LMI": (1179, 1018, 1.16), "Max Cut": (471, 177, 2.66), "Lovasz theta": (10925, 173, 63.1),
          "Robust KF": (312, 40, 7.80), "Robust MPC": (180, 20, 9.00), "Double Integrator": (396, 35, 11.3),
          "Unicycle": (121, 33, 3.67)}
    out = {}
    for k, (c, l, s) in v1.items():
        out[k] = {"v1_cosmo": c, "v1_learned": l, "ratio_calc": c / l, "ratio_reported": s,
                  "v3_learned": T1[k][3][0], "v3_adaptive": None if T1[k][2] is None else T1[k][2][0],
                  "v3_learned_over_v1_learned": T1[k][3][0] / l}
    R["V10_v1_table1"] = out
    # v1 시간 (표 1, 단위 s, 학습 시간 대비 속도비)
    v1t = {"Lyapunov LMI": (11.44, 9.822, 1.16), "Max Cut": (1.401, 0.5353, 2.62), "Lovasz theta": (32.14, 0.6373, 50.4),
           "Robust KF": (19.14, 3.896, 4.91), "Robust MPC": (4.534, 0.9748, 4.65),
           "Double Integrator": (50.48, 4.048, 12.5), "Unicycle": (10.99, 2.013, 5.46)}
    R["V10_v1_time_ratio"] = {k: {"calc": a / b, "reported": s} for k, (a, b, s) in v1t.items()}


# ---------------------------------------------------------------- 요약 수치
def v11_abstract():
    R["V11_abstract_claims"] = {
        "up_to_60x_vs_constant_unicycle": T1["Unicycle"][1][0] / T1["Unicycle"][3][0],
        "up_to_60x_vs_adaptive_lovasz": T1["Lovasz theta"][2][0] / T1["Lovasz theta"][3][0],
        "14x_batch_lovasz_B64": 72.3 / 5.14,
        "8x_batch_maxcut_B64": 3.02 / 0.372,
        "fig1_16_vs_4_vs_12": [16, 4, 12],
    }


def main():
    v1_table1()
    v3_table6()
    v4_table2()
    v5_table3()
    v6_fig1()
    v7_table5()
    v8_fig2()
    v9_training()
    v10_versions()
    v11_abstract()
    OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

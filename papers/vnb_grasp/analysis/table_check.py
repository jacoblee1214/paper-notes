"""VNB(arXiv:2604.25897v2) Table III 재계산과 통계 검정.

입력: 논문 Table III 값(아래 상수, 논문 원문 그대로 옮김). 외부 데이터 없음. CPU, 1초 미만.
출력: analysis/table_check_results.json, 표준 출력.

  (1) 에피소드 수 n 추정: 보고된 백분율이 k/n 의 반올림인지 n별로 확인
  (2) P̂emp = 1 - PertSurv 관계 확인 (표 III 각 칸)
  (3) 보정 오차 |ΔP̂| = |P̂bel - P̂emp| 재계산과 부호(과소/과대 예측)
  (4) 계획 시간 비율, 평균 보정 오차, 최대값
  (5) Fisher 정확 검정(방법 쌍, 체제별), Wilson 95% 구간, Holm 보정
  (6) 보정 오차 Welch 검정(체제 3개뿐이라 검정력이 매우 낮음)
  (7) 실물 12/12 정확 구간
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "table_check_results.json"
REG = ["nominal", "wide", "bimodal"]

# Table III (논문 v2). 순서: nominal, wide, bimodal
T = {
    "Pf":        {"SR": [83, 44, 56], "Robust": [67, 39, 28], "PertSurv": [57, 31, 29],
                  "Pbel": [None, None, None], "Pemp": [0.43, 0.69, 0.71], "dP": [None, None, None], "Time": [49.6, 32.6, 41.7]},
    "Gauss":     {"SR": [100, 71, 54], "Robust": [75, 63, 42], "PertSurv": [69, 56, 36],
                  "Pbel": [0.22, 0.31, 0.57], "Pemp": [0.31, 0.44, 0.64], "dP": [0.09, 0.13, 0.07], "Time": [3.2, 3.4, 4.3]},
    "Gauss-CVaR": {"SR": [100, 71, 54], "Robust": [75, 63, 42], "PertSurv": [69, 56, 36],
                  "Pbel": [0.22, 0.31, 0.57], "Pemp": [0.31, 0.44, 0.64], "dP": [0.09, 0.13, 0.07], "Time": [3.1, 3.4, 4.3]},
    "CEM":       {"SR": [100, 79, 42], "Robust": [63, 67, 33], "PertSurv": [55, 59, 30],
                  "Pbel": [1.00, 1.00, 1.00], "Pemp": [0.45, 0.42, 0.70], "dP": [0.55, 0.58, 0.29], "Time": [5.5, 5.4, 5.1]},
    "VNB":       {"SR": [100, 79, 54], "Robust": [79, 67, 38], "PertSurv": [73, 58, 35],
                  "Pbel": [0.18, 0.28, 0.54], "Pemp": [0.27, 0.42, 0.65], "dP": [0.09, 0.14, 0.11], "Time": [8.5, 7.1, 6.8]},
}
# Table I/Table V 값 (µf 가정 검산용)
TABLE_I = {"PLA primitive": (0.30, 0.50, (0.32, 0.42)), "LDPE bottle": (0.25, 0.45, (0.30, 0.40)),
           "Alum./steel can": (0.20, 0.35, (0.26, 0.35)), "Tennis ball (felt)": (0.50, 0.80, (0.42, 0.53))}
REGIMES_MU_O = {"Nominal U[0.4,1.0]": (0.4, 1.0, (0.37, 0.59)), "Wide U[0.15,1.2]": (0.15, 1.2, (0.23, 0.65))}


def round_half_up(x):
    return int(np.floor(x + 0.5))


def consistent_n(values, n):
    """보고된 정수 백분율 values 모두가 k/n 을 반올림한 값인지 (k는 0..n)."""
    for v in values:
        if v is None:
            continue
        ok = any(round_half_up(100 * k / n) == v for k in range(n + 1))
        if not ok:
            return False
    return True


def main():
    res = {}

    # (1) n 추정: SR과 Robust 칸(에피소드 단위 비율)이 모두 k/n 으로 설명되는 n
    cells = []
    for m in T:
        for key in ("SR", "Robust"):
            cells += [(m, key, r, v) for r, v in zip(REG, T[m][key])]
    fits = {}
    for n in range(20, 31):
        bad = [c for c in cells if not any(round_half_up(100 * k / n) == c[3] for k in range(n + 1))]
        fits[n] = len(bad)
    res["n_fit_bad_cells_by_n"] = fits
    bad_24 = [f"{m}-{key}-{r}={v}" for (m, key, r, v) in cells
              if not any(round_half_up(100 * k / 24) == v for k in range(25))]
    res["cells_not_k_over_24"] = bad_24

    # (2) P̂emp = 1 - PertSurv
    dev = []
    for m in T:
        for i, r in enumerate(REG):
            if T[m]["Pemp"][i] is None:
                continue
            implied = 1 - T[m]["PertSurv"][i] / 100
            if abs(implied - T[m]["Pemp"][i]) > 0.005:
                dev.append((m, r, round(implied, 3), T[m]["Pemp"][i]))
    res["pemp_vs_1minus_pertsurv_mismatch"] = dev

    # (3) 보정 오차 재계산과 부호
    sign = {}
    for m in ("Gauss", "CEM", "VNB"):
        rows = []
        for i, r in enumerate(REG):
            pb, pe = T[m]["Pbel"][i], T[m]["Pemp"][i]
            rows.append({"regime": r, "Pbel": pb, "Pemp": pe, "signed_bel_minus_emp": round(pb - pe, 3),
                         "abs": round(abs(pb - pe), 3), "reported_dP": T[m]["dP"][i]})
        sign[m] = rows
    res["calibration_rows"] = sign
    mean_dp = {m: round(float(np.mean(T[m]["dP"])), 3) for m in ("Gauss", "CEM", "VNB")}
    max_dp = {m: max(T[m]["dP"]) for m in ("Gauss", "CEM", "VNB")}
    res["mean_abs_dP"] = mean_dp
    res["max_abs_dP"] = max_dp

    # (4) 시간 비율
    pf_vnb = [round(a / b, 2) for a, b in zip(T["Pf"]["Time"], T["VNB"]["Time"])]
    vnb_cem = [round(a / b, 2) for a, b in zip(T["VNB"]["Time"], T["CEM"]["Time"])]
    vnb_g = [round(a / b, 2) for a, b in zip(T["VNB"]["Time"], T["Gauss"]["Time"])]
    res["time_ratio_Pf_over_VNB"] = pf_vnb
    res["time_ratio_VNB_over_CEM"] = vnb_cem
    res["time_ratio_VNB_over_Gauss"] = vnb_g
    # 부록 Table V (per-episode time): Pf 65-72 s, Vnb 4-6 s
    res["appendixV_time_ratio_range"] = [round(65 / 6, 1), round(72 / 4, 1)]

    # (5) Fisher 정확 검정 (n = 24 가정; 가정이 틀리면 결과 재해석 필요)
    n = 24
    def fisher(a_pct, b_pct):
        a = round_half_up(a_pct * n / 100)
        b = round_half_up(b_pct * n / 100)
        tab = [[a, n - a], [b, n - b]]
        return float(stats.fisher_exact(tab)[1]), (a, b)

    comparisons = [("VNB", "Gauss"), ("VNB", "CEM"), ("VNB", "Pf")]
    pvals = []
    fisher_rows = []
    for (x, y) in comparisons:
        for i, r in enumerate(REG):
            p, counts = fisher(T[x]["SR"][i], T[y]["SR"][i])
            pvals.append(p)
            fisher_rows.append({"pair": f"{x} vs {y}", "regime": r, "SR_counts(of 24)": counts, "p_fisher": round(p, 4)})
    # Holm
    order = np.argsort(pvals)
    m_tests = len(pvals)
    holm = [None] * m_tests
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m_tests - rank) * pvals[idx])
        running = max(running, adj)
        holm[idx] = running
    for row, h in zip(fisher_rows, holm):
        row["p_holm"] = round(h, 4)
    res["fisher_SR_n24"] = fisher_rows

    # Wilson 95% 구간 (SR)
    def wilson(k, n_):
        z = 1.959963984540054
        p = k / n_
        den = 1 + z * z / n_
        c = (p + z * z / (2 * n_)) / den
        half = z * np.sqrt(p * (1 - p) / n_ + z * z / (4 * n_ * n_)) / den
        return round(c - half, 3), round(c + half, 3)
    wil = {}
    for m in ("VNB", "Gauss", "CEM", "Pf"):
        wil[m] = [wilson(round_half_up(v * n / 100), n) for v in T[m]["SR"]]
    res["wilson95_SR_n24"] = wil

    # Vnb vs Gauss 차이의 Newcombe 구간은 생략. 차이 크기만 기록
    res["SR_diff_VNB_minus_Gauss_pp"] = [T["VNB"]["SR"][i] - T["Gauss"]["SR"][i] for i in range(3)]
    res["SR_diff_VNB_minus_CEM_pp"] = [T["VNB"]["SR"][i] - T["CEM"]["SR"][i] for i in range(3)]

    # (6) 보정 오차 Welch (체제 3개; 방법별 3값)
    w = stats.ttest_ind(T["VNB"]["dP"], T["CEM"]["dP"], equal_var=False)
    w2 = stats.ttest_ind(T["VNB"]["dP"], T["Gauss"]["dP"], equal_var=False)
    res["welch_dP_VNB_vs_CEM"] = {"t": round(float(w.statistic), 3), "p": round(float(w.pvalue), 4)}
    res["welch_dP_VNB_vs_Gauss"] = {"t": round(float(w2.statistic), 3), "p": round(float(w2.pvalue), 4)}

    # (6b) 체제 3개 대 3개의 SR 차이를 Cochran-Mantel-Haenszel 대신 평균 차이 부호만 보고
    res["sign_pattern_dP_VNB_lt_CEM_all_regimes"] = all(a < b for a, b in zip(T["VNB"]["dP"], T["CEM"]["dP"]))

    # (7) 실물 12/12
    res["hardware_12of12_exact95_lower_two_sided"] = round(float(stats.beta.ppf(0.025, 12, 1)), 4)
    res["hardware_12of12_exact95_lower_one_sided"] = round(float(0.05 ** (1 / 12)), 4)
    res["hardware_fisher_p_12v12_vs_12v12"] = 1.0

    # (8) µf 검산: Table I, §V-D 범위가 µf=0.35 인지 µf=1.0 인지
    chk = {}
    for name, (lo, hi, rep) in TABLE_I.items():
        chk[name] = {
            "reported": rep,
            "uf_0.35": [round(float(np.sqrt(0.35 * lo)), 2), round(float(np.sqrt(0.35 * hi)), 2)],
            "uf_1.0": [round(float(np.sqrt(1.0 * lo)), 2), round(float(np.sqrt(1.0 * hi)), 2)],
        }
    for name, (lo, hi, rep) in REGIMES_MU_O.items():
        chk[name] = {
            "reported": rep,
            "uf_0.35": [round(float(np.sqrt(0.35 * lo)), 2), round(float(np.sqrt(0.35 * hi)), 2)],
            "uf_1.0": [round(float(np.sqrt(1.0 * lo)), 2), round(float(np.sqrt(1.0 * hi)), 2)],
        }
    res["mu_eff_check"] = chk

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()

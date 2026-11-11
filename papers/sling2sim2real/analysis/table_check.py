"""Sling2Sim2Real(arXiv:2607.23268v1) 표 수치 재계산: 셀 평균, 'All targets' 산술, Welch 검정, 손실 감소율, N/A 집계.

실행: python3 analysis/table_check.py  -> analysis/table_check_results.json
가정: 표의 '±'는 셀당 5회 시도의 표준편차(SD)로 둔다. 논문은 SD인지 표준오차인지 적지 않았다.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "table_check_results.json"
N = 5  # 셀당 시도 수 (§IV-B: five launch trials per method)
G = ["172.5", "192.5", "212.5"]
NA = None

# Table I: (셀 3개 평균, 셀 3개 SD, 보고된 All targets 평균, 보고된 All targets ±)
T1 = {
    "B1 Package SI": ([13.00, 1.50, 9.50], [1.87, 1.22, 2.50], 8.00, 1.94),
    "B1 Sling2Sim2Real": ([11.00, 1.00, 9.50], [2.00, 1.22, 4.30], 7.17, 2.68),
    "B2 Package SI": ([60.00, 34.50, 11.00], [2.74, 3.32, 6.44], 35.17, 4.58),
    "B2 Sling2Sim2Real": ([15.50, 3.50, 23.50], [1.87, 2.00, 2.55], 14.17, 2.16),
    "B3 Sling2Sim2Real": ([35.50, 18.00, 3.50], [1.87, 1.87, 3.39], 19.00, 2.47),
}
# Table III/IV Band 1 (All targets 열 포함)
T4 = {
    "DEL": ([19.00, 11.50, 3.00], [1.22, 1.22, 1.87], 11.17, 0.85),
    "BOL": ([16.50, 3.00, 8.00], [1.22, 1.87, 6.00], 9.17, 2.12),
    "AdamL": ([11.00, 20.00, 34.50], [2.00, 3.16, 5.79], 21.83, 2.31),
    "S2S (full)": ([11.00, 1.00, 9.50], [2.00, 1.22, 4.30], 7.17, 2.68),
    "S2S-c,-m": ([15.00, 6.00, 0.50], [1.58, 1.22, 2.45], 7.83, 5.39),
    "S2S-m": ([29.50, 15.00, 16.50], [2.92, 3.87, 2.00], 20.33, 7.18),
    "S2S-c": ([22.50, 1.00, 6.50], [3.87, 1.22, 5.39], 10.00, 9.91),
    "S2S-Lwork": ([33.00, 30.00, 20.50], [1.00, 1.58, 2.45], 27.83, 5.62),
}


def pooled_sd_rms(sds):
    # 셀 분산의 평균의 제곱근(셀 내 합동 SD 추정)
    return float(np.sqrt(np.mean(np.square(sds))))


def all_targets_stats(means, sds):
    """셀 3개 x 5회 = 15회 시도의 평균과 SD. 셀 내 분산 평균 + 셀 평균들의 분산(모집단형)."""
    within = np.mean(np.square(sds))
    between = np.var(means)  # ddof=0
    total_var = (within + between) * 15 / 14  # 15개 표본 SD 보정
    return float(np.mean(means)), float(np.sqrt(total_var))


def welch(m1, s1, m2, s2, n=N):
    t, p = stats.ttest_ind_from_stats(m1, s1, n, m2, s2, n, equal_var=False)
    return float(t), float(p)


def main():
    res = {"assumption": "± = 시도별 표준편차(SD), n = 5 per cell", "table_I": {}, "table_IV": {}, "checks": {}}

    # 1) All targets 산술
    res["checks"]["all_targets_arithmetic"] = {}
    for name, (means, sds, rep, rep_pm) in {**T1, **T4}.items():
        calc = float(np.mean(means))
        res["checks"]["all_targets_arithmetic"][name] = {
            "cell_mean_avg": round(calc, 3), "reported": rep, "diff": round(rep - calc, 3),
            "pm_reported": rep_pm,
            "pm_rms_of_cell_SD": round(pooled_sd_rms(sds), 3),
        }

    # 2) Table I 셀별 Welch (Sling2Sim2Real vs Package SI, Band 2 중심)
    b2s, b2sd = T1["B2 Sling2Sim2Real"][0], T1["B2 Sling2Sim2Real"][1]
    b2p, b2pd = T1["B2 Package SI"][0], T1["B2 Package SI"][1]
    res["table_I"]["band2_cellwise_welch"] = []
    for i, g in enumerate(G):
        t, p = welch(b2s[i], b2sd[i], b2p[i], b2pd[i])
        res["table_I"]["band2_cellwise_welch"].append(
            {"g": g, "S2S": b2s[i], "PkgSI": b2p[i], "diff_S2S_minus_PkgSI": round(b2s[i] - b2p[i], 2),
             "t": round(t, 2), "p": float(f"{p:.3g}")})
    # Band 2 전체 15회: 셀 평균들과 SD로 근사
    m_s, sd_s = all_targets_stats(np.array(b2s), np.array(b2sd))
    m_p, sd_p = all_targets_stats(np.array(b2p), np.array(b2pd))
    t, p = welch(m_s, sd_s, m_p, sd_p, n=15)
    res["table_I"]["band2_all_targets_welch_n15"] = {
        "S2S_mean": round(m_s, 2), "S2S_sd_est": round(sd_s, 2),
        "PkgSI_mean": round(m_p, 2), "PkgSI_sd_est": round(sd_p, 2),
        "t": round(t, 2), "p": float(f"{p:.3g}"),
        "rel_reduction_pct": round(100 * (m_p - m_s) / m_p, 1),
    }
    # Band 1: 전체 평균 차이 0.83 cm, n=15 근사
    b1s, b1sd = T1["B1 Sling2Sim2Real"][0], T1["B1 Sling2Sim2Real"][1]
    b1p, b1pd = T1["B1 Package SI"][0], T1["B1 Package SI"][1]
    m1s, s1s = all_targets_stats(np.array(b1s), np.array(b1sd))
    m1p, s1p = all_targets_stats(np.array(b1p), np.array(b1pd))
    t, p = welch(m1s, s1s, m1p, s1p, n=15)
    res["table_I"]["band1_all_targets_welch_n15"] = {
        "S2S_mean": round(m1s, 2), "PkgSI_mean": round(m1p, 2), "diff": round(m1s - m1p, 2),
        "t": round(t, 2), "p": float(f"{p:.3g}")}
    # Band 2 g=212.5: Package SI가 더 낮다
    res["table_I"]["band2_g212_reversal_cm"] = round(b2s[2] - b2p[2], 2)

    # 3) 셀 가중 대비: Table I 전체 9셀 중 승패
    wins = []
    for band in ("B1", "B2"):
        for i in range(3):
            s = T1[f"{band} Sling2Sim2Real"][0][i]
            pk = T1[f"{band} Package SI"][0][i]
            wins.append((band, G[i], "S2S better" if s < pk else ("tie" if s == pk else "PkgSI better")))
    res["table_I"]["cell_outcomes_band1_2"] = wins
    res["table_I"]["N_NA_cells"] = {"Real2Sim2Real": 9, "Package SI": 3, "Sling2Sim2Real": 0}

    # 4) Table II 손실 감소
    loss = {"B1": (1.9785, 2.4718), "B2": (2.1749, 2.3267), "B3": (0.9416, 3.4290)}
    res["table_II_loss_reduction_pct"] = {k: round(100 * (b - a) / b, 1) for k, (a, b) in loss.items()}

    # 5) Table III/IV 대비 Band 1 (전체 열 기준, n=15 근사): 제안 vs 각 방법
    res["table_IV"]["vs_full_welch_n15"] = {}
    fs_m, fs_s = all_targets_stats(np.array(T4["S2S (full)"][0]), np.array(T4["S2S (full)"][1]))
    for name in ("DEL", "BOL", "AdamL", "S2S-c,-m", "S2S-m", "S2S-c", "S2S-Lwork"):
        m, s = all_targets_stats(np.array(T4[name][0]), np.array(T4[name][1]))
        t, p = welch(fs_m, fs_s, m, s, n=15)
        res["table_IV"]["vs_full_welch_n15"][name] = {"mean": round(m, 2), "t": round(t, 2),
                                                      "p": float(f"{p:.3g}")}

    # 6) 검출력: 셀당 n=5, SD 2 cm에서 1 cm 차이를 잡으려면?
    res["power"] = {"n_per_group_for_1cm_diff_SD2_80pct": power_two_sample(1.0, 2.0)}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1)[:3000])


def power_two_sample(delta, sd, alpha=0.05, power=0.8):
    """두 독립 표본 t 근사: n = 2 (z_{1-α/2} + z_{power})² σ² / Δ²."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * z * z * sd * sd / (delta * delta)))


if __name__ == "__main__":
    main()

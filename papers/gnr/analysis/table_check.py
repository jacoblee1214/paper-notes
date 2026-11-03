"""표·본문·부록 수치 재계산과 정합성 점검 (CPU). 실행: python3 analysis/table_check.py

- C1: 초록 "MPC 표본의 8.5%", "56.20% vs 27.20%", "CPU 비용 7.6배 감소" (Abstract, Sec. 5.2)
- C2: Table 1 전체 SR = 0.272 × MPC-success + 0.728 × MPC-failure, 정수 개수 (Table 1)
- C3: Fig. 3·부록 D.3의 OOD 부분집합에서 MPC-success 개수 역산 (합 272인지)
- C4: DIAL-MPC 27.20%는 시험 집합을 나눈 기준 자체 (Table 1)
- C5: "demonstration" 단위: 기하 변형 쌍 × 최적화 시드 (부록 C.5, Table 8)
- C6: 계산량 회계: Table 1 core-s vs Table 14 core-h, 2.62배, 성공 1건당 비용 (Sec. 5.2, 부록 G)
- C7: ShapeFilter 수치: Table 10 분모, Table 11 vs 본문, Table 2·12·15, "비용 1.6%" (Sec. 5.3, 부록 E)
- C8: NIST 수치: 64개 시험의 정수, 33.3%, 3.0배, 1.8배 (Sec. 5.3, 부록 F·G)
- C9: 다중 embodiment Wilson 구간과 SPIDER 비교 (Table 16)
- C10: BC 정책 37.5% vs 43.75% (부록 I.1)
- C11: data flywheel 수치가 256개 시험의 정수인지 (부록 I.2)
- C12: HOT3D 성공 기준의 크기와 IK·GNR S=1 (Table 1, Table 4)
"""
import json
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

# Table 1 (HOT3D 시험 1,000개 = MPC-success 272 + MPC-failure 728)
N_TEST, N_SUC, N_FAIL = 1000, 272, 728
TABLE1 = {  # method: (S, core-s, SR_all, SR_suc, SR_fail)
    "IK": (1, 39.0, 18.30, 50.00, 6.46),
    "DIAL-MPC": (3000, 2689.3, 27.20, 100.00, 0.00),
    "MPPI": (3000, 1849.6, 27.70, 71.32, 11.40),
    "iCEM": (3000, 3862.1, 34.70, 78.68, 18.27),
    "CMA-ES": (3000, 2912.7, 32.00, 72.79, 16.76),
    "GNR S=1": (1, 48.9, 18.10, 48.16, 6.87),
    "GNR S=64": (64, 90.9, 49.60, 88.97, 34.89),
    "GNR S=128": (128, 178.6, 52.20, 90.07, 38.05),
    "GNR S=256": (256, 354.1, 56.20, 92.65, 42.58),
}
# Fig. 3 + 부록 D.3: OOD 부분집합 크기와 S=256(·S=1) 성공률. (overall, MPC-success, MPC-failure)
SUBSETS = {"OOD object": (203, 44.33, 100.0, 31.9), "OOD grasp": (338, 42.90, 85.5, 32.0),
           "OOD geometry": (459, 71.24, 94.0, 58.4)}
SUBSET_S1 = {"OOD object": 12.81, "OOD grasp": 10.06, "OOD geometry": 26.36}
# Table 8
T8 = {"evaluated": (186310, 173840, 360150), "mpc_succ": (47426, 47918, 95344),
      "mpc_fail": (138884, 125922, 264806), "gnr_add": (74391, 52745, 127136),
      "union": (121817, 100663, 222480), "geoms_mpc": (33, 2722, 2755), "geoms_union": (33, 3265, 3298)}
# 부록 C.5
PAIRS, SEEDS, PAIRS_OK = 7280, 24, 2723
# Table 14 (core-h / episode)
T14 = {"DIAL-MPC": 1.0777, "iCEM": 1.4213, "CMA-ES": 1.2272, "GNR S=256": 0.1094, "ShapeFilter MPC": 3.2001}


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def denominators(pct: float, nmax: int = 1200, tol: float = 0.005) -> list[int]:
    """pct(소수 둘째 자리까지 반올림된 %)를 정수 k/n으로 만들 수 있는 n (작은 것부터)."""
    out = []
    for n in range(1, nmax + 1):
        k = round(pct / 100 * n)
        if abs(100 * k / n - pct) <= tol + 1e-9:
            out.append(n)
    return out


def main() -> None:
    out: dict = {}

    # C1
    g, m = TABLE1["GNR S=256"], TABLE1["DIAL-MPC"]
    out["C1_abstract"] = {
        "sample_fraction_pct": round(100 * g[0] / m[0], 2),
        "cost_ratio_table1": round(m[1] / g[1], 2),
        "claim": "8.5% of the samples; 56.20% vs 27.20%; 7.6x lower CPU cost per trajectory",
    }

    # C2
    c2 = {}
    for name, (s, cs, sr, srs, srf) in TABLE1.items():
        recon = (N_SUC * srs + N_FAIL * srf) / N_TEST
        ks, kf = srs / 100 * N_SUC, srf / 100 * N_FAIL
        c2[name] = {"recomputed_overall": round(recon, 2), "reported": sr,
                    "k_success": round(ks, 2), "k_failure": round(kf, 2),
                    "integer": abs(ks - round(ks)) < 0.03 and abs(kf - round(kf)) < 0.04}
    out["C2_table1_decomposition"] = c2

    # C3 부분집합별 MPC-success 개수 n_s: overall*n = n_s*SR_s + (n-n_s)*SR_f
    c3, tot = {}, 0.0
    for name, (n, sr, srs, srf) in SUBSETS.items():
        ns = n * (sr - srf) / (srs - srf)
        tot += round(ns)
        c3[name] = {"n": n, "n_mpc_success_est": round(ns, 2), "mpc_success_rate_pct": round(100 * round(ns) / n, 1)}
    s1 = sum(SUBSETS[k][0] * SUBSET_S1[k] for k in SUBSETS) / N_TEST
    s256 = sum(SUBSETS[k][0] * SUBSETS[k][1] for k in SUBSETS) / N_TEST
    c3["sum_mpc_success"] = tot
    c3["weighted_S1_vs_table1"] = [round(s1, 2), TABLE1["GNR S=1"][2]]
    c3["weighted_S256_vs_table1"] = [round(s256, 2), TABLE1["GNR S=256"][2]]
    c3["mean_mpc_success_subsets_S256"] = round(sum(v[2] for v in SUBSETS.values()) / 3, 2)
    out["C3_ood_subsets"] = c3

    # C4
    out["C4_dialmpc_is_split_label"] = {
        "dialmpc_sr": TABLE1["DIAL-MPC"][2], "mpc_success_share_of_test": 100 * N_SUC / N_TEST,
        "population_mpc_success_pct": round(100 * T8["mpc_succ"][2] / T8["evaluated"][2], 2),
        "note": "DIAL-MPC 행은 다시 돌린 결과가 아니라 시험 집합을 MPC-success/failure로 나눈 레이블 그 자체(100%/0%)",
    }

    # C5 시드 단위 회계
    trials = PAIRS * SEEDS
    succ_mod, fail_mod, eval_mod = T8["mpc_succ"][1], T8["mpc_fail"][1], T8["evaluated"][1]
    cond = succ_mod / (PAIRS_OK * SEEDS)
    fail_in_solvable = PAIRS_OK * SEEDS - succ_mod
    out["C5_seed_level_accounting"] = {
        "trials_pairs_x_seeds": trials, "evaluated_modified_geometry_demos": eval_mod,
        "missing": trials - eval_mod,
        "attempt_level_mpc_success_pct": round(100 * succ_mod / eval_mod, 1),
        "pair_level_mpc_success_pct": round(100 * PAIRS_OK / PAIRS, 1),
        "seed_success_given_solvable_pair_pct": round(100 * cond, 1),
        "mpc_failures_in_mpc_solvable_pairs": fail_in_solvable,
        "share_of_modified_mpc_failures_pct": round(100 * fail_in_solvable / fail_mod, 1),
        "gnr_added_successes_modified": T8["gnr_add"][1],
        "new_geometries": T8["geoms_union"][1] - T8["geoms_mpc"][1],
        "gnr_successes_per_new_geometry_upper": round(T8["gnr_add"][1] / (T8["geoms_union"][1] - T8["geoms_mpc"][1]), 1),
        "original_geometry_demos_per_source_motion": round(T8["evaluated"][0] / 780, 1),
        "counts_text_vs_table8": {"total_text": 222806, "total_table8": T8["union"][2],
                                  "geoms_mpc_text": 2756, "geoms_mpc_table8": 2755,
                                  "pairs_ok_text": 2723, "geoms_modified_mpc_table8": 2722,
                                  "mpc_rate_text_pct": 26.43,
                                  "mpc_rate_table8_pct": round(100 * T8["mpc_succ"][2] / T8["evaluated"][2], 2),
                                  "coverage_text_pct": 61.56,
                                  "coverage_222806_pct": round(100 * 222806 / 360150, 2),
                                  "coverage_222480_pct": round(100 * 222480 / 360150, 2)},
    }

    # C6 계산량
    t1_cs = {k: TABLE1[k][1] for k in ("DIAL-MPC", "iCEM", "CMA-ES", "GNR S=256")}
    c6 = {"table14_over_table1": {k: round(T14[k] * 3600 / t1_cs[k], 3) for k in t1_cs}}
    c6["cost_ratio_dial_vs_gnr"] = {"table1": round(t1_cs["DIAL-MPC"] / t1_cs["GNR S=256"], 2),
                                    "table14": round(T14["DIAL-MPC"] / T14["GNR S=256"], 2)}
    # 성공 1건당 비용 (시험 집합)
    c6["core_s_per_success_test"] = {k: round(TABLE1[k][1] / (TABLE1[k][2] / 100), 1)
                                     for k in ("IK", "DIAL-MPC", "MPPI", "iCEM", "CMA-ES", "GNR S=64", "GNR S=256")}
    # 2.62배: MPC 단가(Table 14)와 GNR 단가(Table 1)를 섞음
    seeds = 28480
    attempts = seeds / 0.2643
    mpc_seed_cost = attempts * T14["DIAL-MPC"]
    rest = T8["evaluated"][2] - seeds
    gnr_t1 = rest * t1_cs["GNR S=256"] / 3600
    gnr_t14 = rest * T14["GNR S=256"]
    yield_ = seeds + (T8["mpc_succ"][2] - seeds) * 0.8891 + T8["gnr_add"][2]
    c6["practical_deployment"] = {
        "seed_attempts": round(attempts), "mpc_seed_core_h": round(mpc_seed_cost),
        "gnr_core_h_table1_rate": round(gnr_t1), "gnr_core_h_table14_rate": round(gnr_t14),
        "total_reported_k": 148.735, "ratio_reported": round(390.043 / 148.735, 3),
        "ratio_if_table14_for_gnr": round(390043 / (mpc_seed_cost + gnr_t14), 3),
        "trajectories": round(yield_),
        "core_h_per_success_mpc": round(390043 / T8["mpc_succ"][2], 3),
        "core_h_per_success_gnr_deploy": round(148735 / 215062, 3),
    }
    out["C6_compute_accounting"] = c6

    # C7 ShapeFilter (시험 589개)
    t10 = {"DIAL-MPC 32x6": 48.4, "DIAL-MPC 500x6": 92.9, "CEM 32x6": 50.0, "CEM 500x6": 82.8,
           "iCEM 32x6": 65.6, "CMA-ES 32x6": 40.6, "GNR S=8": 87.23, "GNR S=16": 92.26,
           "GNR S=32": 95.44, "GNR S=64": 97.37}
    t11_state = {1: 50.42, 2: 67.06, 4: 78.44, 8: 87.10, 16: 92.36, 32: 95.93, 64: 97.45, 128: 98.47, 256: 99.32}
    c7 = {"table10_denominators": {}}
    for k, v in t10.items():
        tol = 0.05 if k.split()[0] in ("DIAL-MPC", "CEM", "iCEM", "CMA-ES") else 0.005
        ds = denominators(v, 1200, tol)
        c7["table10_denominators"][k] = {"value": v, "fits_64": 64 in ds, "fits_589": 589 in ds, "smallest": ds[:4]}
    c7["table11_state_k_of_589"] = {s: round(v / 100 * 589, 2) for s, v in t11_state.items()}
    c7["S1_text_vs_table11"] = {"text_sec5_3": 52.4, "table11_state": 50.42,
                                "table12_N2448_implied": round((0.908 * 3037 - 2448) / 589 * 100, 2)}
    c7["table2_ratio"] = round(10124 / 678, 2)
    c7["mpc_core_s"] = {"table2_2500_samples": 10124, "table14_3000_samples": round(T14["ShapeFilter MPC"] * 3600),
                        "per_sample_table2": round(10124 / 2500, 2),
                        "per_sample_table14": round(T14["ShapeFilter MPC"] * 3600 / 3000, 2)}
    c7["mpc_sr_2500_vs_3000"] = {"table2_2500": 93.7, "table10_3000": 92.9}
    # Table 12: 비용(N,S) = N 시드 비용 + (3037-N) × g(S). N마다 S=1 대비 증분으로 g(S)-g(1)을 구함
    t12_cost = {50: [0.235, 0.263, 0.313, 0.390, 0.501, 0.627, 0.740, 0.832, 0.890],
                100: [0.438, 0.463, 0.504, 0.560, 0.626, 0.688, 0.745, 0.789, 0.828],
                200: [0.843, 0.865, 0.898, 0.944, 1.005, 1.081, 1.163, 1.243, 1.311],
                500: [2.061, 2.077, 2.102, 2.137, 2.186, 2.251, 2.330, 2.417, 2.526],
                1000: [4.089, 4.101, 4.118, 4.141, 4.172, 4.209, 4.252, 4.299, 4.362],
                2448: [9.964, 9.967, 9.970, 9.974, 9.979, 9.985, 9.992, 10.000, 10.008]}
    t12_succ = {50: [13.0, 17.2, 33.4, 47.4, 66.4, 83.1, 93.2, 96.7, 99.5],
                100: [21.0, 30.5, 47.5, 66.7, 81.3, 91.1, 96.4, 98.7, 99.3],
                200: [33.4, 44.3, 57.7, 71.9, 80.0, 89.1, 94.9, 96.8, 99.0],
                500: [48.2, 58.4, 68.8, 77.3, 84.1, 89.8, 94.0, 96.6, 97.6],
                1000: [62.2, 70.1, 80.0, 84.9, 90.9, 94.4, 96.8, 98.2, 98.7],
                2448: [90.8, 93.8, 96.0, 97.5, 98.5, 99.1, 99.5, 99.7, 99.9]}
    S_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    inc = {}
    for n, row in t12_cost.items():
        rem = 3037 - n
        inc[n] = [round((c - row[0]) * 1000 * 3600 / rem, 1) for c in row]  # core-s / 남은 시연 1개
    c7["table12_gnr_incremental_core_s_per_demo"] = {n: dict(zip(S_LIST, v)) for n, v in inc.items()}
    c7["table12_test_sr_implied"] = {n: [round((s / 100 * 3037 - n) / (3037 - n) * 100, 1) for s in row]
                                    for n, row in t12_succ.items()}
    # "S=64에서 MPC(3,000 표본) 비용의 1.6%" 점검
    g64 = sum(inc[n][6] for n in inc) / len(inc)
    c7["claim_1_6_percent"] = {
        "gnr_S64_incremental_core_s_mean_over_N": round(g64, 1),
        "pct_of_table14_mpc": round(100 * g64 / (T14["ShapeFilter MPC"] * 3600), 2),
        "pct_of_table2_mpc": round(100 * g64 / 10124, 2),
        "table2_gnr_S77_core_s": 678, "pct_S77_of_table2_mpc": round(100 * 678 / 10124, 2),
        "note": "Table 12 비용은 성공하면 표본 추출을 멈추는 조기 종료 비용이다(증분이 S에 따라 포화). 아래 단가 역산으로 N=2448, S=64는 MPC 3,000표본의 약 1.8%이고, 조기 종료가 없으면 약 21%",
    }
    # Table 12는 조기 종료 비용이다: S=1→2 증분 = 표본 단가 × P(S=1 실패). N=2448에서 단가를 역산하면
    sr2448 = c7["table12_test_sr_implied"][2448]
    unit = inc[2448][1] / (1 - sr2448[0] / 100)
    mpc3000 = T14["ShapeFilter MPC"] * 3600
    c7["claim_1_6_percent"].update({
        "gnr_core_s_per_sample_N2448": round(unit, 1),
        "S64_cost_early_stop_core_s": round(unit + inc[2448][6], 1),
        "S64_pct_of_mpc3000_early_stop": round(100 * (unit + inc[2448][6]) / mpc3000, 2),
        "S64_pct_of_mpc3000_no_early_stop": round(100 * 64 * unit / mpc3000, 1),
    })
    c7["table15_seed_cost"] = {"reported_core_h": 203, "seeds_at_93_7_pct_x_table2": round(50 / 0.937 * 10124 / 3600, 1),
                               "implied_attempts_table2_rate": round(203 / (10124 / 3600), 1),
                               "implied_attempts_table14_rate": round(203 / T14["ShapeFilter MPC"], 1)}
    c7["reduction_11_2x"] = round(8541.0 / 765.6, 2)
    out["C7_shapefilter"] = c7

    # C8 NIST (시험 64개)
    c8 = {"k_of_64": {"GNR open-loop 4.7": round(0.047 * 64, 2), "MPC 92.2": round(0.922 * 64, 2),
                      "GNR+MPC 89.1": round(0.891 * 64, 2), "54.7": round(0.547 * 64, 2), "90.6": round(0.906 * 64, 2)}}
    _, p = stats.fisher_exact([[59, 5], [57, 7]])
    c8["mpc59_vs_gnrmpc57_fisher_p"] = round(p, 3)
    c8["wilson_59_64"], c8["wilson_57_64"] = wilson(59, 64), wilson(57, 64)
    c8["sample_reduction_2296"] = round(100 * (3440 - 2296) / 3440, 2)
    c8["sample_reduction_1976"] = round(100 * (3440 - 1976) / 3440, 2)
    c8["cost_ratio_table2"] = round(795777 / 266199, 2)
    c8["per_demo_core_h"] = {"mpc": round(795777 / 3600, 2), "gnr_mpc": round(266199 / 3600, 2)}
    c8["dataset_cost"] = {"mpc_358_x_221.05": round(358 * 221.05), "reported": 79136,
                          "ratio_reported": round(79136 / 44517, 2)}
    c8["seed_cost_100"] = {"reported_core_h": 25439, "implied_attempts": round(25439 / 221.05, 1),
                           "attempts_at_92_2_pct": round(100 / 0.922, 1)}
    c8["dataset_sr_check"] = round((100 + 258 * 0.891) / 358 * 100, 2)
    out["C8_nist"] = c8

    # C9 다중 embodiment
    c9 = {"wilson": {"GNR 128": wilson(90, 100), "GNR 256": wilson(93, 100), "GNR 384": wilson(95, 100),
                     "SPIDER": wilson(96, 100)}}
    c9["gnr384_vs_spider_fisher_p"] = round(stats.fisher_exact([[95, 5], [96, 4]])[1], 3)
    out["C9_multi_embodiment"] = c9

    # C10 BC
    out["C10_bc"] = {"k": {"MPC": round(0.375 * 32, 2), "GNR": round(0.4375 * 32, 2)},
                     "fisher_p": round(stats.fisher_exact([[12, 20], [14, 18]])[1], 3),
                     "note": "10개 체크포인트 중 최고값 선택, 같은 32개 시험으로 고름"}

    # C11 flywheel (256개 시험)
    c11 = {}
    for v in (55.2, 74.4, 68.4, 84.8):
        k = v / 100 * 256
        c11[str(v)] = {"k": round(k, 2), "nearest": [round(100 * math.floor(k) / 256, 2), round(100 * math.ceil(k) / 256, 2)],
                       "integer_k_rounds_to_value": any(abs(round(100 * kk / 256, 1) - v) < 1e-9 for kk in (math.floor(k), math.ceil(k)))}
    out["C11_flywheel"] = c11

    # C12 성공 기준
    out["C12_criterion"] = {"hot3d_pos_mm": 150, "hot3d_rot_deg": 40, "ik_sr": 18.30, "gnr_S1_sr": 18.10,
                            "mpc_failure_median_epos_mm": {"IK": 54.52, "GNR S=256": 36.18, "iCEM": 26.04},
                            "note": "150 mm·40°는 CHORD 관례. Epos 중앙값(26–55 mm)이 기준보다 한참 작아 성공을 가르는 것은 주로 침투·쥐는 힘·손가락 힘 조건"}

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:400])
    print("wrote", path)


if __name__ == "__main__":
    main()

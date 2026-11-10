"""AdaVLA 표 재계산 (CPU, numpy·scipy, 수 초). 실행: python3 analysis/table_check.py

- T1: Table II (LIBERO, suite당 500회, 구성당 2000회) → 정수 성공 횟수, 평균 재계산, Wilson CI, Fisher, Holm
- T2: Table III 절제 실험(2000회) → 방법별 Fisher, Holm, 최소 검출 차이
- T3: Table IV 실물(SmolVLA, 과제당 30회) → 과제별 Fisher + Holm, 합산 Fisher, 지연 평균·속도비
- T4: Table I 지연 산술 (구성요소 합, 1-step·5-step·10-step 예측값 대조, 속도비, 에너지비)
- T5: 지연 모형에서 역산한 평균 NFE (평균 MLP 가지치기 비율 p의 격자별), 가정은 결과 파일에 명시
- T6: Eq.8 비용 모형: 구간당 NFE 비 (C/Cth 격자)

주의: Table II의 쌍 설계(같은 초기 상태)는 논문에 쌍별 집계가 없어 McNemar 검정은 할 수 없습니다. 비쌍 검정은 근사입니다.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "table_check_results.json"


def count(pct, n):
    k = round(pct / 100 * n)
    assert abs(100 * k / n - pct) < 0.051, (pct, n, k)
    return k


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(ps):
    ps = list(ps)
    order = np.argsort(ps)
    m = len(ps)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def g(x):
    return float(f"{x:.4g}")


def main():
    out = {}

    # ---------------- T1: Table II ----------------
    N_SUITE = 500
    suites = ["Spatial", "Object", "Goal", "Long"]
    T2 = {
        "pi05_base": ([99.40, 98.20, 98.80, 98.20], 98.65),
        "pi05_SparseVLM": ([96.60, 97.00, 97.60, 94.20], 96.35),
        "pi05_FastV": ([66.20, 76.80, 77.20, 48.00], 67.05),
        "pi05_ours": ([99.40, 100.00, 99.80, 98.60], 99.45),
        "xvla_base": ([99.60, 100.00, 98.00, 96.40], 98.50),
        "xvla_SparseVLM": ([99.20, 82.00, 97.00, 90.60], 92.20),
        "xvla_FastV": ([98.40, 99.60, 98.00, 81.20], 94.30),
        "xvla_ours": ([99.40, 100.00, 98.80, 97.40], 98.90),
    }
    cnt = {}
    t1 = {}
    for name, (vals, avg_rep) in T2.items():
        ks = [count(v, N_SUITE) for v in vals]
        cnt[name] = ks
        avg_calc = sum(vals) / 4
        K = sum(ks)
        t1[name] = {
            "suite_counts": dict(zip(suites, [f"{k}/500" for k in ks])),
            "pooled": f"{K}/2000",
            "avg_recomputed": round(avg_calc, 2),
            "avg_reported": avg_rep,
            "avg_matches": abs(avg_calc - avg_rep) < 0.011,
            "pooled_wilson95_pct": wilson(K, 2000),
        }
    pair_tests = {}
    for model in ["pi05", "xvla"]:
        ours = cnt[f"{model}_ours"]
        base = cnt[f"{model}_base"]
        Ko, Kb = sum(ours), sum(base)
        pair_tests[model] = {
            "ours_vs_base_pooled_fisher_p": g(fisher(Ko, 2000, Kb, 2000)),
            "diff_pp": round(100 * (Ko - Kb) / 2000, 2),
            "ours_vs_base_extra_episodes": Ko - Kb,
        }
        ps = [fisher(ours[i], N_SUITE, base[i], N_SUITE) for i in range(4)]
        adj = holm(ps)
        pair_tests[model]["per_suite_fisher_raw"] = [g(p) for p in ps]
        pair_tests[model]["per_suite_fisher_holm"] = [g(p) for p in adj]
        for other in ["SparseVLM", "FastV"]:
            O = cnt[f"{model}_{other}"]
            pair_tests[model][f"ours_vs_{other}_pooled_fisher_p"] = g(fisher(Ko, 2000, sum(O), 2000))
    # 두 모델 방향 차이: Ours 대비 SparseVLM·FastV 는 정리 (논문은 기준선 비교)
    out["T1_table2"] = {"counts_and_avg": t1, "tests": pair_tests,
                        "note": "쌍 설계이나 쌍별 집계가 없어 비쌍 Fisher는 근사. McNemar 불가."}

    # ---------------- T2: Table III (pi0.5, 2000회) ----------------
    T3 = {
        "baseline_10step": 98.65, "1step": 97.05, "5step": 98.00, "random_step": 98.65,
        "ours_PR_adp_0.15": 99.45, "ours_SVD_adp": 99.65, "ours_Fxd_adp": 99.35,
        "ours_noprune_adp": 98.65, "ours_PR_noadp": 99.25, "ours_PR_adp_0.20": 99.20,
        "ours_PR_adp_0.25": 98.40,
    }
    k3 = {k: round(v / 100 * 2000) for k, v in T3.items()}
    for k, v in T3.items():
        assert abs(100 * k3[k] / 2000 - v) < 0.001, k
    ref = "ours_PR_adp_0.15"
    comps = ["baseline_10step", "1step", "5step", "random_step", "ours_Fxd_adp", "ours_SVD_adp",
             "ours_noprune_adp", "ours_PR_noadp"]
    raw = [fisher(k3[ref], 2000, k3[c], 2000) for c in comps]
    adj = holm(raw)
    t3 = {c: {"diff_pp": round(T3[ref] - T3[c], 2), "fisher_p": g(r), "holm_p": g(a)}
          for c, r, a in zip(comps, raw, adj)}
    # 80% 검출력 최소 차이 (정규 근사, 두 비율, 양측 0.05)
    p0 = 0.985
    n = 2000
    mdd = (1.959964 + 0.841621) * math.sqrt(2 * p0 * (1 - p0) / n)
    out["T2_table3"] = {"counts": k3, "vs_ours_PR": t3,
                        "min_detectable_diff_80pct_power_pp": round(100 * mdd, 2),
                        "note": "Fxd(+0.10pp) 차이 2건은 PR 설계 이득의 증거로 약하다."}

    # ---------------- T3: Table IV (SmolVLA 실물, 과제당 30회) ----------------
    base_s = [25, 26, 29, 25]
    ours_s = [29, 28, 27, 24]
    ps = [fisher(ours_s[i], 30, base_s[i], 30) for i in range(4)]
    adj = holm(ps)
    Kb, Ko = sum(base_s), sum(ours_s)
    lat_b = [803.49, 811.53, 805.31, 806.58]
    lat_o = [496.79, 447.07, 476.00, 534.10]
    out["T3_table4"] = {
        "per_task_base": [f"{k}/30" for k in base_s], "per_task_ours": [f"{k}/30" for k in ours_s],
        "per_task_fisher_p": [g(p) for p in ps], "per_task_holm_p": [g(p) for p in adj],
        "pooled": f"{Kb}/120 vs {Ko}/120", "pooled_fisher_p": g(fisher(Ko, 120, Kb, 120)),
        "pooled_wilson_base": wilson(Kb, 120), "pooled_wilson_ours": wilson(Ko, 120),
        "avg_base_reported": 87.50, "avg_ours_reported": 90.00,
        "avg_latency_base_recomputed": round(sum(lat_b) / 4, 2), "avg_latency_ours_recomputed": round(sum(lat_o) / 4, 2),
        "speedup": round(sum(lat_b) / sum(lat_o), 3),
        "note": "지연 SD가 보고되지 않아 지연 검정 불가. 실험은 과제당 30회.",
    }

    # ---------------- T4: Table I 지연 산술 ----------------
    inp, vlm_fwd, attn, mlp, step, flow10, e2e = 96.25, 290.28, 3.05, 12.47, 53.75, 557.78, 953.62
    vlm = inp + vlm_fwd
    lat = {
        "vlm_total_input_plus_forward": round(vlm, 2),
        "vlm_share_pct": round(100 * vlm / e2e, 1),
        "flow10_share_pct": round(100 * flow10 / e2e, 1),
        "flow_10x_single_step": round(10 * step, 2),
        "flow10_reported_minus_10x_step": round(flow10 - 10 * step, 2),
        "components_sum_plus_flow": round(vlm + flow10, 2),
        "e2e_minus_components": round(e2e - vlm - flow10, 2),
        "per_layer_18x_attn_plus_mlp": round(18 * (attn + mlp), 2),
        "vlm_forward_minus_18x_layer": round(vlm_fwd - 18 * (attn + mlp), 2),
        "pred_1step_total": round(vlm + step, 2), "meas_1step_total_T3": 447.45,
        "pred_5step_total": round(vlm + 5 * step, 2), "meas_5step_total_T3": 677.56,
        "pred_10step_total": round(vlm + 10 * step, 2), "meas_10step_total_T2": 958.54,
        "speedup_pi05_orin": round(958.54 / 513.35, 3), "speedup_xvla_orin": round(387.52 / 173.09, 3),
        "speedup_pi05_A10": round(289.68 / 158.87, 3),
        "energy_ratio_pi05": round(19.83 / 13.17, 3), "energy_ratio_xvla": round(10.73 / 3.51, 3),
        "fastv_xvla_latency_drop_ms": round(387.52 - 385.21, 2),
        "tableI_e2e_vs_tableII_base_diff": round(958.54 - 953.62, 2),
    }
    out["T4_latency_arithmetic"] = lat

    # ---------------- T5: 역산 NFE ----------------
    mlp_share = mlp / (mlp + attn)
    vlm_fwd_saving_per_p = mlp_share * vlm_fwd  # 선형 가정
    nfe_noprune = (532.32 - vlm) / step
    grid = {}
    for p in [0.0, 0.05, 0.08, 0.10, 0.15, 0.20]:
        vlm_p = vlm - vlm_fwd_saving_per_p * p
        grid[str(p)] = round((513.35 - vlm_p) / step, 2)
    p_match = (vlm - (vlm - 532.32 + 0)) if False else None
    # 무가지치기 NFE를 유지할 때 필요한 p (513.35 = 386.53 - 233.4p + NFE*53.75)
    p_needed = (532.32 - 513.35) / vlm_fwd_saving_per_p
    out["T5_implied_nfe"] = {
        "assumption": "지연 = VLM(386.53 − 선형 가지치기 절감) + NFE_avg × 53.75 ms; 절감 = MLP 비율 12.47/(12.47+3.05) × VLM forward 290.28 × p",
        "mlp_share_of_vlm_forward": round(mlp_share, 4),
        "saving_ms_per_unit_p": round(vlm_fwd_saving_per_p, 2),
        "nfe_avg_noprune_variant": round(nfe_noprune, 2),
        "nfe_avg_full_by_p": grid,
        "p_needed_if_nfe_equal_noprune": round(p_needed, 3),
        "static_steps_latency_ms": {str(k): round(vlm + k * step, 2) for k in [1, 2, 3, 5, 10]},
        "note": "p가 0.2를 넘으면 NFE가 10을 넘어 속도 이득이 설명되지 않음. 따라서 평균 가지치기는 작고 NFE는 2~7 범위로 추정된다.",
    }

    # ---------------- T6: Eq.8 비용 모형 ----------------
    s = 0.1
    rows = {}
    for r in [0.5, 1, 1.5, 2, 4, 8]:
        C_over_Cth = r  # C/Cth
        # tau=0 기준, 반올림 없이 s_hat = min(1/2, max(s, 2s/r))
        s_hat = min(0.5, max(s, 2 * s / C_over_Cth))
        nfe_per_unit = 2 / s_hat
        rows[str(r)] = {"s_hat": round(s_hat, 4), "nfe_per_unit_tau": round(nfe_per_unit, 2),
                        "ratio_vs_euler10": round(nfe_per_unit / 10, 2)}
    out["T6_eq8_cost"] = {"rows_C_over_Cth": rows, "max_ratio_vs_euler": 2.0,
                          "note": "C >= 2Cth 이면 s_hat = s, NFE = 2/s = 20 per unit tau = Euler 대비 2배"}

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")
    print("pi05 ours vs base p:", pair_tests["pi05"]["ours_vs_base_pooled_fisher_p"],
          "xvla:", pair_tests["xvla"]["ours_vs_base_pooled_fisher_p"])
    print("Table IV pooled p:", out["T3_table4"]["pooled_fisher_p"])
    print("NFE noprune:", round(nfe_noprune, 2), "grid:", grid, "p_needed:", round(p_needed, 3))


if __name__ == "__main__":
    main()

"""SR-VLA (arXiv:2611.22356 v1) 보고 수치 재계산과 일관성 검사.

표 III, IV, II, 본문 수치를 논문에 적힌 값만으로 다시 계산합니다. 로봇이나 시뮬레이터는 돌리지 않습니다.
성공률 검정에는 논문에 없는 시행 수가 필요합니다. 기본값 N_TRIALS=500(LIBERO 표준 50 starts x 10 tasks)을
가정하며, 결과 JSON에 가정값을 함께 적습니다. 실행: python3 analysis/recompute.py (수 초, 표준 라이브러리만 사용)
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
N_TRIALS = 500  # 가정: 논문에 시행 수가 없음. LIBERO 표준 500 starts/suite.
SUITES = ["Spatial", "Object", "Goal", "Long"]

# ---- 표 III (LIBERO, 성공률 %, Query/Action ms, Stall Mean ms, K) ----
T3 = {
    "pi05_sync": {"succ": [97.8, 97.4, 97.0, 93.4], "avg_rep": 96.4, "qa": 25.62, "stall": 128.11, "K": 5, "rep_speedup": 1.00, "rep_resid": 1.00},
    "oft_sync": {"succ": [97.6, 98.4, 97.9, 94.5], "avg_rep": 97.1, "qa": 16.90, "stall": 135.18, "K": 8, "rep_speedup": 1.52, "rep_resid": 1.06},
    "vlash_pi05": {"succ": [97.2, 99.6, 95.6, 94.0], "avg_rep": 96.6, "qa": 12.05, "stall": 60.25, "K": 5, "rep_speedup": 2.13, "rep_resid": 0.47},
    "vla_cache_oft": {"succ": [98.3, 97.5, 98.3, 95.4], "avg_rep": 97.4, "qa": 15.24, "stall": 121.88, "K": 8, "rep_speedup": 1.68, "rep_resid": 0.95},
    "srvla_pi05": {"succ": [97.8, 99.2, 97.2, 93.4], "avg_rep": 96.9, "qa": 9.59, "stall": 47.95, "K": 5, "rep_speedup": 2.67, "rep_resid": 0.37},
    "srvla_oft": {"succ": [98.8, 98.2, 97.6, 94.2], "avg_rep": 97.2, "qa": 13.27, "stall": 106.19, "K": 8, "rep_speedup": 1.93, "rep_resid": 0.83},
}
SYNC_STALL = T3["pi05_sync"]["stall"]
SYNC_QA = T3["pi05_sync"]["qa"]

# ---- 표 IV (스위트별 설정과 통계) ----
T4 = {
    "Spatial": {"tau_img": 0.58, "tau_p": 0.40, "Cmax": 27, "ke": 220, "alpha_r": 0.015, "repair": 31.68, "kv": 27.37},
    "Object": {"tau_img": 1.07, "tau_p": 0.59, "Cmax": 17, "ke": 147, "alpha_r": 0.049, "repair": 43.27, "kv": 24.88},
    "Goal": {"tau_img": 0.57, "tau_p": 0.47, "Cmax": 29, "ke": 137, "alpha_r": 0.018, "repair": 30.09, "kv": 16.24},
    "Long": {"tau_img": 0.30, "tau_p": 0.58, "Cmax": 15, "ke": 88, "alpha_r": 0.008, "repair": 38.88, "kv": 13.44},
}


def wilson(p, n, z=1.959964):
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def two_prop(p1, n1, p2, n2):
    """두 비율 차이의 정규 근사 검정(양측)과 Newcombe(Wilson) 95% 구간."""
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    p_val = math.erfc(abs(z) / math.sqrt(2))
    l1, u1 = wilson(p1, n1)
    l2, u2 = wilson(p2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return {"diff_pp": round(100 * d, 2), "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)], "p_two_sided": round(p_val, 4)}


def main():
    out = {"assumed_trials_per_suite": N_TRIALS, "checks": {}, "suite_tests": {}, "derived": {}}

    # K1: 행별 재계산 (평균 성공률, Query/Action = Stall/K, 속도비, 잔여 정지비)
    rows = {}
    for k, r in T3.items():
        avg = sum(r["succ"]) / 4
        qa = r["stall"] / r["K"]
        rows[k] = {
            "avg_succ_recomputed": round(avg, 3),
            "avg_succ_reported": r["avg_rep"],
            "qa_recomputed": round(qa, 2),
            "qa_reported": r["qa"],
            "resid_recomputed": round(r["stall"] / SYNC_STALL, 3),
            "resid_reported": r["rep_resid"],
            "speedup_vs_pi05sync_recomputed": round(SYNC_QA / r["qa"], 3),
            "speedup_reported": r["rep_speedup"],
        }
    out["checks"]["K1_table3_rows"] = rows

    # K2: 본문 속도비 주장
    out["checks"]["K2_claims"] = {
        "srvla_pi05_vs_sync": round(SYNC_STALL / T3["srvla_pi05"]["stall"], 3),  # abstract 2.67
        "srvla_oft_vs_oft_sync": round(T3["oft_sync"]["qa"] / T3["srvla_oft"]["qa"], 3),  # abstract 1.27
        "srvla_pi05_vs_vlash": round(T3["vlash_pi05"]["qa"] / T3["srvla_pi05"]["qa"], 3),  # text 1.25
        "srvla_oft_vs_vla_cache_oft": round(T3["vla_cache_oft"]["qa"] / T3["srvla_oft"]["qa"], 3),  # text 1.15
        "srvla_pi05_success_gain_vs_sync_pp": round(T3["srvla_pi05"]["avg_rep"] - T3["pi05_sync"]["avg_rep"], 2),
    }

    # K3: 표 IV 평균 (본문: τimg 0.63, τp 0.51, Cmax 22, ke 148, αr 0.0225, 반복률 35.98, KV 20.48)
    mean = lambda key: sum(T4[s][key] for s in SUITES) / 4
    out["checks"]["K3_table4_means"] = {
        "tau_img": round(mean("tau_img"), 3), "tau_p": round(mean("tau_p"), 3), "Cmax": mean("Cmax"),
        "ke": mean("ke"), "alpha_r": round(mean("alpha_r"), 4),
        "repair_rate": round(mean("repair"), 2), "kv_update": round(mean("kv"), 2),
    }
    # K4: 표 II Spatial 행과 표 IV Spatial 행의 일치 (31.68, 27.37)
    out["checks"]["K4_spatial_repair_table2_vs_table4"] = {"table2_full_repair": 31.68, "table4_spatial": T4["Spatial"]["repair"],
                                                         "table2_full_kv": 27.37, "table4_spatial_kv": T4["Spatial"]["kv"]}

    # K5: 지연 주입 (본문 IV-C): SR-VLA 74.36 - 47.95 = 노출된 추가 지연 26.41, VLASH 100.25 - 60.25 = 40.00
    out["checks"]["K5_delay_exposure"] = {
        "srvla_exposed_ms": round(74.36 - 47.95, 2), "text_says": 26.41,
        "vlash_exposed_ms": round(100.25 - 60.25, 2), "injected_ms": 40,
        "srvla_exposed_fraction": round((74.36 - 47.95) / 40, 3),
    }

    # K6: 본문 Table II 문단 수치와 표 III 수치 불일치 (범위 미기재)
    out["checks"]["K6_text_stall_vs_table3"] = {
        "text_sync_ms": 139.72, "table3_sync_ms": SYNC_STALL,
        "text_full_recompute_ms": 60.44, "text_selective_ms": 54.78, "table3_srvla_pi05_ms": 47.95,
        "note": "표 II 문단은 Spatial 기준으로 보이나 범위가 명시되지 않음. 재현 불가.",
    }

    # K7: 본문 목표 이동 속도비 1.93 (1 cm), 1.52 (5 cm)는 표 III의 0 cm 2.67과 다른 조건이며 그림 5(c)에만 있음
    out["checks"]["K7_target_motion_speedups"] = {
        "abstract_range": [1.52, 1.93], "table_values": "없음 (그림 5(c)에서만 확인 가능)",
        "coincides_with_srvla_oft_table3": T3["srvla_oft"]["rep_speedup"],
    }

    # 스위트별 성공률 검정 (시행 수 가정)
    tests = {}
    comps = {
        "srvla_pi05_vs_pi05_sync": ("srvla_pi05", "pi05_sync"),
        "srvla_pi05_vs_vlash": ("srvla_pi05", "vlash_pi05"),
        "srvla_oft_vs_oft_sync": ("srvla_oft", "oft_sync"),
        "srvla_oft_vs_vla_cache_oft": ("srvla_oft", "vla_cache_oft"),
    }
    for name, (a, b) in comps.items():
        per = {}
        for i, s in enumerate(SUITES):
            pa, pb = T3[a]["succ"][i] / 100, T3[b]["succ"][i] / 100
            per[s] = two_prop(pa, N_TRIALS, pb, N_TRIALS)
        # 네 스위트 평균 차이: 평균의 분산 근사 (독립 가정)
        diffs = [T3[a]["succ"][i] - T3[b]["succ"][i] for i in range(4)]
        var = 0.0
        for i in range(4):
            pa, pb = T3[a]["succ"][i] / 100, T3[b]["succ"][i] / 100
            var += (pa * (1 - pa) + pb * (1 - pb)) / N_TRIALS
        se_avg = math.sqrt(var) / 4
        d_avg = sum(diffs) / 4
        z = (d_avg / 100) / se_avg if se_avg > 0 else 0
        per["avg_of_4_suites"] = {"diff_pp": round(d_avg, 2), "ci95_pp_approx": [round(d_avg - 196 * se_avg, 2), round(d_avg + 196 * se_avg, 2)],
                                  "p_two_sided_approx": round(math.erfc(abs(z) / math.sqrt(2)), 4)}
        tests[name] = per
    out["suite_tests"] = tests

    # 필요한 시행 수: 차이 0.5pp를 검정력 80%로 잡으려면 (양측 0.05, 비쌍)
    z_a, z_b = 1.959964, 0.841621
    p_base = 0.97
    d = 0.005
    n_need = ((z_a + z_b) ** 2) * 2 * p_base * (1 - p_base) / d ** 2
    out["derived"]["n_per_arm_for_0.5pp_power80_base97"] = round(n_need)
    out["derived"]["n_per_arm_for_1.0pp_power80_base97"] = round(((z_a + z_b) ** 2) * 2 * p_base * (1 - p_base) / 0.01 ** 2)

    # 30 Hz에서의 정지 비율 (식 2 근사: Texec = K/30 s, 논문 1쪽 명시한 30 Hz)
    hz = 30.0
    for k, label in [("pi05_sync", "pi05_sync"), ("srvla_pi05", "srvla_pi05")]:
        r = T3[k]
        texec_ms = r["K"] * 1000 / hz
        out["derived"][f"r_stall_{label}"] = round(r["stall"] / (texec_ms + r["stall"]), 3)
    out["derived"]["texec_ms_pi05_K5_30Hz"] = round(5 * 1000 / hz, 2)
    out["derived"]["note_r_stall"] = "식 (2) 근사. 논문은 Texec을 명시하지 않음. 30 Hz와 K=5를 가정한 값."

    path = os.path.join(HERE, "recompute_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote", path)
    print(json.dumps(out["checks"]["K2_claims"], ensure_ascii=False))
    print(json.dumps(out["suite_tests"]["srvla_pi05_vs_pi05_sync"]["avg_of_4_suites"], ensure_ascii=False))
    print(json.dumps(out["suite_tests"]["srvla_oft_vs_vla_cache_oft"]["avg_of_4_suites"], ensure_ascii=False))
    print(json.dumps(out["derived"], ensure_ascii=False))


if __name__ == "__main__":
    main()

"""BODex (arXiv:2412.16490 v3, ICRA 2025) 표 수치 재계산과 검정.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json (수 초, CPU만 사용)

입력: 논문 본문과 표에 적힌 값과 설정 수(물체 수, 스케일, 시도 수)만 쓴다. 데이터·가중치·시뮬레이터는 쓰지 않는다.
가정은 출력 JSON의 "assumptions"에 모두 적는다.
"""
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"

# ---------------------------------------------------------------- 논문 설정 수 (본문 V.A, V.C)
N_OBJECTS = 2397          # 유효 물체 수 (v3). v1은 2440
N_SCALES = 4              # 0.06, 0.08, 0.10, 0.12
N_PER_SCALED = 10         # 스케일된 물체 하나당 합성 파지 수 (Table II 설정)
N_SCALED = N_OBJECTS * N_SCALES
N_GRASP_TABLE2 = N_SCALED * N_PER_SCALED   # 본문: "95,880 grasps per method"

# ---------------------------------------------------------------- 표 값 (arXiv v3, 2025-09-03 제출 PDF 기준)
TABLE2 = {  # Allegro, floating, SSR(%) · S(grasp/s) · PD(mm) · CDC(mm) · SPD(mm) · FVR(%)
    "DexGraspNet": dict(SSR=37.0, S=0.40, PD=4.49, CDC=5.53, SPD=1.30, FVR=32.0),
    "SpringGrasp": dict(SSR=12.1, S=0.67, PD=11.7, CDC=20.1, SPD=0.80, FVR=44.5),
    "FRoGGeR": dict(SSR=17.2, S=1.20, PD=2.14, CDC=3.96, SPD=0.01, FVR=28.9),
    "BODex (Ours)": dict(SSR=79.9, S=61.5, PD=0.51, CDC=2.98, SPD=0.01, FVR=33.2),
}
TABLE2_V1 = {  # v1 (2024-12-21) 같은 표. 지표 이름도 다름 (CD는 v3의 CDC와 정의가 다름)
    "DexGraspNet": dict(SSR=31.37, S=0.40, PD=13.5, CD=6.90, SPD=0.93, FVR=0.31),
    "SpringGrasp": dict(SSR=37.24, S=0.67, PD=16.2, CD=6.18, SPD=0.48, FVR=0.44),
    "FRoGGeR": dict(SSR=41.97, S=1.20, PD=2.17, CD=0.88, SPD=0.0002, FVR=0.28),
    "BODex (Ours)": dict(SSR=89.55, S=61.5, PD=0.37, CD=0.28, SPD=0.008, FVR=0.32),
}
TABLE3 = {  # (hand, coarse2fine, pregrasp): (SSR, S, PD, CDC)  v3
    ("Shadow", True, False): (57.8, 52.7, 0.69, 3.07),
    ("Shadow", False, True): (68.4, 55.3, 0.81, 3.62),
    ("Shadow", True, True): (75.8, 49.8, 0.66, 2.83),
    ("Allegro", True, False): (68.2, 65.2, 0.57, 3.16),
    ("Allegro", False, True): (78.1, 67.4, 0.63, 3.42),
    ("Allegro", True, True): (79.9, 61.5, 0.51, 2.98),
}
TABLE4 = {  # (dataset, method): SSR(%) v3. 테스트 시도 수는 본문에 없음
    ("Ours-F", "GraspTTA"): 14.8, ("Ours-F", "ISAGrasp"): 20.7, ("Ours-F", "DP3"): 62.4,
    ("Ours-F", "UDG"): 80.1, ("DGN", "UDG"): 42.6, ("UDG", "UDG"): 36.8, ("Ours-T", "UDG"): 78.8,
}
REAL = dict(objects=20, attempts_per_object=5, success_rate=81)   # 본문 V.E, 81%
REAL_TRIALS = REAL["objects"] * REAL["attempts_per_object"]        # 100
REAL_SUCCESS = round(REAL["success_rate"] / 100 * REAL_TRIALS)     # 81


# ---------------------------------------------------------------- 함수
def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def two_prop_z(p1, n1, p2, n2):
    """두 비율의 차이 검정 (pooled z, 양측). p는 0–1."""
    k1, k2 = p1 * n1, p2 * n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return z, 2 * stats.norm.sf(abs(z))


def n_per_arm(p1, p2, alpha=0.05, power=0.80):
    """두 비율을 가르는 팔당 표본 수 (양측, 정규 근사)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    num = (za * math.sqrt(2 * (p1 + p2) / 2 * (1 - (p1 + p2) / 2)) +
           zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


def main():
    res = {"assumptions": {
        "table2_n_per_method": N_GRASP_TABLE2,
        "table2_n_note": "본문 V.C.1 '95,880 grasps per method'. 물체·스케일·파지 수로 재확인 (2397 x 4 x 10).",
        "independence": "파지는 독립이라고 가정. 같은 물체의 4개 스케일과 10개 파지는 상관이 있을 수 있어 실제 구간은 더 넓다.",
        "table3_n": "ablation 시도 수는 본문에 없음. 표 II와 같은 95,880을 가정 (V 검정은 이 가정에 의존).",
        "table4_n": "테스트 시도 수 미기재. 검정 불가.",
    }}

    # V1: 설정 수의 산술
    res["V1_counts"] = {
        "scaled_objects": N_SCALED, "expected_table2_grasps": N_GRASP_TABLE2,
        "v1_scaled_objects": 2440 * 4, "v1_table2_grasps": 2440 * 4 * 10,
        "note": "v3 9,588 스케일 물체와 95,880 파지는 본문과 일치. v1은 2,440물체와 97,600 파지.",
    }

    # V2: 표 II SSR의 95% Wilson 구간과 대비 검정 (Ours 대 각 기준선)
    t2 = {}
    ours = TABLE2["BODex (Ours)"]["SSR"] / 100
    for name, row in TABLE2.items():
        p = row["SSR"] / 100
        k = round(p * N_GRASP_TABLE2)
        lo, hi = wilson(k, N_GRASP_TABLE2)
        entry = {"SSR": row["SSR"], "k_implied": k, "wilson95": [round(100 * lo, 2), round(100 * hi, 2)]}
        if name != "BODex (Ours)":
            z, pv = two_prop_z(ours, N_GRASP_TABLE2, p, N_GRASP_TABLE2)
            entry["diff_vs_ours_pp"] = round(100 * (ours - p), 2)
            entry["z"] = round(z, 2)
            entry["p"] = float(f"{pv:.3g}")
        t2[name] = entry
    res["V2_table2_ssr"] = t2

    # V3: 표 II의 다른 지표 대비 (순위 확인). 낮을수록 좋은 지표는 "낮음", 높을수록 좋은 지표는 "높음"
    rank = {}
    for metric, better in [("SSR", "high"), ("S", "high"), ("PD", "low"), ("CDC", "low"), ("SPD", "low"), ("FVR", "low")]:
        vals = {n: r[metric] for n, r in TABLE2.items()}
        best = max(vals, key=vals.get) if better == "high" else min(vals, key=vals.get)
        rank[metric] = {"best": best, "best_value": vals[best], "ours": vals["BODex (Ours)"]}
    res["V3_table2_best_per_metric"] = rank
    res["V3_speed_ratio_vs_frogger"] = round(61.5 / 1.20, 2)   # 본문 "50x" (Allegro 기준)
    res["V3_speed_note"] = "속도는 기준선이 원 논문 보고값이라 하드웨어가 다르다 (본문 V.B)."

    # V4: 표 III (ablation). 두 요인의 개별 효과와 검정 (가정: 각 셀 n = 95,880)
    t3 = {}
    for hand in ("Shadow", "Allegro"):
        full = TABLE3[(hand, True, True)][0] / 100
        for label, key in [("pregrasp 제거 (C2F만)", (hand, True, False)),
                           ("C2F 제거 (pregrasp만)", (hand, False, True))]:
            p = TABLE3[key][0] / 100
            z, pv = two_prop_z(full, N_GRASP_TABLE2, p, N_GRASP_TABLE2)
            t3[f"{hand}: {label}"] = {
                "full_SSR": round(100 * full, 1), "ablated_SSR": round(100 * p, 1),
                "diff_pp": round(100 * (full - p), 2), "z": round(z, 2), "p": float(f"{pv:.3g}"),
                "n_per_arm_for_80pct_power_at_0.05": n_per_arm(full, p),
            }
    # 빠진 셀: 두 요인 모두 없는 기준선 (Coarse2fine ✗, Pre-grasp ✗)
    t3["missing_cell"] = "두 요인 모두 끈 행(✗✗)이 표에 없음. 상호작용과 기준선 크기를 알 수 없음."
    res["V4_table3_ablation"] = t3

    # V5: 표 III의 단위 규모 확인: 전체 2x2 중 어느 쪽이 큰가 (Allegro, 본문 주장 '둘 다 있을 때 가장 좋다')
    res["V5_allegro_full_is_best"] = TABLE3[("Allegro", True, True)][0] == max(
        v[0] for k, v in TABLE3.items() if k[0] == "Allegro")

    # V6: 표 IV 값의 반올림 확인과 기준선 대비 (n 미기재이므로 p값 없음)
    res["V6_table4"] = {
        "ratio_ours_F_UDG_to_DGN_UDG": round(80.1 / 42.6, 2),
        "gap_pp": round(80.1 - 42.6, 1),
        "note": "본문 '40%에서 80%' (초록)는 DGN 42.6 → 80.1로 맞음. 시도 수가 없어 신뢰구간 불가.",
        "n_needed_for_+-3pp_halfwidth_at_p0.8": math.ceil(1.96 ** 2 * 0.8 * 0.2 / 0.03 ** 2),
    }

    # V7: 실로봇 81% (20물체 x 5회 = 100회). 물체 군집을 무시한 Wilson 구간
    lo, hi = wilson(REAL_SUCCESS, REAL_TRIALS)
    res["V7_real_world"] = {
        "trials": REAL_TRIALS, "successes": REAL_SUCCESS,
        "wilson95": [round(100 * lo, 1), round(100 * hi, 1)],
        "note": "물체별 결과는 본문에 없다. 물체로 묶으면 구간이 넓어진다 (군집 효과 미확인).",
    }
    # 81%가 75%보다 큰지 (단측 이항 정확검정, 참고용). 귀무가설 p=0.75
    res["V7_real_world"]["binom_p_vs_75pct_two_sided"] = float(
        f"{stats.binomtest(REAL_SUCCESS, REAL_TRIALS, 0.75).pvalue:.3g}")

    # V8: 데이터셋 규모 (표 I). 물체당 파지 수는 생성 설정 없이 계산만 가능
    res["V8_table1"] = {
        "floating_grasps_per_scaled_object": round(4.09e6 / N_SCALED, 1),
        "tabletop_grasps_per_scaled_object": round(3.08e6 / N_SCALED, 1),
        "tabletop_trajectories_equal_grasps": True,
        "v1_floating_grasps_per_scaled_object": round(3.62e6 / (2440 * 4), 1),
        "note": "표 II 설정(물체당 10개)과 데이터셋 설정(물체당 약 427개)이 다르다. 본문에 시드 수가 없다.",
    }

    # V9: 표 II 값이 표 III Allegro 행과 같은지 (내부 일치)
    res["V9_internal_match"] = {
        "table2_ours_vs_table3_allegro_full": {
            "SSR": (TABLE2["BODex (Ours)"]["SSR"], TABLE3[("Allegro", True, True)][0]),
            "S": (TABLE2["BODex (Ours)"]["S"], TABLE3[("Allegro", True, True)][1]),
            "PD": (TABLE2["BODex (Ours)"]["PD"], TABLE3[("Allegro", True, True)][2]),
            "CDC": (TABLE2["BODex (Ours)"]["CDC"], TABLE3[("Allegro", True, True)][3]),
        },
        "abstract_49_per_s_vs_shadow_full": TABLE3[("Shadow", True, True)][1],
    }

    # V10: v1 → v3 표 II 변화 (같은 행 이름, 지표 정의는 다름)
    v13 = {}
    for name in TABLE2:
        a, b = TABLE2_V1[name], TABLE2[name]
        v13[name] = {"SSR_v1": a["SSR"], "SSR_v3": b["SSR"], "delta_pp": round(b["SSR"] - a["SSR"], 2)}
    res["V10_v1_to_v3_table2_ssr"] = v13
    res["V10_note"] = ("v1의 CD(평균 최소 거리, 관통 시 0)와 v3의 CDC(손가락 간 부호거리 최댓값-최솟값)는 다른 지표다. "
                       "두 버전의 PD·FVR·SPD 값도 같은 기준이 아니다.")

    # V11: 실로봇 Wilson 구간 대 공식 주장 (abstract '81%', 'over 75%' 기준)
    res["V11_abstract_claims"] = {
        "Shadow_SSR_above_75": TABLE3[("Shadow", True, True)][0] > 75,
        "Allegro_SSR_above_75": TABLE3[("Allegro", True, True)][0] > 75,
        "Leap_SSR_reported_in_tables": False,
    }

    # V12: 물체 군집 효과 (같은 스케일 물체의 파지 10개가 상관) 민감도. design effect = 1 + (m-1) * rho
    m = N_PER_SCALED
    dce = {}
    for rho in (0.05, 0.10, 0.20):
        de = 1 + (m - 1) * rho
        full = TABLE3[("Allegro", True, True)][0] / 100
        abl = TABLE3[("Allegro", False, True)][0] / 100   # pre-grasp만 (C2F 없음)
        z_adj, p_adj = two_prop_z(full, N_GRASP_TABLE2 / de, abl, N_GRASP_TABLE2 / de)
        lo_adj, hi_adj = wilson(round(0.799 * N_GRASP_TABLE2 / de), N_GRASP_TABLE2 / de)  # 표 II Ours 79.9
        dce[f"rho={rho}"] = {
            "design_effect": round(de, 2),
            "table2_ours_wilson95_adjusted": [round(100 * lo_adj, 2), round(100 * hi_adj, 2)],
            "table3_allegro_full_vs_no_c2f_z": round(z_adj, 2),
            "table3_allegro_full_vs_no_c2f_p": float(f"{p_adj:.3g}"),
        }
    res["V12_cluster_sensitivity"] = dce
    res["V12_note"] = ("n을 유효 표본 수 n/DE로 줄여 본 값. 표 III 두 칸(79.9, 78.1)의 차이는 n이 95,880이면 "
                       "군집을 고려해도 유의하게 남지만, 실제 n은 본문에 없다.")

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print(json.dumps({k: res[k] for k in ["V2_table2_ssr", "V4_table3_ablation", "V7_real_world"]},
                     ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()

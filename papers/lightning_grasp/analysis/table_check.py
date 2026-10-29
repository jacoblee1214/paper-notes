"""Table 1·Fig. 1·Fig. 10·본문 수치 재계산 (CPU). 실행: python3 analysis/table_check.py

- C1: Table 1 trimmed µ(최솟값·최댓값 제외 평균) 재계산
- C2: "1,000–10,000 grasps를 2–5초에"와 Table 1 SPS·"6초 안" 정합성
- C3: 속도 배수 주장(Fig. 1, Sec. 1, Sec. 6.3, Fig. 10 캡션) 재계산
- C4: Fig. 10 단계별 비율 합과 "접촉·운동학 최적화가 각각 약 33%"
- C5: 메모리 추정(Sec. 4.4)과 S² 256개 원뿔 반각(Sec. 4.3)
- C6: 공개 코드 기본값으로 본 한 번 forward의 후보 수와 수율
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OBJECTS = ["Capsule", "Apple", "Spoon", "Cup", "Scissors", "Screwdriver", "Plier", "Hammer"]
# Table 1 (A100, amortized effective SPS), 마지막 값은 논문이 적은 trimmed µ
TABLE1 = {
    "Allegro": ([1296.1, 1578.8, 955.6, 1090.0, 989.2, 1020.6, 1545.0, 944.2], 1090.8),
    "LEAP": ([3306.0, 729.0, 408.3, 281.6, 138.6, 356.6, 403.0, 343.0], 420.2),
    "Shadow": ([1060.2, 288.4, 329.4, 181.5, 416.2, 895.0, 745.1, 678.6], 558.8),
    "DClaw": ([2823.5, 221.3, 158.9, 138.1, 126.1, 154.5, 619.3, 203.2], 249.1),
}
# Fig. 1 (A100): 방법별 (SPS 범위, forward 시간 범위 초)
FIG1 = {
    "DexGraspNet": ((0, 3), (1800, 2000)),
    "SpringGrasp": ((0, 3), (10, 40)),
    "BODex": ((30, 50), (100, 120)),
    "LightningGrasp": ((300, 1000), (2, 5)),
}
# Fig. 10 (Allegro + YCB Apple): 단계별 비율(%)과 SPS
STAGES = ["Object Placement", "Contact Domain", "Contact Opt", "Kinematics (I)", "Kinematics (II)", "Postprocess"]
FIG10 = {
    "TITAN X": ([2.6, 22.0, 33.8, 22.5, 11.1, 8.0], 626.1),
    "V100": ([4.5, 14.3, 34.9, 19.7, 17.2, 9.1], 1060.6),
    "A100": ([2.8, 14.6, 27.2, 23.6, 14.4, 17.2], 1578.0),
    "RTX 4090": ([2.0, 10.8, 43.4, 16.7, 16.9, 10.1], 2173.1),
}


def trimmed(v):
    s = sorted(v)
    return sum(s[1:-1]) / (len(s) - 2)


def main() -> None:
    out: dict = {}

    # C1 trimmed µ
    c1 = {}
    for hand, (vals, reported) in TABLE1.items():
        mine = trimmed(vals)
        row = {"recomputed": round(mine, 2), "reported": reported, "diff": round(reported - mine, 2)}
        if abs(reported - mine) > 0.15:
            # 어떤 두 칸을 빼면 보고값이 나오는지
            pairs = [(OBJECTS[i], OBJECTS[j]) for i, j in itertools.combinations(range(8), 2)
                     if abs((sum(vals) - vals[i] - vals[j]) / 6 - reported) < 0.06]
            row["pairs_that_reproduce_reported"] = pairs
            row["min_max_objects"] = [OBJECTS[vals.index(min(vals))], OBJECTS[vals.index(max(vals))]]
        c1[hand] = row
    out["C1_table1_trimmed_mean"] = c1

    # C2 grasps per pass = SPS × time
    cells = [(h, OBJECTS[i], v) for h, (vals, _) in TABLE1.items() for i, v in enumerate(vals)]
    below = [(h, o, v, round(v * 6)) for h, o, v in cells if v * 6 < 1000]
    out["C2_grasps_per_pass"] = {
        "claim": "1,000–10,000 diverse valid grasps within 2–5 s (Sec. 1); all configurations complete within 6 s (Table 1)",
        "implied_sps_range_from_claim": [round(1000 / 5), round(10000 / 2)],
        "table1_sps_range": [min(v for *_, v in cells), max(v for *_, v in cells)],
        "cells_below_1000_grasps_even_at_6s": below,
        "n_cells": len(cells),
        "fig1_sps_range_vs_trimmed_mu": {"fig1": FIG1["LightningGrasp"][0],
                                          "trimmed_mu": {h: c1[h]["recomputed"] for h in c1}},
    }

    # C3 속도 배수
    ours_sps, ours_t = FIG1["LightningGrasp"]
    c3 = {}
    for m, (sps, t) in FIG1.items():
        if m == "LightningGrasp":
            continue
        c3[m] = {
            "sps_ratio": ("> %d" % round(ours_sps[0] / max(sps[1], 1e-9))) if sps[0] == 0
            else [round(ours_sps[0] / sps[1], 1), round(ours_sps[1] / sps[0], 1)],
            "forward_time_ratio": [round(t[0] / ours_t[1], 1), round(t[1] / ours_t[0], 1)],
        }
    titan = FIG10["TITAN X"][1]
    c3["TITAN_X_vs_A100_baselines"] = {
        "claim_sec6.3": "TITAN X is already 20–100× higher than existing methods on an A100",
        "claim_fig10": "hundreds of times faster than a baseline running on an A100",
        "vs_BODex": [round(titan / 50, 1), round(titan / 30, 1)],
        "vs_DexGraspNet_SpringGrasp_lower_bound": round(titan / 3, 1),
    }
    out["C3_speedup_claims"] = c3

    # C4 Fig. 10
    c4 = {}
    for arch, (pct, sps) in FIG10.items():
        c4[arch] = {"sum_pct": round(sum(pct), 1), "contact_opt_pct": pct[2],
                    "kinematics_I_plus_II_pct": round(pct[3] + pct[4], 1),
                    "ms_per_valid_grasp": round(1000 / sps, 3)}
    c4["A100_fig10_vs_table1_allegro_apple"] = [FIG10["A100"][1], TABLE1["Allegro"][0][1]]
    c4["sps_ratio_4090_over_titanx"] = round(FIG10["RTX 4090"][1] / FIG10["TITAN X"][1], 2)
    out["C4_fig10_profile"] = c4

    # C5 메모리·원뿔 반각
    boxes = 30 * 10 * 10  # 1 cm 해상도
    leaf_bytes = boxes * 256 * 16
    node_bytes = 6000 * 48
    half = math.degrees(math.acos(1 - 2 / 256))  # 2π(1−cosθ) = 4π/256
    out["C5_memory_and_cone"] = {
        "boxes": boxes, "leaf_MB": round(leaf_bytes / 1e6, 2), "node_MB": round(node_bytes / 1e6, 3),
        "per_bvh_MB": round((leaf_bytes + node_bytes) / 1e6, 2),
        "100_fields_GB": round(100 * (leaf_bytes + node_bytes) / 1e9, 3),
        "normal_compression": 16 / 2, "cone_half_angle_deg_256": round(half, 2),
    }

    # C6 공개 코드 기본값 (demo.py, pipeline/module/*.py)
    outer, inner, n_retry = 128, 128, 10
    c6 = {
        "contact_opt_candidates": outer * inner,
        "ik_solves_coarse": outer * inner * n_retry,
        "zo_outer_steps": 5, "zo_variants_per_block": 6, "n_contact": 3,
        "score_evaluations_per_candidate": 5 * 3 * (6 + 1),
        "nnls_iters_cold_warm": [15, 4],
    }
    for sps in (1000, 3000):
        c6[f"yield_if_{sps}_valid_in_one_pass"] = round(sps / (outer * inner), 3)
    out["C6_code_default_budget"] = c6

    # C7 표본 간격 (issue #15의 계산과 같은 식)
    def spacing(area_m2, n=2048):
        return math.sqrt(area_m2 / n)
    out["C7_point_spacing"] = {
        "apple_r4cm_spacing_mm": round(1000 * spacing(4 * math.pi * 0.04 ** 2), 2),
        "apple_zo_step_sigma_mm": round(1000 * 5 * spacing(4 * math.pi * 0.04 ** 2), 1),
        "cube_18cm_spacing_mm": round(1000 * spacing(6 * 0.18 ** 2), 2),
        "cube_30cm_spacing_mm": round(1000 * spacing(6 * 0.30 ** 2), 2),
    }

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:400])
    print("wrote", path)


if __name__ == "__main__":
    main()

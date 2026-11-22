"""DexGraspNet (arXiv:2210.02697 v2) 보고 수치의 재계산과 내부 일관성 검사.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json (numpy, scipy, 1초 미만)
논문 값은 모두 paper/paper.txt(v2, pdftotext)에서 옮겼고, 원자료(grasp 데이터, 벤치마크 샘플 수)는 공개되지 않았습니다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

# ---------- C1, C2, C3: Table I (데이터셋 비교) ----------
# (grasps, objects, 보고된 "grasps per obj" 문자열의 하한 또는 값)
table1 = {
    "ObMan": (27_000, 2772, 10),
    "HO3D": (77_000, 10, 7_000),
    "DexYCB": (582_000, 20, 29_000),
    "ContactDB": (3_750, 50, 75),
    "ContactPose": (2_306, 25, 92),
    "DDGdata": (6_900, 565, 100),           # 표는 ">100"
    "DexGraspNet": (1_320_000, 5_355, 200),  # 표는 ">200"
}
t1 = {}
for k, (g, o, rep) in table1.items():
    per = g / o
    t1[k] = {"grasps": g, "objects": o, "per_object": round(per, 1), "reported_per_object_floor": rep,
             "consistent": per >= rep * 0.95}
res["C1_table1_grasps_per_object"] = t1
res["C3_objects_sum"] = {"3980_cad_plus_1375_scanned": 3980 + 1375, "reported": 5355}
res["C3_dexgraspnet_per_object"] = round(1_320_000 / 5_355, 1)
res["C3_scale_ratio_vs_ddg"] = {"grasps": round(1_320_000 / 6_900, 0), "objects": round(5355 / 565, 1)}

# ---------- C4, C5: 계산 비용 (본문 §IV) ----------
# 신 파이프라인: A100, 10000 grasp, 6000 step, 74 min, 유효 18%
# 원 알고리즘: 3090, 512 grasp, 10000 step, 37 min, 유효 3%
new = dict(batch=10_000, steps=6_000, minutes=74, valid=0.18)
old = dict(batch=512, steps=10_000, minutes=37, valid=0.03)
h_new_per10k_valid = (new["minutes"] / 60) / (new["batch"] * new["valid"]) * 10_000
h_old_per10k_valid = (old["minutes"] / 60) / (old["batch"] * old["valid"]) * 10_000
gpu_h_new_1p32m = (new["minutes"] / 60) / (new["batch"] * new["valid"]) * 1_320_000
gpu_h_old_1p32m = (old["minutes"] / 60) / (old["batch"] * old["valid"]) * 1_320_000
yield_ratio = new["valid"] / old["valid"]
step_ratio = old["steps"] / new["steps"]
per_gs_new = new["minutes"] * 60 / (new["batch"] * new["steps"])   # 초 / (grasp·step)
per_gs_old = old["minutes"] * 60 / (old["batch"] * old["steps"])
throughput_ratio = per_gs_old / per_gs_new
res["C4_cost"] = {
    "new_gpuh_per_10k_valid": round(h_new_per10k_valid, 2), "reported_new": 7,
    "old_gpuh_per_10k_valid": round(h_old_per10k_valid, 1), "reported_old": 400,
    "new_gpuh_1p32M_valid": round(gpu_h_new_1p32m), "reported_new_total": 950,
    "old_gpuh_1p32M_valid": round(gpu_h_old_1p32m), "reported_old_total": 50_000,
    "new_valid_per_hour": round(new["batch"] * new["valid"] / (new["minutes"] / 60)),
    "old_valid_per_hour": round(old["batch"] * old["valid"] / (old["minutes"] / 60), 1),
}
res["C5_speedup_decomposition"] = {
    "speedup_reported_400_over_7": round(400 / 7, 1),
    "speedup_recomputed": round(h_old_per10k_valid / h_new_per10k_valid, 1),
    "yield_ratio": round(yield_ratio, 2),
    "step_ratio": round(step_ratio, 3),
    "per_grasp_step_throughput_ratio": round(throughput_ratio, 2),
    "product_check": round(yield_ratio * step_ratio * throughput_ratio, 1),
    "note": "A100 FP32 19.49 TFLOPS 대 3090 35.58 TFLOPS (본문 명시). 처리량 비율 5.9는 하드웨어 피크로 설명되지 않음.",
}

# ---------- C6, C7: Table III, IV 방향 검사 ----------
# Table III: (success up, Q1 up, pen down). 열 순서: DexGraspNet 테스트 / DDGdata 테스트
t3 = {
    "DDG": {"train_DDGdata": {"DexGraspNet": (57.4, 0.0493, 0.353), "DDGdata": (56.4, 0.0461, 0.333)},
            "train_DexGraspNet": {"DexGraspNet": (67.5, 0.0582, 0.173), "DDGdata": (75.9, 0.0524, 0.134)}},
    "GraspTTA": {"train_DDGdata": {"DexGraspNet": (17.1, 0.0126, 0.720), "DDGdata": (23.7, 0.0265, 0.666)},
                 "train_DexGraspNet": {"DexGraspNet": (24.5, 0.0271, 0.678), "DDGdata": (39.3, 0.0790, 0.547)}},
}
checks = []
for m, d in t3.items():
    for test in ["DexGraspNet", "DDGdata"]:
        a = d["train_DDGdata"][test]; b = d["train_DexGraspNet"][test]
        checks.append({"method": m, "test": test, "success_up": b[0] > a[0], "Q1_up": b[1] > a[1], "pen_down": b[2] < a[2]})
res["C6_table3_monotonic"] = {"pairs": checks, "all_hold": all(c["success_up"] and c["Q1_up"] and c["pen_down"] for c in checks)}
res["C8_pen_vs_5mm"] = {"GraspTTA_DexGraspNet_mean_pen_mm": 6.78, "DDG_DexGraspNet_mean_pen_mm": 1.73,
                        "criterion_mm": 5.0}

h3 = {  # Table IV: (mean, std) 열: DexGraspNet 테스트 / DDGdata 테스트
    "DDG_train_DDGdata": ((4.958, 2.653), (3.709, 1.942)),
    "DDG_train_DexGraspNet": ((5.683, 1.993), (4.272, 1.287)),
    "GraspTTA_train_DDGdata": ((5.952, 0.934), (5.837, 1.047)),
    "GraspTTA_train_DexGraspNet": ((6.111, 0.569), (5.947, 0.528)),
}
t4ok = []
for k in ["DDG", "GraspTTA"]:
    a, b = h3[f"{k}_train_DDGdata"], h3[f"{k}_train_DexGraspNet"]
    for col, name in [(0, "DexGraspNet"), (1, "DDGdata")]:
        t4ok.append({"method": k, "test": name, "mean_up": b[col][0] > a[col][0], "std_down": b[col][1] < a[col][1]})
res["C7_table4_monotonic"] = {"pairs": t4ok, "all_hold": all(c["mean_up"] and c["std_down"] for c in t4ok)}

# ---------- C9: 엔트로피 단위와 유효 빈 수 ----------
K = 100  # 관절당 100 bin (본문 §IV)
H_max_bits = math.log2(K)
H_max_nats = math.log(K)
ent = {"DDGdata_dataset": 4.246, "DexGraspNet_dataset": 5.962,
       "T4_DDG_on_DexGraspNet": 5.683, "T4_GraspTTA_on_DexGraspNet": 6.111, "T4_GraspTTA_DDGdata_on_DexGraspNet": 5.952}
res["C9_entropy"] = {
    "max_bits_100bins": round(H_max_bits, 3), "max_nats_100bins": round(H_max_nats, 3),
    "reported_DexGraspNet_exceeds_nats_max": 5.962 > H_max_nats,  # 단위가 bit임을 보임
    "effective_bins_2^H": {k: round(2 ** v, 1) for k, v in ent.items()},
}
# 표본 수 의존성: plug-in 엔트로피는 log2 N 이하이고, Miller-Madow 보정은 (K-1)/(2N ln2) 비트
sens = {}
for N in [100, 500, 1_000, 6_900, 1_320_000]:
    sens[str(N)] = {"log2_N_cap": round(math.log2(N), 2),
                    "MM_bias_bits_if_K100": round((K - 1) / (2 * N * math.log(2)), 3)}
res["C9_sample_size_sensitivity"] = sens
res["C9_gap_T4_DDG_vs_GraspTTA_on_DexGraspNet_bits"] = round(6.111 - 5.683, 3)

# ---------- C11: 성공률 차이의 검정 (n 미기재) ----------
def two_prop_z(p1, p2, n):
    p1, p2 = p1 / 100, p2 / 100
    pbar = (p1 + p2) / 2
    se = math.sqrt(2 * pbar * (1 - pbar) / n)
    z = (p2 - p1) / se if se > 0 else float("inf")
    return z, 2 * (1 - stats.norm.cdf(abs(z)))

def min_n(p1, p2, alpha=0.05):
    lo, hi = 2, 10 ** 7
    while lo < hi:
        mid = (lo + hi) // 2
        _, p = two_prop_z(p1, p2, mid)
        if p < alpha:
            hi = mid
        else:
            lo = mid + 1
    return lo

def wilson(p, n, z=1.96):
    p = p / 100
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)

pairs = {
    "DDG_DexGraspNet_test_67.5_vs_57.4": (57.4, 67.5),
    "GraspTTA_DexGraspNet_test_24.5_vs_17.1": (17.1, 24.5),
    "DDG_DDGdata_test_75.9_vs_56.4": (56.4, 75.9),
    "GraspTTA_DDGdata_test_39.3_vs_23.7": (23.7, 39.3),
}
sig = {}
for name, (a, b) in pairs.items():
    sig[name] = {"diff_pts": round(b - a, 1), "min_n_per_arm_p05": min_n(a, b)}
res["C11_success_significance"] = sig
res["C11_wilson_examples"] = {f"p{p}_n{n}": wilson(p, n) for p in [17.1, 24.5] for n in [100, 500, 1000]}

# ---------- C12: Table II 비율 ----------
res["C12_table2_ratios"] = {"Q1_mean_100pct": round(0.1145 / 0.0712, 2), "Q1_mean_best10pct": round(0.2533 / 0.2277, 2),
                           "H_mean_diff_bits": round(5.962 - 4.246, 3),
                           "note": "표준편차·검정 없음. 'significantly better' 주장은 이 표만으로 확인 불가."}

# ---------- C10: 임계값 불일치 (본문 정의 기준) ----------
res["C10_thresholds"] = {"Q1_contact_threshold_TableII_mm": 1, "Q1_contact_threshold_TableIII_cm": 1,
                         "validation_penetration_SecIIIC_mm": 1, "benchmark_penetration_SecV_mm": 5}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    print(json.dumps(res, ensure_ascii=False, indent=2))

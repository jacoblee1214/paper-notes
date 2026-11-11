"""논문 수치 재계산 (CPU, 몇 초). 실행: python3 analysis/table_check.py → analysis/table_check_results.json

논문 Table II, III, 부록 B·D의 수치와 본문 값을 다시 계산한다. 논문에 없는 값(반복 분산, 시드별 결과)은 계산할 수 없다.
"""
import json
import math
from pathlib import Path

from scipy.stats import fisher_exact, norm

HERE = Path(__file__).resolve().parent
out = {}

# 1) 입자 수 → 표면 셀 (§III-C, 부록 D)
cells = {"218→451": (218, 451, 89, 105), "451→876": (451, 876, 105, 151)}
out["surface_cells"] = {
    k: {"added_particles": n1 - n0, "new_cells": s1 - s0, "share_into_occupied": 1 - (s1 - s0) / (n1 - n0)}
    for k, (n0, n1, s0, s1) in cells.items()
}
out["surface_cells"]["cell_growth_151_over_89"] = 151 / 89
out["surface_cells"]["particle_growth_876_over_218"] = 876 / 218

# 2) 강하 횟수 (Table III). 80 에폭은 2 시드 × 2 정밀도가 겹친 것이라 독립 표본이 아니다.
pairs = {"PRT-CD vs PCD-CD": ((78, 2), (43, 37)),
         "PRT-CD vs PRT-EMD": ((78, 2), (54, 26)),
         "PRT-CD vs PCD-EMD": ((78, 2), (46, 34))}
out["descent_fisher"] = {k: {"p_two_sided": float(fisher_exact([list(a), list(b)])[1])} for k, (a, b) in pairs.items()}

# 3) 라인서치 반대 방향 (부록 D): f32 14/58, f64 13/42
out["linesearch_fisher"] = {"p_two_sided": float(fisher_exact([[14, 44], [13, 29]])[1])}

# 4) HMD 차이 (반복 분산 없음 → 검정 불가, 차이 비율만)
out["hmd_rel_diff_PRTCD_vs_PRTEMD_pct"] = 100 * (3738.4 - 3698.6) / 3698.6

# 5) Table II: f32 손실 산포 V의 단조성. 논문은 "롤아웃이 길수록 빠르게 커짐"이라고 쓴다.
steps = [2, 5, 10, 20, 40]
v_f32 = [2.2e-7, 2.9e-5, 7.1e-5, 4.7e-5, 2.8e-5]   # Table II 의 f32 spread (하한 값으로 기록, 2스텝은 범위 시작)
fd_f64 = [1.5e-5, 2.75e-5, 2.93e-5, 3.41e-1, 2.70e-1]  # Table II 의 f64 최악 AD/FD (2스텝은 범위 끝)
out["table2_check"] = {
    "steps": steps,
    "f32_spread_monotone": all(b > a for a, b in zip(v_f32, v_f32[1:])),
    "f32_spread_peak_step": steps[v_f32.index(max(v_f32))],
    "f32_growth_2_to_10_factor": v_f32[2] / v_f32[0],
    "f32_growth_10_to_40_factor": v_f32[4] / v_f32[2],
    "fd_f64_jump_20_over_10": fd_f64[3] / fd_f64[2],
    "note": "f64 AD/FD는 f32 손실 산포 V와 비교해 판정했다. 산포의 단위가 다르다(04_claim_verification.md C2).",
}

# 6) ∆L/V 비율과 산포가 맞는지: 2스텝에서 ∆L = ratio × V
ratio_2 = (93, 806)
v2 = (2.2e-7, 1.4e-6)
out["dL_implied_2step"] = {"min": ratio_2[0] * v2[0], "max": ratio_2[1] * v2[1]}

# 7) 결정론 리덕션 비트 예산 (식 20, 부록 B)
def budget(n, p):
    N = math.ceil(math.log2(n))
    W = 53 - N
    B = 2 * W
    return {"N": N, "W": W, "B": B, "need_B_ge": p + 8, "ok": B >= p + 8, "bits_beyond_p": B - p}

out["bit_budget"] = {"DDBot_27440_f64": budget(27440, 53), "f32_example": budget(27440, 24)}

# 8) 부록 B 예시 검산 (식 17–19)
q = [round(c * 8) for c in (1.25, 0.875, 1.50)]
out["toy_example"] = {"q": q, "Q": sum(q), "Sb": sum(q) / 8, "Qhi": sum(q) // 16, "Qlo": sum(q) % 16}

# 9) 실행 시간·속도
out["timing"] = {"cpu_gpu_speedup": 260.821 / 20.488, "fd_slowdown_2_to_40": 178 / 25}

# 10) 메트릭 정밀도 불일치: ∆L/V에서 V는 f32, FD 분자는 f32/f64 중 어느 것인지 불명
out["metric_note"] = "V(식 13)는 f32 반복 산포로 보고되었다. f64 AD/FD 판정에 쓰였다면 잡음 바닥이 과대평가된다."

(HERE / "table_check_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
for k, v in out.items():
    print(k, json.dumps(v, ensure_ascii=False, default=float)[:300])

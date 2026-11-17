"""Play2Perfect (arXiv:2606.26428 v4) 보고 수치 재계산.

입력: 논문 본문·표 1 (v4), 부록 H, 부록 B, 부록 A/Fig. 10의 텍스트 수치.
출력: analysis/recompute_results.json
실행: python3 analysis/recompute.py   (CPU, 1초 안팎)
"""
import json
import math
import os

import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recompute_results.json")
Z = 1.959964


def wilson(k, n, z=Z):
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


res = {"source": "arXiv:2606.26428v4 (2026-10-15); v3 values from v3 Table 1"}

# 1. 표 1 실로봇 성공률 (v4)와 Wilson 95% 구간
cells_v4 = {
    "Tight 10mm": (10, 10), "Tight 2mm": (9, 10), "Tight 0.5mm": (6, 10),
    "Beam Step1 (end-to-end)": (16, 20), "Beam Step2 | Step1 ok": (12, 16),
    "Beam overall (12/20)": (12, 20),
    "Screw Insert": (7, 10), "Screw | Insert ok": (5, 7),
    "Screw overall (5/10, 유도)": (5, 10),
}
res["real_wilson_v4"] = {
    k: {"k": a, "n": b, "p": round(a / b, 3), "ci95": [round(x, 3) for x in wilson(a, b)]}
    for k, (a, b) in cells_v4.items()
}

# 2. 시험 수 합계와 실패 계수 점검 (본문 "60 rollouts, 18 failures")
rollouts = {"tight": 10 + 10 + 10, "beam": 20, "screw": 10}
fail = {"tight": (10 - 10) + (10 - 9) + (10 - 6), "beam": (20 - 16) + (16 - 12), "screw": (10 - 7) + (7 - 5)}
res["rollout_accounting"] = {
    "rollouts_by_task": rollouts,
    "rollouts_total": sum(rollouts.values()),
    "failures_by_task": fail,
    "failures_total": sum(fail.values()),
    "fig10_categories": {"timeout": 6, "drop_insertion": 5, "drop_reorientation": 3, "pose_tracking": 4},
    "fig10_total": 6 + 5 + 3 + 4,
    "note": "본문 60회 / 실패 18회와 일치하는지 확인. 표 1 tight 10mm·2mm·0.5mm는 각 10회로 가정.",
}

# 3. v3 표 1 값과 v4 값 비교 (같은 셀의 변경)
res["v3_to_v4_changes"] = {
    "Beam Step1": {"v3": "8/10 (80%), 6.9±1.9 s", "v4": "16/20 (80%), 시간 21.0±8.9 s (Beam 합산으로 보임)"},
    "Beam Step2": {"v3": "7/10 (70%), 6.4±2.5 s (단계별 독립 평가)", "v4": "12/16 (75%), Step1 성공 조건부 (연속 실행)"},
    "Screw": {"v3": "insert 7/10, screw 5/10, 15.6±2.9 s", "v4": "insert 7/10, screw 5/7 조건부, 15.6±2.9 s"},
    "note": "비율은 두 판본에서 거의 같지만 시행 수와 평가 방식이 바뀌었다. 판본 간 재현 여부는 확인 불가.",
}
res["v3_wilson_beam_step1_8_10"] = [round(x, 3) for x in wilson(8, 10)]
res["v3_wilson_beam_step2_7_10"] = [round(x, 3) for x in wilson(7, 10)]

# 4. 본문 주장 검정: Play2Perfect 대 Play-only (실로봇). Play-only의 n은 논문에 없음.
#    비율 60%, 20%, 0%를 n=5, 10, 20으로 바꿔 Fisher 정확 검정 민감도 확인.
p2p = {"10mm": (10, 10), "2mm": (9, 10), "0.5mm": (6, 10)}
play_only_pct = {"10mm": 0.60, "2mm": 0.20, "0.5mm": 0.00}
res["fisher_p2p_vs_playonly"] = {}
for clr, (k1, n1) in p2p.items():
    row = {}
    for n2 in (5, 10, 20):
        k2 = round(play_only_pct[clr] * n2)
        _, pv = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
        row[f"n2={n2}, k2={k2}"] = round(pv, 4)
    res["fisher_p2p_vs_playonly"][clr] = row

# 5. 시뮬레이션 33배 주장 산술
#    Play2Perfect 약 4 h로 Scratch(dense)가 도달한 성공률 도달. Scratch는 "100 h 이상".
#    사전학습 7일 (단일 RTX A6000, 부록 H)과 과제별 미세조정 1일은 4 h에 포함되지 않음.
scratch_h = np.array([100.0, 132.0])
p2p_h_finetune = 4.0
pre_h = 7 * 24.0
fin_h = 24.0
res["speedup_arithmetic"] = {
    "ratio_if_scratch_100h": round(100 / p2p_h_finetune, 2),
    "ratio_if_scratch_132h": round(132 / p2p_h_finetune, 2),
    "scratch_h_needed_for_33x": round(33 * p2p_h_finetune, 1),
    "play_pretrain_gpu_h_single_gpu": pre_h,
    "finetune_gpu_h_one_task_budget": fin_h,
    "p2p_total_h_incl_pretrain_one_task": round(pre_h + p2p_h_finetune, 1),
    "breakeven_tasks_vs_scratch_100h": round(pre_h / (100 - p2p_h_finetune), 2),
    "note": "33x는 scratch 약 132 h를 가정해야 나온다. 본문은 '100 h 이상'이라 25x 이상만 확인된다. 사전학습 168 h를 포함하면 한 과제에서는 불리하고 약 2과제부터 유리하다(부록 H 기준, 가정: 과제당 미세조정 4 h).",
}
res["sample_efficiency_claim"] = "초록은 '33x more sample-efficient'라 하지만 본문 근거는 벽시계 시간이다(Fig. 5, 4.1). 샘플 수는 보고하지 않는다."

# 6. 실로봇 0.5mm 60%와 스크류 '50% 초과' 점검
res["claim_checks"] = {
    "abstract_0.5mm_60pct": {"k": 6, "n": 10, "ci95": [round(x, 3) for x in wilson(6, 10)]},
    "abstract_screw_over_50pct": {
        "k": 5, "n": 10, "p": 0.5, "ci95": [round(x, 3) for x in wilson(5, 10)],
        "verdict": "점추정이 정확히 50%이고 구간이 50%를 포함한다. '50% 초과'는 근거가 약하다.",
    },
    "abstract_multipart_over_50pct": {"k": 12, "n": 20, "ci95": [round(x, 3) for x in wilson(12, 20)]},
}

# 7. 클리어런스 정의 산술 (부록 본문: 30×20 peg, 1 mm clearance = 31×21 hole)
res["clearance_geometry"] = {
    "peg_mm": [30, 20],
    "hole_for_1mm": [31, 21],
    "hole_for_40mm": [70, 60],
    "hole_for_0p2mm": [30.2, 20.2],
    "note": "클리어런스는 치수 차이(양쪽 합)로 정의됨. 한쪽 간극은 절반이다.",
}

# 8. 관측 차원 140 산술 (부록 J 목록과 저장소 obs_list 비교)
obs_dims = {
    "joint_pos": 29, "joint_vel": 29, "prev_action_targets": 29, "palm_pos": 3, "palm_rot": 4,
    "object_rot": 4, "fingertip_pos_rel_palm": 15, "keypoints_rel_palm": 12,
    "keypoints_rel_goal": 12, "object_scales": 3,
}
res["obs_dim_sum"] = {"terms": obs_dims, "sum": sum(obs_dims.values()), "paper": 140, "match": sum(obs_dims.values()) == 140}

# 9. 학습 자원 산술
res["compute"] = {
    "pretrain_envs": 24576, "pretrain_days": 7, "finetune_envs": 12288, "finetune_days": 1,
    "pretrain_gpu_h": 7 * 24, "note": "단일 A6000 가정(부록 H). 환경 수 24,576은 SAPG 블록 4,096의 배수라 저장소 제약과 맞는다.",
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(json.dumps({k: res[k] for k in ["rollout_accounting", "speedup_arithmetic", "claim_checks", "obs_dim_sum"]}, ensure_ascii=False, indent=1)[:3000])
print("wrote", OUT)

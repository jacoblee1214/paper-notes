"""ForceMimic (arXiv:2410.07554 v3) 보고 수치 재계산과 일관성 검사 (numpy, scipy).
실행: python3 analysis/recompute.py   결과: analysis/recompute_results.json
원자료(시행별 기록)는 없다. 모든 검사는 논문에 보고된 값끼리 맞는지, 그리고 보고된 성공 수로 구간과 검정을 다시 계산할 수 있는지를 본다.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def clopper(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (lo, hi)


# ---------------------------------------------------------------- 표 I (영상 제어 피링, 실로봇)
# 보고된 %와 시행 수에서 성공 수를 되살린다. 모든 %가 정수 성공 수로 맞는지 확인한다.
T1 = {
    # method: (n_trials, reported motion %, reported >10cm %)
    "Raw DP": (20, 80, 55),
    "Force DP": (10, 60, 10),
    "Force+Hybrid DP": (10, 80, 20),
    "HybridIL": (20, 100, 85),
}
counts = {}
for m, (n, p_motion, p_len) in T1.items():
    k_motion = p_motion / 100 * n
    k_len = p_len / 100 * n
    counts[m] = {
        "n": n,
        "motion_k": round(k_motion, 6),
        "motion_k_is_integer": bool(abs(k_motion - round(k_motion)) < 1e-9),
        "len10_k": round(k_len, 6),
        "len10_k_is_integer": bool(abs(k_len - round(k_len)) < 1e-9),
    }
    if abs(k_motion - round(k_motion)) < 1e-9:
        counts[m]["motion_wilson95"] = [round(x, 3) for x in wilson(round(k_motion), n)]
        counts[m]["motion_cp95"] = [round(x, 3) for x in clopper(round(k_motion), n)]
    if abs(k_len - round(k_len)) < 1e-9:
        counts[m]["len10_wilson95"] = [round(x, 3) for x in wilson(round(k_len), n)]
        counts[m]["len10_cp95"] = [round(x, 3) for x in clopper(round(k_len), n)]
res["table1_counts"] = counts

# 정확 검정 (두 쪽 Fisher). 시행 수가 작아 검정력이 낮다.
def fisher(a, b):
    na, nb = T1[a][0], T1[b][0]
    ka_m, ka_l = round(T1[a][1] / 100 * na), round(T1[a][2] / 100 * na)
    kb_m, kb_l = round(T1[b][1] / 100 * nb), round(T1[b][2] / 100 * nb)
    out = {}
    for name, ka, kb in [("motion", ka_m, kb_m), ("len10", ka_l, kb_l)]:
        tab = [[ka, na - ka], [kb, nb - kb]]
        out[name] = {"table": tab, "p_two_sided": round(float(stats.fisher_exact(tab)[1]), 4)}
    return out

res["fisher"] = {
    "HybridIL_vs_RawDP": fisher("HybridIL", "Raw DP"),
    "HybridIL_vs_ForceHybridDP": fisher("HybridIL", "Force+Hybrid DP"),
    "RawDP_vs_ForceDP": fisher("Raw DP", "Force DP"),
}

# 상대 향상률. 초록의 54.5%는 >10cm 기준 HybridIL 대 Raw DP와 맞는다(85/55).
res["relative_gain"] = {
    "len10_HybridIL_vs_RawDP_pct": round((85 - 55) / 55 * 100, 2),
    "motion_HybridIL_vs_RawDP_pct": round((100 - 80) / 80 * 100, 2),
    "note": "초록의 '54.5% relatively'는 >10cm 기준 Raw DP 대비 상대 향상과 일치. 동작 성공 기준은 25%",
}

# ---------------------------------------------------------------- 데이터와 시간 (본문 IV)
res["dataset"] = {
    "zucchinis": 15,
    "segments": 438,
    "action_sequences": 30199,
    "segments_per_zucchini": round(438 / 15, 2),
    "sequences_per_segment": round(30199 / 438, 2),
    "note": "시퀀스/구간 평균 약 69. 학습 샘플 정의(20스텝 창, 겹침 여부)는 본문에 없음",
}
res["teleop_time"] = {
    "fig5b_minutes_per_zucchini": {"Human": 2.9, "ForceCapture": 4.5, "Teleoperation": 13},
    "ratio_teleop_over_forcecapture_fig": round(13 / 4.5, 2),
    "ratio_teleop_over_forcecapture_abstract_5min": round(13 / 5, 2),
    "note": "Fig. 5(b) 막대에 오차 막대 없음, 각 방법 1회 측정으로 보임. 본문의 '약 3배'는 Fig 값(2.9)과 맞음",
}

# ---------------------------------------------------------------- 힘 크기 (Fig. 7 본문)
res["force_levels_N"] = {
    "HybridIL_mean_v1": 7.5,
    "HybridIL_mean_v3": 9.0,
    "dataset_typical_v1": 6.0,
    "dataset_typical_v3": 10.0,
    "RawDP_typical": 20.0,
    "note": "v1과 v3의 본문 힘 수치가 다름: HybridIL 평균 7.5 → 9 N, 데이터 힘 6 → 10 N. Raw DP 20 N과 40 N 이상은 두 버전 같음",
}

# ---------------------------------------------------------------- 학습 설정 일관성 (코드와 대조는 04)
res["training_claims"] = {
    "epochs_paper": 500,
    "epochs_config_train_json": 1000,
    "epochs_deploy_checkpoint": 1000,
    "ratio_samples_per_epoch_note": "config의 num_epochs 1000과 본문 500이 다름",
    "force_threshold_N": 6,
    "horizon_pred_act_obs": [20, 10, 1],
    "loss_weights_arm_hand": [0.7, 0.3],
}

# ---------------------------------------------------------------- 평가 기준의 통계력 (참고)
# 실제 성공률 p1, p2에서 두 쪽 Fisher 검정의 검정력을 이항 분포 전개로 정확히 계산한다(몬테카를로 아님).
def power_fisher(p1, p2, n1, n2, alpha=0.05):
    pw = 0.0
    for a in range(n1 + 1):
        pa = stats.binom.pmf(a, n1, p1)
        for b in range(n2 + 1):
            if stats.fisher_exact([[a, n1 - a], [b, n2 - b]])[1] < alpha:
                pw += pa * stats.binom.pmf(b, n2, p2)
    return pw

res["power_len10_85_vs_55_n20"] = round(power_fisher(0.85, 0.55, 20, 20), 3)
res["power_motion_100_vs_80_n20"] = round(power_fisher(1.00, 0.80, 20, 20), 3)
res["power_len10_85_vs_55_by_n"] = {str(n): round(power_fisher(0.85, 0.55, n, n), 3) for n in (10, 20, 30, 40, 60)}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))

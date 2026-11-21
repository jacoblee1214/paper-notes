"""TacBridge (arXiv:2611.20939v1) 표 수치 재계산과 일관성 검사.

논문이 주는 숫자(표 I–III와 본문)만 쓴다. 원자료(시험 샘플별 점수)는 공개되지 않아
이 스크립트는 '재계산 가능한 것'과 '불가능한 것'을 분리해 기록한다.
실행: python3 analysis/recompute.py   (수 초, 표준 라이브러리만 사용)
"""
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "recompute_results.json")

# ---- 표 I (frozen linear probe). 열별 값: 낮을수록 좋은 열은 RMSE 표기 ----
TABLE_I_A = {  # 열: (방향, {모델: 값})
    "TAG_GS": ("higher", {"CLIP": 51.65, "VJEPA_S": 54.67, "MAE_S": 59.47, "AnyTouch": 80.82, "TacBridge": 73.13}),
    "Cloth_GS": ("higher", {"CLIP": 26.76, "VJEPA_S": 18.66, "MAE_S": 19.40, "AnyTouch": 24.34, "TacBridge": 35.65}),
    "Pose_DG": ("higher", {"CLIP": 54.54, "VJEPA_S": 55.09, "MAE_S": 55.92, "AnyTouch": 56.22, "TacBridge": 63.79}),
    "SlipF1_DG": ("higher", {"CLIP": 33.13, "VJEPA_S": 83.33, "MAE_S": 83.30, "AnyTouch": 40.60, "TacBridge": 85.77}),
    "ForceChg_DG": ("lower", {"CLIP": 174.39, "VJEPA_S": 105.63, "MAE_S": 98.33, "AnyTouch": 169.42, "TacBridge": 91.26}),
    "SlipF1_Mini": ("higher", {"CLIP": 85.47, "VJEPA_S": 97.00, "MAE_S": 97.50, "AnyTouch": 88.92, "TacBridge": 97.90}),
    "ForceChg_Mini": ("lower", {"CLIP": 177.67, "VJEPA_S": 121.31, "MAE_S": 102.64, "AnyTouch": 162.41, "TacBridge": 104.56}),
}
TABLE_I_B = {
    "Force_DG": ("lower", {"CLIP": 1278.08, "VJEPA_S": 957.73, "MAE_S": 821.26, "AnyTouch": 1235.11, "TacBridge": 680.83}),
    "Force_Mini": ("lower", {"CLIP": 553.19, "VJEPA_S": 428.56, "MAE_S": 297.96, "AnyTouch": 488.31, "TacBridge": 185.94}),
    "ToucHD_DG": ("lower", {"CLIP": 4880.94, "VJEPA_S": 4766.11, "MAE_S": 1953.82, "AnyTouch": 3968.81, "TacBridge": 1011.79}),
    "ToucHD_Mini": ("lower", {"CLIP": 4492.77, "VJEPA_S": 3208.10, "MAE_S": 3655.39, "AnyTouch": 4050.45, "TacBridge": 1009.31}),
}


def wins(table):
    res = {}
    for col, (direction, vals) in table.items():
        base = {k: v for k, v in vals.items() if k != "TacBridge"}
        tb = vals["TacBridge"]
        best_base = max(base.values()) if direction == "higher" else min(base.values())
        best_all = (max if direction == "higher" else min)(vals.values())
        res[col] = {
            "tacbridge": tb,
            "best_baseline": best_base,
            "tacbridge_best_overall": tb == best_all,
            "relative_change_vs_best_baseline_pct": round(100 * (tb - best_base) / best_base, 1),
        }
    return res


def main():
    out = {}
    wa = wins(TABLE_I_A)
    wb = wins(TABLE_I_B)
    out["table1_panelA"] = wa
    out["table1_panelB"] = wb
    n_wins = sum(v["tacbridge_best_overall"] for v in {**wa, **wb}.values())
    out["table1_wins"] = {"r": n_wins, "of": len(wa) + len(wb)}
    # 본문: 패널 B 값은 최강 기준선 대비 17%–69% 감소
    red_b = [-v["relative_change_vs_best_baseline_pct"] for v in wb.values()]
    out["table1_panelB_reduction_pct"] = {"min": round(min(red_b), 1), "max": round(max(red_b), 1)}

    # 표 II·III 본문 수치 검산
    t2 = {"same_psnr": (33.38, 30.69), "same_cpsnr": (28.98, 26.46), "x_psnr": (33.23, 30.52),
          "x_cpsnr": (28.83, 26.30), "x_dcpsnr": (11.93, 10.30)}
    out["table2_gaps_vs_anytouch"] = {k: round(a - b, 2) for k, (a, b) in t2.items()}
    out["table2_crosssensor_vs_same_gap"] = {"psnr": round(33.38 - 33.23, 2), "cpsnr": round(28.98 - 28.83, 2)}
    out["table2_rfid_drop"] = {"anytouch": 65.8, "tacbridge": 60.1}

    out["table3_gap_ood_vs_anytouch_pp"] = round(73.0 - 70.2, 1)
    out["table3_gap_id_vs_best_baseline_pp"] = round(95.5 - 94.5, 1)  # 최강 기준선은 MAE(S) 94.5

    # 구조 검산: 표본 수
    out["counts"] = {
        "table2_pairs_expected_9x8_per_contact": 9 * 8,
        "table2_contacts_implied": 15552 // (9 * 8),
        "table2_pairs_check": 216 * 72 == 15552,
        "table3_images": 50 * 325,
        "table3_per_fabric_split_check": 224 + 101 == 325,
        "table3_fabrics_check": 45 + 5 == 50,
    }

    # OOD 분할: 독립 단위는 test fabric 5개. 기준 차이 2.8 pp를 유의하게 만들 fabric 간 SD 상한
    # 쌍체 t 근사: 차이의 SE = s/sqrt(5). t(4, .975)=2.776.
    t_crit = 2.776
    gap = 73.0 - 70.2
    s_max = gap / (t_crit / math.sqrt(5))  # SD of paired difference at which t = 2.776
    out["ood_effective_n"] = 5
    out["ood_gap_pp"] = round(gap, 2)
    out["ood_sd_threshold_pp_for_p05"] = round(s_max, 2)
    out["ood_ci_halfwidth_pp_for_sd"] = {str(s): round(t_crit * s / math.sqrt(5), 2) for s in (1.0, 2.0, 4.0, 6.0)}

    # ID 분할 차이 1.0 pp: 같은 계산 (5 fabrics 기준이 아니라 50 fabrics 기준이라 가정은 별도 표기)
    out["id_gap_pp"] = 1.0
    out["id_sd_threshold_pp_if_50_fabrics"] = round(1.0 / (2.009 / math.sqrt(50)), 2)  # t(49,.975)=2.0096

    # 수식 (11) 검산: vψ = (Zτ - Ẑ)/τ 와 Ẑ - Z = -τ (vψ - (ε - Z)) 가 같은 식인지 수치로 확인
    import random
    random.seed(0)
    max_err = 0.0
    for _ in range(10000):
        z = random.gauss(0, 1)
        eps = random.gauss(0, 1)
        tau = random.uniform(0.01, 1.0)
        zt = (1 - tau) * z + tau * eps
        zh = random.gauss(0, 1)  # 임의의 예측값
        v = (zt - zh) / tau
        lhs = zh - z
        rhs = -tau * (v - (eps - z))
        max_err = max(max_err, abs(lhs - rhs))
    out["eq11_identity_max_abs_err"] = max_err

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Table I–IV·본문 수치 재계산과 정합성 점검 (CPU). 실행: python3 analysis/table_check.py

- C1: "F*_min 8.6×, 12.3×, 8.2× 향상" (Sec. IV-A)
- C2: Table II 여유(margin)에서 f_max 역산, 그리고 "여유 > 1인데 위반이 있음" 모순 점검
- C3: Table III 여유와 "허용 하중의 2.5배 초과" (Sec. IV-B)
- C4: Table IV 여유 = 1/λ 인지 (제어 법칙의 동어반복 여부)
- C5: 실험 1과 실험 2의 "제안 방법" 실행이 같은지 (플라스틱 컵)
- C6: 마찰로 물체를 들기 위한 최소 수직력과 보고된 힘의 비교
- C7: 실패한 under-stiffness(0.08 N)와 성공한 gently(0.09 N)의 차이
- C8: 시행 수 (Table II·III의 x/5는 손가락 수)
- C9: DexGraspNet 관절 수 (논문 24 vs DexGraspNet 코드 22)
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G = 9.81

# Table I: 영역별 허용 횡하중 [N]
TABLE1 = {
    "paper cup": {"rim": 41.2, "wall": 4.8, "base": 88.8},
    "plastic cup": {"rim": 3.7, "wall": 0.3, "base": 8.0},
    "glass goblet": {"bowl": 69.0, "stem": 565.0, "base": 989.0},
}
MASS_G = {"paper cup": 7.6, "plastic cup": 1.9, "glass goblet": 155.0}  # Sec. IV-A

# Table II: (zone, F*_min [N], margin = F*_min / f_max, violations / 5 fingers)
TABLE2 = {
    "paper cup": {"Baseline": ("wall", 4.8, 1.4, 1), "Ours": ("rim", 41.2, 7.8, 0)},
    "plastic cup": {"Baseline": ("wall", 0.3, 1.2, 2), "Ours": ("rim", 3.7, 5.9, 0)},
    "glass goblet": {"Baseline": ("bowl", 69.0, 2.1, 1), "Ours": ("stem", 565.0, 12.4, 0)},
}
# Table III: 플라스틱 컵, F*_wall = 0.3 N. (f_max [N], margin, violations/5, success)
F_WALL = 0.3
TABLE3 = {
    "Under-stiffness (0.3 K_base)": (0.08, None, 0, False),
    "Over-stiffness (2.0 K_base)": (0.74, 0.41, 4, False),
    "Ours (lambda=0.7)": (0.21, 1.43, 0, True),
}
# Table IV: mode -> (lambda, f_max [N], margin, success)
TABLE4 = {"Gently": (0.3, 0.09, 3.33, True), "Default": (0.7, 0.21, 1.43, True), "Firmly": (1.0, 0.29, 1.03, True)}


def main() -> None:
    out: dict = {}

    # C1 향상 배수
    out["C1_fmin_improvement"] = {
        obj: round(r["Ours"][1] / r["Baseline"][1], 2) for obj, r in TABLE2.items()
    } | {"claim": "8.6x, 12.3x, 8.2x (Sec. IV-A 1)"}

    # C2 Table II: f_max = F*_min / margin. 여유 > 1이면 모든 손가락에서 f_k <= f_max < F*_min <= F*_k
    #     이므로, 위반(f_k > F*_k)은 F*_k < F*_min 인 손가락(계획에 없던 접촉)이 있을 때만 가능
    c2 = {}
    for obj, rows in TABLE2.items():
        for m, (zone, fmin, margin, viol) in rows.items():
            fmax = fmin / margin
            c2[f"{obj}/{m}"] = {
                "zone": zone,
                "f_max_implied_N": round(fmax, 3),
                "grip_over_weight": round(fmax / (MASS_G[obj] / 1000 * G), 1),
                "violations": f"{viol}/5",
                "margin_gt_1_but_violation": margin > 1 and viol > 0,
            }
    out["C2_table2_backcalc"] = c2

    # C3 Table III
    out["C3_table3"] = {
        name: {
            "margin_recomputed": None if f is None else round(F_WALL / f, 3),
            "reported": mg,
            "overload_factor": round(f / F_WALL, 2),
        }
        for name, (f, mg, _, _) in TABLE3.items()
    } | {"claim": "over-stiffness exceeds the admissible load by 2.5x"}

    # C4 Table IV: f_max / (lambda * F_wall)와 margin * lambda
    out["C4_table4_margin_is_inverse_lambda"] = {
        mode: {
            "f_over_lambdaF": round(f / (lam * F_WALL), 3),
            "margin_times_lambda": round(mg * lam, 3),
            "margin_recomputed": round(F_WALL / f, 3),
        }
        for mode, (lam, f, mg, _) in TABLE4.items()
    }

    # C5 실험 1(Table II)과 실험 2(Table III)의 플라스틱 컵 "Ours"
    e1_zone, e1_fmin, e1_margin, _ = TABLE2["plastic cup"]["Ours"]
    out["C5_exp1_vs_exp2_plastic_ours"] = {
        "exp1_reference_F_N": e1_fmin,
        "exp1_zone": e1_zone,
        "exp1_f_max_N": round(e1_fmin / e1_margin, 3),
        "exp2_reference_F_N": F_WALL,
        "exp2_f_max_N": TABLE3["Ours (lambda=0.7)"][0],
        "exp2_over_stiffness_vs_rim_admissible": round(TABLE3["Over-stiffness (2.0 K_base)"][0] / e1_fmin, 3),
        "note": "실험 2는 '힘 지도로 고른 파지를 고정'한다고 하지만 기준값은 벽(0.3 N)이고 Fig. 4의 손끝도 벽에 있음. "
                "실험 1의 선택 결과는 테두리(3.7 N). 테두리 기준이면 over-stiffness 0.74 N도 위반이 아님",
    }

    # C6 마찰 최소 수직력: 손가락 n개가 같은 크기 f로 누를 때 mu * n * f >= m g (정적, 중력 방향 마찰만)
    c6 = {}
    for obj, mass in MASS_G.items():
        w = mass / 1000 * G
        c6[obj] = {
            "weight_N": round(w, 4),
            **{f"f_per_finger_N_mu{mu}_n{n}": round(w / (mu * n), 4) for mu in (0.3, 0.6) for n in (2, 5)},
        }
    c6["plastic_under_stiffness_f_max_N"] = TABLE3["Under-stiffness (0.3 K_base)"][0]
    c6["plastic_under_stiffness_ratio_to_need_mu0.3_n2"] = round(
        TABLE3["Under-stiffness (0.3 K_base)"][0] / c6["plastic cup"]["f_per_finger_N_mu0.3_n2"], 2)
    c6["note"] = "공식 코드 기본 마찰계수: 정지 0.6, 동 0.4 (scripts/convert_mesh_to_usd_isaaclab.py)"
    out["C6_friction_hold"] = c6

    # C7 성공 경계
    f_fail = TABLE3["Under-stiffness (0.3 K_base)"][0]
    f_ok = TABLE4["Gently"][1]
    out["C7_success_boundary"] = {
        "under_stiffness_f_max_N": f_fail,
        "gently_f_max_N": f_ok,
        "relative_gap": round((f_ok - f_fail) / f_fail, 3),
        "stiffness_factor_under": 0.3,
        "stiffness_factor_gently_if_Fk_eq_Fmax": 0.3,
        "note": "실험 2의 제안 제어기 K_k = K_base * F*_k / F_max * lambda. F*_k = F_max이면 gently(λ=0.3)와 "
                "under-stiffness(0.3 K_base)는 같은 강성. F_max는 정의되지 않음",
    }

    # C8 시행 수: x/5는 손가락 다섯 개. 조건당 rollout 1회로 보임
    out["C8_trials"] = {
        "exp1_rollouts": sum(len(r) for r in TABLE2.values()),
        "exp2_rollouts": len(TABLE3) + len(TABLE4) - 1,  # Default = Ours(λ=0.7) 중복
        "denominator_5_means": "fingers of one grasp (Table II caption: fingers with f_k > F*_k)",
    }

    # C9 DexGraspNet 관절 수: grasp_generation/utils/initializations.py의 joint_angles_mu 길이 22,
    #     hand_model.hand_pose = 3 + 6 + n_dofs (PKU-EPIC/DexGraspNet @ bd1b13d)
    out["C9_dexgraspnet_dims"] = {"paper_eq1_theta": 24, "dexgraspnet_n_dofs": 22,
                                  "paper_g_dim": 3 + 6 + 24, "dexgraspnet_hand_pose_dim": 3 + 6 + 22}

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:260])
    print("wrote", path)


if __name__ == "__main__":
    main()

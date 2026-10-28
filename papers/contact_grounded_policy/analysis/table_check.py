"""Table II–VI과 그림 수치 재계산 (CPU). 실행: python3 analysis/table_check.py

- C1: Table II 성공률 × 시행 수가 정수인지 (sim 250회, real 15회)
- C2: Wilson 95% 구간, CGP vs 최강 기준선 Fisher 정확검정, 과제 묶음별 Mantel–Haenszel 검정
- C3: Table III 상대 감소율과 3 seed Welch t 검정 (mean ± std에서)
- C4: Table IV에서 활성 taxel 비율 상한 (MAE / Active MAE)
- C5: Table VI에서 PSNR ↔ MAE 일관성 (RMSE/MAE 비), Digit360 압축률
- C6: Table V 차원 확인과 diffusion 출력 차원 (CGP vs DP)
- C7: 그림에서 읽은 값: Fig. 7 (KL ablation), Fig. 8 (추론 시간), Fig. 13 (촉각 잡음)
- C8: 실행 시간 예산 (5 Hz 해석 두 가지)
"""
import json
import math
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

# Table II: 과제 → (시행 수, CGP, Visuotactile DP, Visuomotor DP) 성공률
TABLE2 = {
    "sim/in_hand_box_flipping": (250, 0.660, 0.580, 0.532),
    "sim/fragile_egg_grasping": (250, 0.748, 0.700, 0.532),
    "sim/dish_wiping": (250, 0.584, 0.436, 0.424),
    "real/jar_opening": (15, 0.933, 0.667, 0.733),
    "real/in_hand_box_flipping": (15, 0.800, 0.600, 0.600),
}
METHODS = ("CGP", "Visuotactile DP", "Visuomotor DP")

# Table III: (입력, 인코더) → (abs mean, abs std, res mean, res std), ×1e-3 rad, 3 seeds
TABLE3 = {
    "S+T/ResNet1D": (8.80, 0.24, 5.94, 0.20),
    "S+T/MLP": (12.50, 0.32, 8.33, 0.32),
    "S+T/Transformer": (14.39, 0.38, 9.58, 0.48),
    "S/-": (16.05, 0.39, 10.64, 0.38),
    "T/ResNet1D": (35.93, 0.89, 12.15, 0.20),
    "T/MLP": (36.86, 0.25, 12.72, 0.25),
    "T/Transformer": (43.11, 0.91, 14.62, 0.43),
}

# Table IV: 인코더 → 과제 → (MAE, Active MAE, KL), MAE 단위 1e-2 N
TABLE4 = {
    "MLP (w/ KL)": {"box": (1.37, 12.24, 0.21), "egg": (0.77, 6.70, 0.15), "dish": (1.86, 8.14, 0.15)},
    "MLP (w/o KL)": {"box": (1.12, 9.93, 0.59), "egg": (0.54, 4.51, 0.40), "dish": (1.40, 6.20, 0.42)},
    "ResNet1D (w/ KL)": {"box": (1.26, 12.07, 0.12), "egg": (0.69, 5.95, 0.22), "dish": (1.54, 6.80, 0.24)},
    "ResNet1D (w/o KL)": {"box": (0.97, 9.91, 0.73), "egg": (0.45, 3.92, 0.43), "dish": (1.02, 4.49, 0.45)},
    "Transformer (w/ KL)": {"box": (1.66, 14.88, 0.19), "egg": (1.58, 10.58, 0.28), "dish": (3.19, 13.57, 0.21)},
    "Transformer (w/o KL)": {"box": (1.69, 14.60, 0.97), "egg": (1.18, 9.08, 0.61), "dish": (3.28, 13.57, 0.53)},
}

# Table VI: (latent, KL 유무) → 과제 → (MAE ×1e-3, KL loss, PSNR dB, SSIM), 이미지 [0, 1] 정규화
TABLE6 = {
    (80, "w/ KL"): {"box": (9.01, 0.0694, 35.2050, 0.9870), "jar": (8.10, 0.0647, 35.9445, 0.9883)},
    (80, "w/o KL"): {"box": (4.55, 0.7781, 40.4590, 0.9939), "jar": (3.88, 0.6247, 42.9845, 0.9957)},
    (160, "w/ KL"): {"box": (9.41, 0.0341, 35.2768, 0.9874), "jar": (8.40, 0.0337, 36.5394, 0.9891)},
    (160, "w/o KL"): {"box": (4.66, 0.7902, 40.4800, 0.9941), "jar": (3.98, 0.6899, 42.9143, 0.9957)},
    (320, "w/ KL"): {"box": (9.02, 0.0171, 35.2693, 0.9870), "jar": (8.29, 0.0155, 36.1177, 0.9885)},
    (320, "w/o KL"): {"box": (4.56, 0.7539, 40.5191, 0.9941), "jar": (3.84, 0.6787, 43.0426, 0.9957)},
}

# 그림에서 눈으로 읽은 근사값 (±1 눈금 정도의 오차)
FIG7_SUCCESS = {"box": (66, 62), "egg": (75, 64)}  # (w/ KL, w/o KL) %
FIG8_MS = {  # (encoders, unet, mapping) ms, A100, 8 DDIM steps
    "tactile_array": {"Visuomotor": (8, 85, 0), "Visuotactile": (14, 83, 0), "CGP": (15, 85, 12)},
    "digit360": {"Visuomotor": (10, 86, 0), "Visuotactile": (47, 86, 0), "CGP": (46, 85, 3)},
}
FIG13 = {  # 잡음 0% → 50% 에서의 MAE (×1e-3 rad)
    "S+T residual": (6.0, 10.0), "S+T absolute": (9.0, 13.3),
    "T-only residual": (12.0, 22.3), "T-only absolute": (36.0, 70.5),
}


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 3), round(c + h, 3)]


def mantel_haenszel(tables: list[tuple[int, int, int, int]]) -> dict:
    """2×2 표 여러 개 [(a, b, c, d)] = [[성공, 실패] 방법 A; [성공, 실패] 방법 B]의 MH 공통 OR과 χ² (연속성 보정)."""
    num = den = 0.0
    sa = ea = va = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        num += a * d / n
        den += b * c / n
        r1, c1 = a + b, a + c
        sa += a
        ea += r1 * c1 / n
        va += r1 * (c + d) * c1 * (b + d) / (n * n * (n - 1))
    chi2 = (abs(sa - ea) - 0.5) ** 2 / va
    return {"common_OR": round(num / den, 2), "chi2": round(chi2, 2), "p": float(f"{stats.chi2.sf(chi2, 1):.2e}")}


def welch(m1, s1, m2, s2, n=3) -> dict:
    t, p = stats.ttest_ind_from_stats(m1, s1, n, m2, s2, n, equal_var=False)
    return {"t": round(float(t), 2), "p": float(f"{p:.1e}")}


def main() -> None:
    out: dict = {}

    # C1 정수성
    c1 = {}
    for task, (n, *srs) in TABLE2.items():
        ks = [sr * n for sr in srs]
        c1[task] = {"counts": [round(k) for k in ks], "integer": all(abs(k - round(k)) < 0.02 for k in ks)}
    out["C1_table2_integer_counts"] = c1

    # C2 구간과 검정
    c2, mh_tables = {}, {"sim": [], "real": []}
    for task, (n, *srs) in TABLE2.items():
        ks = [round(sr * n) for sr in srs]
        best = max(range(1, 3), key=lambda i: ks[i])
        _, p = stats.fisher_exact([[ks[0], n - ks[0]], [ks[best], n - ks[best]]])
        c2[task] = {
            "CGP": [ks[0], n, wilson(ks[0], n)],
            "best_baseline": [METHODS[best], ks[best], wilson(ks[best], n)],
            "gain_pp": round(100 * (srs[0] - srs[best]), 1),
            "fisher_p_two_sided": round(float(p), 4),
        }
        mh_tables[task.split("/")[0]].append((ks[0], n - ks[0], ks[best], n - ks[best]))
    c2["MH_sim_CGP_vs_best"] = mantel_haenszel(mh_tables["sim"])
    c2["MH_real_CGP_vs_best"] = mantel_haenszel(mh_tables["real"])
    c2["note"] = "sim 250회는 한 학습 실행의 마지막 5개 체크포인트 × 50회라 독립이 아님. 구간은 낙관적"
    out["C2_ci_and_tests"] = c2

    # C3 Table III
    s_t, s, t = TABLE3["S+T/ResNet1D"], TABLE3["S/-"], TABLE3["T/ResNet1D"]
    out["C3_table3"] = {
        "residual_S+T_vs_S_only_reduction": round(1 - s_t[2] / s[2], 3),
        "residual_S+T_vs_T_only_reduction": round(1 - s_t[2] / t[2], 3),
        "residual_vs_absolute_reduction": {k: round(1 - v[2] / v[0], 3) for k, v in TABLE3.items()},
        "welch_residual_S+T_vs_S_only": welch(s_t[2], s_t[3], s[2], s[3]),
        "welch_residual_S+T_vs_T_only": welch(s_t[2], s_t[3], t[2], t[3]),
        "best_MAE_deg": round(math.degrees(s_t[2] * 1e-3), 3),
        "missing_baseline": "a = x (잔차 0) 기준선과 평균 |a − x| 크기가 보고되지 않음",
    }

    # C4 활성 taxel 비율 상한: MAE = f·Active + (1 − f)·Inactive, Inactive ≥ 0 → f ≤ MAE / Active
    out["C4_active_fraction_upper_bound"] = {
        enc: {task: round(v[0] / v[1], 3) for task, v in d.items()} for enc, d in TABLE4.items()
    }
    out["C4_kl_ratio_wo_over_w"] = {
        enc: {task: round(TABLE4[f"{enc} (w/o KL)"][task][2] / TABLE4[f"{enc} (w/ KL)"][task][2], 1)
              for task in ("box", "egg", "dish")}
        for enc in ("MLP", "ResNet1D", "Transformer")
    }

    # C5 PSNR = 10 log10(1 / MSE) → RMSE, RMSE/MAE (가우시안이면 1.25)
    c5 = {}
    for (dim, kl), d in TABLE6.items():
        for task, (mae, _, psnr, _) in d.items():
            rmse = 10 ** (-psnr / 20)
            c5[f"{dim}/{kl}/{task}"] = {"rmse": round(rmse, 4), "rmse_over_mae": round(rmse / (mae * 1e-3), 2)}
    out["C5_psnr_vs_mae"] = c5
    out["C5_digit360_compression"] = {
        "values_per_sensor": 72 * 72 * 3,
        "latent_per_sensor": 20,
        "ratio": round(72 * 72 * 3 / 20, 1),
        "tactile_array_values": 748 * 3,
        "tactile_array_latent": 32,
        "tactile_array_ratio": round(748 * 3 / 32, 1),
    }

    # C6 차원
    T = 16
    out["C6_dimensions"] = {
        "sim_action_3+6+20": 3 + 6 + 20, "sim_state_3+4+20": 3 + 4 + 20,
        "real_action_3+6+16": 3 + 6 + 16, "real_state_3+4+16": 3 + 4 + 16,
        "diffusion_output_dim": {
            "sim_CGP_T*(27+32)": T * (27 + 32), "sim_DP_T*29": T * 29,
            "real_CGP_T*(23+80)": T * (23 + 80), "real_DP_T*25": T * 25,
        },
        "ratio_CGP_over_DP": {"sim": round((27 + 32) / 29, 2), "real": round((23 + 80) / 25, 2)},
    }

    # C7 그림 값
    out["C7_fig7_kl_ablation"] = {
        task: {"w_kl": a, "wo_kl": b, "diff_pp": a - b,
               "fisher_p_if_250_each": round(float(stats.fisher_exact(
                   [[round(a * 2.5), 250 - round(a * 2.5)], [round(b * 2.5), 250 - round(b * 2.5)]])[1]), 4)}
        for task, (a, b) in FIG7_SUCCESS.items()
    }
    out["C7_fig8_inference_ms"] = {
        sensor: {m: {"total": sum(v), "mapping_share": round(v[2] / sum(v), 3)} for m, v in d.items()}
        for sensor, d in FIG8_MS.items()
    }
    s_only_res = TABLE3["S/-"][2]
    st0, st50 = FIG13["S+T residual"]
    out["C7_fig13_noise"] = {
        "S+T_residual_increase_at_50pct": round(st50 / st0 - 1, 2),
        "state_only_residual_table3": s_only_res,
        "tactile_benefit_remaining_at_50pct": round((s_only_res - st50) / (s_only_res - st0), 2),
        "noise_pct_where_S+T_equals_state_only_linear_extrapolation": round(50 * (s_only_res - st0) / (st50 - st0), 0),
    }

    # C8 실행 시간 예산
    infer_s = max(sum(v) for d in FIG8_MS.values() for v in d.values()) / 1000
    out["C8_timing"] = {
        "A_5Hz_is_action_step": {"step_s": 0.2, "replan_every_s": 8 * 0.2, "horizon_s": 16 * 0.2},
        "B_5Hz_is_inference_rate": {"inference_period_s": 0.2, "action_step_s": round(0.2 / 8, 4),
                                    "max_inference_s_A100": infer_s, "budget_used": round(infer_s / 0.2, 2)},
        "note": "논문은 '5 Hz rollout'과 Ta = 8만 밝힘. 어느 해석이든 tactile 피드백은 재계획 때만 정책에 들어감",
    }

    # C9 기타
    out["C9_misc"] = {
        "hand_config_frames_per_episode": round(4114 / 150, 1),
        "hand_config_train_episodes_1to1": 75,
        "real_trials": 15,
        "real_min_detectable_gap_pp_at_alpha_0.05": "15 vs 15에서 p < 0.05가 되려면 대략 14/15 vs 8/15 (40%p) 이상",
    }
    _, p_14_8 = stats.fisher_exact([[14, 1], [8, 7]])
    _, p_14_9 = stats.fisher_exact([[14, 1], [9, 6]])
    out["C9_misc"]["fisher_14v8"] = round(float(p_14_8), 4)
    out["C9_misc"]["fisher_14v9"] = round(float(p_14_9), 4)

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:400])
    print("wrote", path)


if __name__ == "__main__":
    main()

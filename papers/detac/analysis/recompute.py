"""DeTac(arXiv:2610.16463) 표 값의 재계산과 일관성 검사.

입력은 논문 표 1–4, 표 A1에서 옮긴 값이다(코드 안 상수). 결과는 recompute_results.json 에 저장한다.
실행: python3 analysis/recompute.py   (numpy, scipy, 1초 미만)
검사 라벨 V1–V12는 03_analysis.md 의 절과 대응한다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"

# 표 1: (FID, KID x1e-2, PSNR, SSIM) — Unseen pose / Unseen object
T1 = {
    "TELEA":            {"pose": (237.95, 18.327, 31.33, 0.870), "obj": (232.66, 17.592, 31.29, 0.868)},
    "TacDiff":          {"pose": (158.94, 11.447, 31.52, 0.866), "obj": (155.63, 10.654, 31.40, 0.863)},
    "LaMa":             {"pose": (358.16, 37.535, 29.09, 0.804), "obj": (360.27, 37.117, 29.16, 0.803)},
    "SD 1.5":           {"pose": (121.70, 7.401, 25.44, 0.861),  "obj": (114.29, 6.703, 24.16, 0.826)},
    "U-Net":            {"pose": (108.35, 7.489, 38.56, 0.950),  "obj": (111.28, 7.750, 39.03, 0.942)},
    "Pix2Pix":          {"pose": (112.53, 7.472, 38.53, 0.951),  "obj": (109.66, 7.065, 38.98, 0.948)},
    "dual-head":        {"pose": (104.27, 6.923, 37.99, 0.949),  "obj": (105.18, 6.794, 38.37, 0.947)},
    "TacDiff (enl.)":   {"pose": (172.01, 11.898, 30.93, 0.852), "obj": (180.96, 12.384, 30.84, 0.847)},
    "LaMa (enl.)":      {"pose": (330.13, 32.754, 30.23, 0.840), "obj": (330.57, 32.351, 30.27, 0.839)},
    "DeTac w/o Lsur":   {"pose": (25.78, 0.733, 34.64, 0.946),   "obj": (22.59, 0.698, 35.05, 0.945)},
    "DeTac w/o Lcon":   {"pose": (22.65, 0.689, 37.77, 0.949),   "obj": (19.44, 0.634, 38.28, 0.948)},
    "DeTac Full":       {"pose": (20.44, 0.683, 39.45, 0.953),   "obj": (17.06, 0.573, 39.07, 0.950)},
}
METRICS = ["FID", "KID", "PSNR", "SSIM"]
LOWER = {"FID": True, "KID": True, "PSNR": False, "SSIM": False}

# 표 2: MCLE (mm)
T2 = {"UVTac": 0.21, "SAM 3.1": 0.41, "DeTac w/o Lsur": 0.00839, "DeTac w/o Lcon": 0.00832, "DeTac Full": 0.00762}
# 표 3: 평균 힘 MAE (N) / 표 4: Chamfer(mm), RMSE(mm)
T3 = {"Mixed": 0.202, "UVTac": 0.119, "SAM 3.1": 0.482, "DeTac": 0.089}
T4 = {"Mixed": (0.165, 0.111), "Shading (DeTac)": (0.113, 0.071)}
# 표 A1: 성공률(%) 세 직물, 각 30회
TA1 = {"raw": (16.7, 43.3, 90.0, 50.0), "deTac": (83.3, 73.3, 96.7, 84.4)}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def feasible_counts(pct, n, tol=0.05):
    return [k for k in range(n + 1) if abs(100 * k / n - pct) <= tol]


def main():
    R = {}

    # V1 데이터 분할 산술
    objs, poses = 54, 20
    total = objs * poses
    unseen_obj_imgs = 5 * poses
    unseen_pose_imgs = 49
    train = (objs - 5) * poses - unseen_pose_imgs
    R["V1_split"] = {"total": total, "unseen_object_images": unseen_obj_imgs,
                     "unseen_pose_images": unseen_pose_imgs, "train_images": train,
                     "sum_check": train + unseen_obj_imgs + unseen_pose_imgs,
                     "steps_per_epoch_bs2": math.ceil(train / 2),
                     "total_steps_300_epochs": 300 * math.ceil(train / 2),
                     "aux_loss_steps_after_100_epochs": 200 * math.ceil(train / 2),
                     "fid_cov_rank_max_pose": unseen_pose_imgs - 1,
                     "fid_cov_rank_max_object": unseen_obj_imgs - 1, "inception_feature_dim": 2048}

    # V2 "over 7 dB over TacDiff"
    R["V2_psnr_gain_vs_tacdiff"] = {s: round(T1["DeTac Full"][s][2] - T1["TacDiff"][s][2], 2) for s in ("pose", "obj")}
    R["V2_psnr_gain_vs_unet"] = {s: round(T1["DeTac Full"][s][2] - T1["U-Net"][s][2], 2) for s in ("pose", "obj")}
    R["V2_psnr_gain_vs_best_deterministic"] = {
        s: round(T1["DeTac Full"][s][2] - max(T1[m][s][2] for m in ("U-Net", "Pix2Pix", "dual-head")), 2) for s in ("pose", "obj")}

    # V3 열별 최고값과 DeTac Full 이 최고인지
    best = {}
    for s in ("pose", "obj"):
        for mi, m in enumerate(METRICS):
            vals = {k: v[s][mi] for k, v in T1.items()}
            b = min(vals, key=vals.get) if LOWER[m] else max(vals, key=vals.get)
            second = sorted(vals.items(), key=lambda kv: kv[1], reverse=not LOWER[m])[1]
            best[f"{s}_{m}"] = {"best": b, "second": second[0], "gap": round(abs(vals[b] - second[1]), 3)}
    R["V3_column_best"] = best

    # V4 절제: Full 대비 변화
    abl = {}
    for name in ("DeTac w/o Lsur", "DeTac w/o Lcon"):
        for s in ("pose", "obj"):
            full, ab = T1["DeTac Full"][s], T1[name][s]
            abl[f"{name}|{s}"] = {m: round(ab[i] - full[i], 3) for i, m in enumerate(METRICS)}
            abl[f"{name}|{s}"]["FID_ratio"] = round(ab[0] / full[0], 2)
    R["V4_ablation_delta_vs_full"] = abl
    R["V4_wo_Lsur_psnr_below_deterministic"] = {
        s: {m: T1[m][s][2] for m in ("U-Net", "Pix2Pix", "dual-head")} | {"w/o Lsur": T1["DeTac w/o Lsur"][s][2]} for s in ("pose", "obj")}

    # V5 확대 마스크의 효과(원 모델 대비)
    enl = {}
    for base, e in (("TacDiff", "TacDiff (enl.)"), ("LaMa", "LaMa (enl.)")):
        for s in ("pose", "obj"):
            d = {m: round(T1[e][s][i] - T1[base][s][i], 3) for i, m in enumerate(METRICS)}
            d["improved_metrics"] = [m for i, m in enumerate(METRICS)
                                     if (T1[e][s][i] < T1[base][s][i]) == LOWER[m] and T1[e][s][i] != T1[base][s][i]]
            enl[f"{base}|{s}"] = d
    R["V5_enlarged_mask_vs_original"] = enl

    # V6 표 2 MCLE 비율
    R["V6_mcle"] = {"ratio_uvtac_over_full": round(T2["UVTac"] / T2["DeTac Full"], 1),
                    "ratio_sam_over_full": round(T2["SAM 3.1"] / T2["DeTac Full"], 1),
                    "wo_Lsur_rel_increase_pct": round(100 * (T2["DeTac w/o Lsur"] / T2["DeTac Full"] - 1), 1),
                    "wo_Lcon_rel_increase_pct": round(100 * (T2["DeTac w/o Lcon"] / T2["DeTac Full"] - 1), 1),
                    "wo_Lsur_minus_wo_Lcon_pct_of_full": round(100 * (T2["DeTac w/o Lsur"] - T2["DeTac w/o Lcon"]) / T2["DeTac Full"], 1),
                    "full_in_um": round(T2["DeTac Full"] * 1000, 2),
                    "mm_per_px_for_1px_error_equals_full": T2["DeTac Full"]}

    # V7 표 3, 4 상대 개선
    R["V7_force_surface"] = {
        "force_rel_reduction_vs_mixed_pct": round(100 * (1 - T3["DeTac"] / T3["Mixed"]), 1),
        "force_rel_reduction_vs_uvtac_pct": round(100 * (1 - T3["DeTac"] / T3["UVTac"]), 1),
        "force_sam_over_mixed_ratio": round(T3["SAM 3.1"] / T3["Mixed"], 2),
        "chamfer_rel_reduction_pct": round(100 * (1 - T4["Shading (DeTac)"][0] / T4["Mixed"][0]), 1),
        "rmse_rel_reduction_pct": round(100 * (1 - T4["Shading (DeTac)"][1] / T4["Mixed"][1]), 1),
        "chamfer_abs_diff_um": round(1000 * (T4["Mixed"][0] - T4["Shading (DeTac)"][0]), 1),
        "rmse_abs_diff_um": round(1000 * (T4["Mixed"][1] - T4["Shading (DeTac)"][1]), 1),
    }

    # V8 표 A1 정수 가능성, 합산, Wilson, Fisher
    n = 30
    a1 = {"feasible_counts": {}, "counts": {}, "wilson": {}, "fisher_p": {}}
    for key, (a, b, c, tot) in TA1.items():
        ks = [feasible_counts(x, n)[0] for x in (a, b, c)]
        a1["counts"][key] = ks
        a1["feasible_counts"][key] = [feasible_counts(x, n) for x in (a, b, c)]
        a1["counts"][key + "_overall"] = sum(ks)
        a1["overall_pct_recomputed_" + key] = round(100 * sum(ks) / 90, 2)
        a1["wilson"][key] = [[round(100 * v, 1) for v in wilson(k, n)] for k in ks] + [[round(100 * v, 1) for v in wilson(sum(ks), 90)]]
    for i, f in enumerate(("A", "B", "C")):
        t = [[a1["counts"]["deTac"][i], n - a1["counts"]["deTac"][i]], [a1["counts"]["raw"][i], n - a1["counts"]["raw"][i]]]
        a1["fisher_p"][f] = float(stats.fisher_exact(t)[1])
    t = [[a1["counts"]["deTac_overall"], 90 - a1["counts"]["deTac_overall"]], [a1["counts"]["raw_overall"], 90 - a1["counts"]["raw_overall"]]]
    a1["fisher_p"]["overall"] = float(stats.fisher_exact(t)[1])
    ps = [a1["fisher_p"][f] for f in ("A", "B", "C")]
    order = np.argsort(ps)
    adj, run = [0.0] * 3, 0.0
    for rank, idx in enumerate(order):
        run = max(run, min(1.0, (3 - rank) * ps[idx]))
        adj[idx] = run
    a1["holm_adjusted_three_fabrics"] = {f: adj[i] for i, f in enumerate(("A", "B", "C"))}
    # 분류기 범주 수 미기재: 우연 수준 하한(범주 3–5 가정)
    a1["chance_if_k_classes_pct"] = {k: round(100 / k, 1) for k in (3, 4, 5)}
    R["V8_tableA1"] = a1

    # V9 부록 A.3: 200 g 대 500 g 이미지 차이 MAE 3.072 의 PSNR 상한
    mae = 3.072
    R["V9_force_variation_mae"] = {"mae_0_255": mae, "psnr_upper_bound_db_if_rmse_ge_mae": round(20 * math.log10(255 / mae), 2),
                                   "note": "RMSE >= MAE 이므로 같은 두 이미지의 PSNR은 이 값 이하"}

    # V10 학습 비용
    steps = R["V1_split"]["total_steps_300_epochs"]
    R["V10_training_cost"] = {"total_steps": steps, "hours_reported": 14, "sec_per_step_implied": round(14 * 3600 / steps, 3)}

    # V11 Table 1 상호 일관성: 순위 합산
    ranks = {}
    for k in T1:
        r = 0
        for s in ("pose", "obj"):
            for mi, m in enumerate(METRICS):
                vals = sorted([T1[x][s][mi] for x in T1], reverse=not LOWER[m])
                r += vals.index(T1[k][s][mi]) + 1
        ranks[k] = r / 8
    R["V11_mean_rank_over_8_columns"] = {k: round(v, 2) for k, v in sorted(ranks.items(), key=lambda kv: kv[1])}

    # V12 FID 와 PSNR 의 관계(방법 간): SD 1.5 와 deterministic 사이 FID 비
    R["V12_fid_ratio"] = {"deterministic_best_over_full_pose": round(104.27 / 20.44, 2),
                          "deterministic_best_over_full_obj": round(105.18 / 17.06, 2),
                          "sd15_over_full_pose": round(121.70 / 20.44, 2)}

    OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(R, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

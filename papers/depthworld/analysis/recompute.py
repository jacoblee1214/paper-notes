"""DepthWorld (arXiv:2610.08780 v1) 표 수치 재계산과 일관성 검사.

논문이 준 평균·중앙값만 쓴다. 원자료(창별 값, 시드별 값)는 없으므로 신뢰구간은 만들 수 없고,
대신 '검정이 유의해지려면 창별 SD가 얼마여야 하는가'를 계산한다(가정 명시).
실행: python3 analysis/recompute.py  → analysis/recompute_results.json (수 초, 표준 라이브러리만 사용)
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).with_name("recompute_results.json")
RES = {}

# ---- 표 1: 보정 품질 (중앙값 px·mm, 평균 IoU). 논문 값 그대로 ----
t1 = {
    "all": {"n": 995, "EE": (5.84, 0.25), "WE": (13.19, 1.03), "RD": (111.3, 14.0), "IoU": (0.440, 0.810)},
    "converged": {"n": 815, "EE": (5.67, 0.25), "WE": (12.52, 1.02), "RD": (104.6, 13.9), "IoU": (0.458, 0.819)},
}
RES["table1_ratios"] = {}
for k, v in t1.items():
    RES["table1_ratios"][k] = {
        "EE_x": round(v["EE"][0] / v["EE"][1], 2),
        "WE_x": round(v["WE"][0] / v["WE"][1], 2),
        "RD_x": round(v["RD"][0] / v["RD"][1], 2),
        "IoU_diff": round(v["IoU"][1] - v["IoU"][0], 3),
    }
RES["table1_subset_check"] = {"fail": 180, "n_all": 995, "fail_pct": round(100 * 180 / 995, 2), "n_converged": 995 - 180}

# ---- 표 2: 세계 모델 RGB 품질 (PSNR, SSIM, LPIPS) 차이 ----
t2 = {
    "external": {"RGB": (22.63, 0.8170, 0.0930), "RGBD": (24.09, 0.8333, 0.0889), "PM": (24.11, 0.8333, 0.0891)},
    "wrist": {"RGB": (16.98, 0.5737, 0.3228), "RGBD": (17.96, 0.6106, 0.3137), "PM": (18.00, 0.6129, 0.3131)},
}
RES["table2_rgb_deltas"] = {}
for view, d in t2.items():
    RES["table2_rgb_deltas"][view] = {
        "RGBD_minus_RGB": [round(a - b, 4) for a, b in zip(d["RGBD"], d["RGB"])],
        "PM_minus_RGB": [round(a - b, 4) for a, b in zip(d["PM"], d["RGB"])],
        "PM_minus_RGBD": [round(a - b, 4) for a, b in zip(d["PM"], d["RGBD"])],
    }
# 표 2 깊이 지표 (AbsRel, RMSE, δ1): RGB+D 대 PM-DPT
t2d = {
    "external": {"RGBD": (0.0765, 0.220, 0.9432), "PM": (0.0782, 0.219, 0.9434)},
    "wrist": {"RGBD": (0.2230, 0.150, 0.8182), "PM": (0.2182, 0.148, 0.8210)},
}
RES["table2_depth_PM_minus_RGBD"] = {v: [round(a - b, 4) for a, b in zip(d["PM"], d["RGBD"])] for v, d in t2d.items()}
# 교차 뷰 재투영 (본문): RGB+D 17.60 대 RGB 16.60
RES["cross_view_psnr_gain"] = round(17.60 - 16.60, 2)

# ---- 검정력 감각: 창 단위 쌍체 비교라고 가정했을 때 유의해지는 창별 SD ----
# 실제 창별 SD와 시드 간 분산은 논문에 없다. 아래 값은 '이 정도 SD면 유의하다/아니다'를 보여 줄 뿐이다.
n = 256
d_psnr = 1.46  # ext, RGB+D - RGB (표 2)
z_crit = 1.97  # 양측 5%, n = 256 근사 (t 분포 ~1.97)
RES["sensitivity_psnr_ext"] = {
    "n_windows": n,
    "diff_dB": d_psnr,
    "sd_for_p05_if_paired_windows": round(d_psnr * math.sqrt(n) / z_crit, 2),
    "se_if_sd_2dB": round(2 / math.sqrt(n), 3),
    "se_if_sd_4dB": round(4 / math.sqrt(n), 3),
    "note": "창별 SD는 논문에 없음. 시드 1개, 학습 실행 1회라서 학습 시드 분산은 측정되지 않음",
}

# ---- 표 A2: 통합 방식 비교 (약 70k 스텝, 조건 불일치 명시) ----
a2 = {
    "external": {"dual": (21.94, 0.8039, 0.1010, 0.0861, 0.232, 0.9383),
                 "channel": (22.22, 0.8017, 0.1021, 0.0796, 0.220, 0.9414),
                 "tiling": (23.66, 0.8252, 0.0945, 0.0854, 0.235, 0.9383)},
    "wrist": {"dual": (16.05, 0.5441, 0.3554, 0.2523, 0.161, 0.7685),
              "channel": (16.34, 0.5593, 0.3422, 0.2801, 0.158, 0.7930),
              "tiling": (17.58, 0.5954, 0.3276, 0.2420, 0.159, 0.8001)},
}
RES["tableA2_tiling_minus_other"] = {}
for view, d in a2.items():
    RES["tableA2_tiling_minus_other"][view] = {
        "vs_dual_PSNR_SSIM_LPIPS": [round(d["tiling"][i] - d["dual"][i], 4) for i in range(3)],
        "vs_channel_PSNR_SSIM_LPIPS": [round(d["tiling"][i] - d["channel"][i], 4) for i in range(3)],
        "tiling_depth_worse_than_channel": [name for name, i in
                                            [("AbsRel", 3), ("RMSE", 4)]
                                            if d["tiling"][i] > d["channel"][i]],
    }

# ---- 표 A3: 깊이 소스 (50k 스텝, K=10) ----
a3 = {
    "external": {"S2M2": (24.07, 0.0877, 0.074, 0.946), "NEURAL": (24.26, 0.0881, 0.079, 0.939),
                 "ULTRA": (22.11, 0.0947, 0.194, 0.830)},
    "wrist": {"S2M2": (17.68, 0.3270, 0.266, 0.797), "NEURAL": (17.77, 0.3260, 0.254, 0.794),
              "ULTRA": (16.28, 0.3395, 0.461, 0.655)},
}
RES["tableA3_vs_S2M2"] = {}
for view, d in a3.items():
    RES["tableA3_vs_S2M2"][view] = {
        src: [round(d[src][i] - d["S2M2"][i], 4) for i in range(4)] for src in ("NEURAL", "ULTRA")
    }
    RES["tableA3_vs_S2M2"][view]["ULTRA_AbsRel_ratio"] = round(d["ULTRA"][2] / d["S2M2"][2], 2)
# 본문 '2.0 dB, 1.4 dB' 확인
RES["tableA3_text_check"] = {"ext_dB_drop": round(24.07 - 22.11, 2), "wrist_dB_drop": round(17.68 - 16.28, 2)}
# 표 2(RGB+D)와 표 A3(S2M2 = 같은 모델로 기재)의 불일치
RES["table2_vs_A3_mismatch"] = {
    "ext_PSNR": round(24.09 - 24.07, 2),
    "wrist_PSNR": round(17.96 - 17.68, 2),
    "ext_AbsRel": round(0.0765 - 0.074, 4),
    "wrist_AbsRel": round(0.2230 - 0.266, 4),
    "wrist_delta1": round(0.8182 - 0.797, 4),
}

# ---- 표 A4: 외부 파라미터 출처 (PM-DPT) ----
RES["tableA4_refined_minus_factory"] = {
    "ext_PSNR": round(24.13 - 24.10, 2), "wrist_PSNR": round(18.03 - 18.01, 2),
    "wrist_AbsRel": round(0.220 - 0.233, 3),
}

# ---- 표 A5: PointWorld 대 DepthWorld (중앙값, mm) ----
a5 = {
    "tracked_l2_moving_2s": (23, 31), "tracked_l2_moving_8s": (56, 46),
    "chamfer_moving_2s": (21, 25), "chamfer_moving_8s": (58, 32),
    "chamfer_whole_2s": (8, 8), "chamfer_whole_8s": (18, 9),
}
RES["tableA5"] = {k: {"PW": v[0], "DW": v[1], "DW_minus_PW": v[1] - v[0]} for k, v in a5.items()}
RES["tableA5_growth_2s_to_8s"] = {
    "PW_tracked": round(56 / 23, 2), "DW_tracked": round(46 / 31, 2),
    "PW_chamfer_whole": round(18 / 8, 2), "DW_chamfer_whole": round(9 / 8, 2),
}

# ---- TesserAct 비교 (본문 §5, 부록 D.5) ----
RES["tesseract_gap"] = {"psnr_dB": round(24.07 - 19.43, 2), "absrel_ratio": round(0.154 / 0.074, 2),
                      "eval_windows": {"DepthWorld": 256, "TesserAct": 128}}

# ---- 보정 GT 자세 비교 (본문 §5): 중앙값 ----
RES["gt_pose_ratio"] = {"extrinsic_trans_mm_ratio": round(72 / 4.9, 1),
                      "extrinsic_rot_deg_ratio": round(3.0 / 0.39, 1)}

# ---- DROID-3D 에피소드 수 대조 (논문 vs 공개 데이터 카드) ----
hf_calibrated, hf_missing = 71006, 2111
RES["episode_count_check"] = {
    "paper_scenes": 71100,
    "hf_card_calibrated": hf_calibrated,
    "paper_minus_hf": 71100 - hf_calibrated,
    "hf_raw_indexed_sum": hf_calibrated + hf_missing,
    "coverage_pct": round(100 * hf_calibrated / (hf_calibrated + hf_missing), 2),
    "code_readme_coverage_pct": 97.1,
    "hf_pseudolabel_episodes_not_in_paper": 25616,
}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(RES, ensure_ascii=False, indent=2))
print("wrote", OUT)

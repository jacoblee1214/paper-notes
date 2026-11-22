"""VisionSIM (arXiv:2611.08368 v1) 본문과 부록의 수치를 다시 계산하고 일관성을 검사한다.

각 검사는 V1, V2, ... 번호를 가진다 (analysis/03_analysis.md §1과 같은 번호).
출력: analysis/recompute_results.json
실행: python3 analysis/recompute.py   (numpy만 사용, 수 초)
논문에 적힌 수치는 PAPER 딕셔너리에 위치(절, 식, 표, 그림)와 함께 옮겼다.
"""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "recompute_results.json"
C = 299_792_458.0  # m/s

R = {}


def put(key, **kw):
    R[key] = kw


# ---- V1. 데이터셋 개수 (Sec. 7) -------------------------------------------------
scenes, seconds, fps, stereo = 100, 24, 100, 2
gt_steps = scenes * seconds * fps
put(
    "V1_dataset_counts",
    paper_claim_views=480_000,
    gt_time_steps=gt_steps,
    gt_views_with_stereo=gt_steps * stereo,
    ok_480k=gt_steps * stereo == 480_000,
    spc_factor=16_000 // fps,
    event_factor=4_000 // fps,
    conv_stride=fps // 25,
    spc_frames_total=scenes * seconds * 16_000,
    event_slices_total=scenes * seconds * 4_000,
    conv_frames_total=scenes * seconds * 25,
    imu_samples_total=scenes * seconds * 250,
    project_page_claim="100M raw data points from eight sensors (프로젝트 페이지)",
    note="SPC 프레임 38.4M + 이벤트 구간 9.6M + GT 0.48M = 48.5M. dToF, iToF, 열화상 프레임 수가 논문에 없어 100M은 확인 불가",
)

# ---- V2. 저장 용량 추정 (Sec. 7, S.4) -------------------------------------------
px = 1000 * 1000
spc_frames = scenes * seconds * 16_000
spc_bitpacked_tb = spc_frames * px / 8 / 1e12
spc_bytewise_tb = spc_frames * px / 1e12
f32_channel_tb = gt_steps * stereo * px * 4 / 1e12
put(
    "V2_storage_estimates",
    spc_bitpacked_TB=spc_bitpacked_tb,
    spc_one_byte_per_pixel_TB=spc_bytewise_tb,
    one_float32_channel_all_views_TB=f32_channel_tb,
    one_scene_spc_bitpacked_GB=seconds * 16_000 * px / 8 / 1e9,
    compression_ratio_bitpack=8,
    note="1000x1000, SPC 단일 시점 가정. 논문이 원본 대신 재생성 스크립트를 배포하는 이유(Sec. 7 Release)와 맞는 규모",
)

# ---- V3. 보간 배율과 RIFE 2의 거듭제곱 제한 --------------------------------------
def is_pow2(n):
    return n >= 2 and (n & (n - 1)) == 0


put(
    "V3_interpolation_factors",
    factors={"SPC 16 kHz": 160, "event 4 kHz": 40, "S.2.7 6.4 kHz": 16},
    is_power_of_two={str(k): is_pow2(v) for k, v in {"SPC": 160, "event": 40, "S27": 16}.items()},
    nearest_pow2_spc=[128, 256],
    nearest_pow2_event=[32, 64],
    note="공식 코드는 보간 배율이 2의 거듭제곱이어야 한다 (cli/interpolate.py). 160과 40은 아님",
)

# ---- V4. 렌더 시간 (S.2.7) -------------------------------------------------------
full_s = 4 * 3600 + 3 * 60
speed_99 = full_s / (4 * 60)
t_at_99_8 = full_s * 99.8 / 6400
t_at_400 = full_s * 400 / 6400
put(
    "V4_render_time",
    full_render_s=full_s,
    claimed_speedup_x=60,
    speedup_if_exactly_4min=speed_99,
    predicted_time_at_99_8fps_min=t_at_99_8 / 60,
    predicted_time_at_400fps_min=t_at_400 / 60,
    claimed_400fps_min="just under 20",
    ratio_400fps_vs_full=full_s / (20 * 60),
    note="렌더 시간이 프레임 수에 비례한다고 두면 99.8 fps는 약 3.8분, 400 fps는 약 15분(+보간). 서술과 일치",
)

# ---- V5. 식 S8 키프레임 배수 탐색 ------------------------------------------------
def s8_step(k, v, vt=1.0, alpha=0.95, i=0, power_i=True, lo=1 / 15, hi=15.0):
    ratio = min(max(v / vt, lo), hi)
    damp = alpha ** i if power_i else alpha
    return k + damp * k * (ratio - 1.0)


k0, k1_paper, k2_paper = 5.0, 30.05, 11.17
# 첫 단계: V0 ~ 6 px
v0_needed_pow = (k1_paper - k0) / (1.0 * k0) + 1.0            # alpha^0 = 1
v0_needed_const = (k1_paper - k0) / (0.95 * k0) + 1.0         # 감쇠가 상수일 때
# 두 번째 단계: V1 = 0.37 px (논문)
k2_pred_pow = s8_step(k1_paper, 0.37, i=1)
k2_pred_const = s8_step(k1_paper, 0.37, i=1, power_i=False)
v1_needed = 1.0 + (k2_paper - k1_paper) / (0.95 * k1_paper)
put(
    "V5_keyframe_multiplier_search",
    paper={"K0": k0, "V0_approx_px": 6.0, "K1": k1_paper, "V1_px": 0.37, "K2": k2_paper, "target_px": 1.0, "alpha": 0.95},
    V0_implied_by_K1_if_alpha_pow_i=v0_needed_pow,
    V0_implied_by_K1_if_alpha_const=v0_needed_const,
    K2_predicted_with_V1_0_37_alpha_pow_i=k2_pred_pow,
    K2_predicted_alpha_const=k2_pred_const,
    V1_implied_by_K2=v1_needed,
    relative_gap_K2=(k2_paper - k2_pred_pow) / k2_paper,
    product_V_times_K_at_K0=6.01 * k0,
    product_V_times_K_at_K1=0.37 * k1_paper,
    note="V이 1/K에 비례하면 V*K가 일정해야 한다. K=5에서 30, K=30에서 11로 2.7배 어긋남 (작은 흐름이 많아 1차 근사가 깨짐). 마지막 K=11.2에서 V = 0.37*30.05/11.2 = 0.99 px로 목표에 맞음",
    base_fps_from_K_4_16=99.8 / 4.16,
    code_default_damping="scale_decay**i (cli/blender.py optimize_rate)",
)

# ---- V6. iToF 모호 거리와 위상 오차 -> 깊이 오차 (Sec. 4.2, Sec. 6, S.5.3) ---------
def amb_range(f):
    return C / (2 * f)


def depth_from_phase(dphi, f):
    return dphi * C / (4 * math.pi * f)


d013_40 = depth_from_phase(0.13, 40e6)
phase_for_5_3cm = 0.053 * 4 * math.pi * 40e6 / C
f_for_5_3cm = 0.13 * C / (4 * math.pi * 0.053)
put(
    "V6_itof_range_and_phase_error",
    range_35MHz_m=amb_range(35e6),
    claimed_35MHz_m=4.3,
    range_20MHz_m=amb_range(20e6),
    range_40MHz_m=amb_range(40e6),
    phase_error_rad=0.13,
    depth_error_at_40MHz_cm=100 * d013_40,
    claimed_depth_error_cm=5.3,
    ratio_computed_over_claimed=100 * d013_40 / 5.3,
    phase_that_gives_5_3cm_rad=phase_for_5_3cm,
    frequency_that_gives_5_3cm_for_0_13rad_MHz=f_for_5_3cm / 1e6,
    note="식 depth = phase * c / (4 pi f). 0.13 rad은 40 MHz에서 7.75 cm이지 5.3 cm가 아니다 (논문 Sec. 6)",
)

# ---- V7. 마이크로볼로미터 IIR (식 S16-S20) ---------------------------------------
def a_of(dt, tau):
    return math.exp(-dt / tau)


fps60, tau = 60.0, 0.010
a60 = a_of(1 / fps60, tau)
frames_to = lambda frac, a: math.log(1 - frac) / math.log(a)  # 1 - a^n = frac
# 식 S20이 ODE의 정확한 적분인지 (계단 입력)
dt = 1 / fps60
sub = 2000
yy = 0.0
ts = np.linspace(0, dt * 6, 6 * sub + 1)
y_ode = np.zeros_like(ts)
for j in range(1, len(ts)):
    y_ode[j] = y_ode[j - 1] + (1.0 - y_ode[j - 1]) * (ts[j] - ts[j - 1]) / tau  # 계단 입력 1, 오일러 세분
y_iir = np.array([1 - a60 ** n for n in range(7)])
iir_vs_ode_max_err = float(np.max(np.abs(y_iir - y_ode[::sub])))
put(
    "V7_microbolometer",
    a_60fps_tau10ms=a60,
    claimed_a=0.19,
    frames_to_90pct=frames_to(0.90, a60),
    frames_to_99pct=frames_to(0.99, a60),
    frac_after_3_frames=1 - a60 ** 3,
    claimed="a hot pixel reaches ~90% of its steady response in about three frames",
    tau_needed_for_90pct_in_3_frames_ms=1000 * dt * 3 / math.log(10),
    a_100fps_tau10ms=a_of(0.01, tau),
    frames_to_90pct_at_100fps=frames_to(0.90, a_of(0.01, tau)),
    iir_vs_fine_euler_max_abs_err=iir_vs_ode_max_err,
    note="식 S20은 구간 상수 입력에서 ODE의 정확한 적분이다(세분 오일러와 최대 차이 위 값). a = 0.19는 맞지만 90%에 도달하는 데 약 1.4프레임이고, 3프레임이면 99.3%다",
)

# ---- V8. 열확산도 비 (Sec. 4.3, Fig. 3, S.2.5.2) ---------------------------------
alpha_al, alpha_glass, alpha_steel, alpha_ceramic = 97.0, 0.5, 18.8, 2.0
def diff_len_mm(alpha_mm2_s, t_s):
    return math.sqrt(alpha_mm2_s * t_s)


mats = {  # 대표 물성 (교과서 근사값): k [W/mK], rho [kg/m3], cp [J/kgK]
    "aluminum": (237.0, 2700.0, 897.0),
    "borosilicate_glass": (1.14, 2230.0, 830.0),
    "soda_lime_glass": (1.0, 2500.0, 840.0),
}
alpha_calc = {k: 1e6 * v[0] / (v[1] * v[2]) for k, v in mats.items()}
put(
    "V8_diffusivity",
    claimed_ratio="roughly 500x (aluminum 97 vs glass 0.5 mm2/s)",
    ratio_from_paper_numbers=alpha_al / alpha_glass,
    glass_alpha_needed_for_500x=alpha_al / 500,
    fig3_values_mm2_s={"ceramic": alpha_ceramic, "steel": alpha_steel, "aluminum": alpha_al},
    ratio_Al_over_steel=alpha_al / alpha_steel,
    ratio_Al_over_ceramic=alpha_al / alpha_ceramic,
    diffusion_length_4s_mm={k: diff_len_mm(v, 4.0) for k, v in {"ceramic": alpha_ceramic, "steel": alpha_steel, "aluminum": alpha_al}.items()},
    textbook_alpha_mm2_s=alpha_calc,
    ratio_textbook_Al_over_borosilicate=alpha_calc["aluminum"] / alpha_calc["borosilicate_glass"],
    ratio_textbook_Al_over_sodalime=alpha_calc["aluminum"] / alpha_calc["soda_lime_glass"],
    note="97/0.5 = 194이지 500이 아니다. 교과서 값으로도 약 160-290배. 확산도 순서(알루미늄 > 강철 > 세라믹)는 맞다",
)

# ---- V9. ADS6311 사양 산술 (S.6.1) ------------------------------------------------
bin_ps = 754e-12
dead_ns = 11e-9
put(
    "V9_ads6311",
    pixels_with_3x3_binning=[256 * 3, 192 * 3],
    claimed=[768, 576],
    range_per_time_bin_cm=100 * C * bin_ps / 2,
    dead_time_in_bins=dead_ns / bin_ps,
    dead_time_as_range_m=C * dead_ns / 2,
    n_psf=49,
    note="256x192 와 3x3 비닝은 768x576과 맞음. 1개 시간 구간은 약 11.3 cm, 데드타임 11 ns는 약 14.6 구간(약 1.65 m)",
)

# ---- V10. Suppl. Fig. 4 RMSE 분해와 본문 1.53 m ------------------------------------
rm = {"generic": 0.87, "irf_learned": 0.77, "pileup_modeled": 0.74, "bloom_modeled": 0.54}
steps = {
    "IRF": rm["generic"] - rm["irf_learned"],
    "pileup": rm["irf_learned"] - rm["pileup_modeled"],
    "bloom": rm["pileup_modeled"] - rm["bloom_modeled"],
}
tot = rm["generic"] - rm["bloom_modeled"]
put(
    "V10_digital_twin_rmse",
    suppl_fig4_rmse_m=rm,
    step_reduction_m=steps,
    total_reduction_m=tot,
    total_reduction_pct=100 * tot / rm["generic"],
    step_share_pct={k: 100 * v / tot for k, v in steps.items()},
    main_text_claim_m=1.53,
    ratio_main_over_suppl_total=1.53 / tot,
    note="본문은 RMSE가 약 1.53 m 줄었다고 하나, 보충자료의 같은 실험 오차 지도에서는 0.87 -> 0.54 m(0.33 m)다. 논문은 '더 좁은 자르기' 탓이라 설명하지만 어느 영역인지 수치가 없다",
)

# ---- V11. Fig. 5 보고 수치의 정밀도 -------------------------------------------------
put(
    "V11_fig5_numbers",
    spad_sim=0.221,
    spad_real=0.229,
    spad_abs_diff=0.229 - 0.221,
    spad_rel_diff_pct=100 * (0.229 - 0.221) / 0.229,
    event_diff_pct={"positive": 1.5, "negative": 5.8},
    thermal_mae_pct_of_range=7.6,
    itof_phase_rad=0.13,
    itof_phase_pct_of_2pi=100 * 0.13 / (2 * math.pi),
    n_scenes_per_modality=1,
    note="각 모달리티 1개 장면. 분산, 반복 횟수, 신뢰구간이 없어 유의성이나 구간 계산이 불가능하다",
)

# ---- V12. Table 3 규모 비교 ------------------------------------------------------------
put(
    "V12_table3_scale",
    visionsim_frames=480_000,
    visionsim_time_steps=240_000,
    others_frames={"nuScenes": 1_400_000, "SPICE-HL3": 1_200_000, "TartanAir": 1_000_000, "CroMo": 80_000, "MPD": 103_000, "CARLA-Loc": 66_000, "MSE": 100_000, "ThermVision-DB": 30_000, "FLAT": 2_000},
    note="VisionSIM의 480k는 스테레오 두 시점을 센 값이다(시간 240k). 표 3의 프레임 수는 정의가 데이터셋마다 다를 수 있고 논문은 '모달리티와 주석 폭'으로 비교하며 규모로 우위를 주장하지는 않는다",
)

# ---- V13. 키프레임 배수와 장면 길이 -------------------------------------------------------
put(
    "V13_keyframe_arithmetic",
    anim_frames=600,
    base_fps_if_24s=600 / 24,
    keyframe_multiplier_for_100fps=100 / (600 / 24),
    paper_example_cmd_K=4.0,
    docs_example_K=2.0,
    docs_base_fps=50,
    note="논문 S.7 예시(K=4.0, 100 fps)는 기본 25 fps 장면(600프레임=24 s)과 맞고, 저장소 튜토리얼 예시(K=2.0, 기본 50 fps)는 12 s 장면이다. 장면 fps는 blend 파일 속성이라 모순은 아님",
)

# ---- V14. S.2.7 안의 "기본 프레임율" 세 값 ---------------------------------------------------
put(
    "V14_native_rate_consistency",
    inflection_fps_stated=200,
    native_rate_range_fps_stated=[200, 400],
    native_rate_from_eq_S8_fps=99.8,
    ratio_inflection_over_S8=200 / 99.8,
    speedup_if_render_at_200fps=6400 / 200,
    speedup_if_render_at_400fps=6400 / 400,
    speedup_claimed_at_99_8fps=60,
    note="같은 장면(그림 1B)에 대해 변곡점은 200 fps, 기본 프레임율은 200-400 fps라고 쓰고, 두 문단 뒤에서 식 S8이 99.8 fps를 찾았고 변곡점과 '완벽히 일치'한다고 쓴다. 99.8 fps는 변곡점의 절반이다. 렌더 시간이 fps에 비례하면 속도 향상은 200 fps에서 32배, 400 fps에서 16배이고 60배는 99.8 fps에서만 나온다. 논문 스스로 99.8 fps 아래는 보간 아티팩트가 크다고 읽었다",
)

OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print("wrote", OUT)
for k, v in R.items():
    print(k, {kk: vv for kk, vv in v.items() if kk != "note"})

"""IRASim 논문 수치의 재계산과 검정 (수 초, CPU).

입력: 논문 본문과 표에 적힌 값(아래 상수에 직접 옮김). 그림에서 읽은 값은 별도 표시.
출력: 화면 요약 + analysis/recompute_results.json

실행: python3 analysis/recompute.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
results = {}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


# ---------------------------------------------------------------- 표 1 (짧은 궤적), 표 3 (긴 궤적)
# 행: VDM, LVDM, Video-Ada, Frame-Ada (Frame-Ada는 IRASim 최종형)
T1 = {
    "RT-1":   {"VDM": (13.762, 0.554, 0.4983, 41.23, 371.13), "LVDM": (25.041, 0.815, 0.2244, 4.26, 30.72),
               "Video-Ada": (25.446, 0.823, 0.2191, 4.34, 29.27), "Frame-Ada": (26.048, 0.833, 0.2099, 5.60, 25.58)},
    "Bridge": {"VDM": (18.520, 0.741, 0.3709, 39.82, 127.25), "LVDM": (23.546, 0.810, 0.2155, 10.59, 35.06),
               "Video-Ada": (24.733, 0.827, 0.2021, 10.30, 23.03), "Frame-Ada": (25.275, 0.833, 0.1947, 10.51, 20.91)},
    "Language-Table": {"VDM": (23.067, 0.857, 0.3204, 64.63, 136.56), "LVDM": (28.254, 0.889, 0.1704, 6.85, 24.34),
                       "Video-Ada": (23.893, 0.859, 0.2028, 7.05, 73.84), "Frame-Ada": (28.818, 0.888, 0.1660, 6.38, 48.49)},
}
# 열: PSNR(높을수록 좋음), SSIM(높음), Latent L2(낮음), FID(낮음), FVD(낮음)
METRIC_DIR = {"PSNR": 1, "SSIM": 1, "LatentL2": -1, "FID": -1, "FVD": -1}
names = ["PSNR", "SSIM", "LatentL2", "FID", "FVD"]
t1_check = {}
for ds, rows in T1.items():
    for j, m in enumerate(names):
        best = sorted(rows, key=lambda r: METRIC_DIR[m] * -rows[r][j])[0]
        t1_check[f"{ds}/{m}"] = best
results["table1_best_per_metric"] = t1_check
results["table1_frame_ada_not_best"] = [k for k, v in t1_check.items() if v != "Frame-Ada"]

T3 = {  # 긴 궤적 (Latent L2, PSNR)
    "RT-1": {"LVDM": (0.2567, 23.573), "Video-Ada": (0.2519, 23.984), "Frame-Ada": (0.2408, 24.615)},
    "Bridge": {"LVDM": (0.2534, 21.792), "Video-Ada": (0.2385, 22.868), "Frame-Ada": (0.2306, 23.260)},
    "Language-Table": {"LVDM": (0.1776, 26.215), "Video-Ada": (0.2112, 22.551), "Frame-Ada": (0.1730, 26.773)},
}
results["table3_frame_ada_latentL2_margin_pct_vs_LVDM"] = {
    ds: round(100 * (r["LVDM"][0] - r["Frame-Ada"][0]) / r["LVDM"][0], 1) for ds, r in T3.items()}

# ---------------------------------------------------------------- 표 2 RoboNet
results["table2_robonet_psnr_gain_db"] = round(24.6 - 23.8, 2)

# ---------------------------------------------------------------- 표 4 (정책 평가, LIBERO, 4개 체크포인트, 각 50회)
gt = np.array([0.18, 0.50, 0.80, 1.00])
ir = np.array([0.28, 0.48, 0.74, 0.96])
n_roll = 50
gt_k = np.round(gt * n_roll).astype(int)
ir_k = np.round(ir * n_roll).astype(int)
r, p = stats.pearsonr(gt, ir)
rho, p_s = stats.spearmanr(gt, ir)
z = math.atanh(r)
se = 1 / math.sqrt(len(gt) - 3)
r_lo, r_hi = math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)
results["table4"] = {
    "gt_counts_of_50": gt_k.tolist(), "irasim_counts_of_50": ir_k.tolist(),
    "pearson_r": round(r, 4), "pearson_p": float(f"{p:.3g}"),
    "pearson_fisher_95ci_n4": [round(r_lo, 3), round(r_hi, 3)],
    "spearman_rho": round(rho, 3),
    "mean_abs_diff": round(float(np.mean(np.abs(gt - ir))), 3),
    "max_abs_diff": round(float(np.max(np.abs(gt - ir))), 3),
    "gt_wilson95": [wilson(k, n_roll) for k in gt_k],
    "irasim_wilson95": [wilson(k, n_roll) for k in ir_k],
}

# ---------------------------------------------------------------- 그림 4 (사람 선호, 참가자 5명 x 클립 10개 = 쌍 50개, 부록 H)
# 그림에서 읽은 백분율을 50쌍의 개수로 바꿈 (2%p 단위로 정확히 떨어짐)
pref = {
    ("RT-1", "VDM"): (100, 0, 0), ("Bridge", "VDM"): (100, 0, 0), ("Language-Table", "VDM"): (100, 0, 0),
    ("RT-1", "LVDM"): (60, 12, 28), ("Bridge", "LVDM"): (72, 16, 12), ("Language-Table", "LVDM"): (68, 10, 22),
    ("RT-1", "Video-Ada"): (38, 32, 30), ("Bridge", "Video-Ada"): (36, 30, 34), ("Language-Table", "Video-Ada"): (66, 20, 14),
}
pref_out = {}
for (ds, base), (w, t, l) in pref.items():
    k = (round(w * 0.5), round(t * 0.5), round(l * 0.5))
    assert sum(k) == 50, (ds, base)
    nontie = k[0] + k[2]
    sign_p = stats.binomtest(k[0], nontie, 0.5).pvalue if nontie else float("nan")
    pref_out[f"{ds} vs {base}"] = {
        "win_tie_loss_of_50": k, "win_share_excl_ties_wilson95": wilson(k[0], nontie) if nontie else None,
        "sign_test_p_two_sided": float(f"{sign_p:.3g}"),
    }
results["figure4_preference"] = pref_out

# ---------------------------------------------------------------- 그림 6 (실로봇 계획, 표 6)
# 셀당 시행 수를 표에서 역산: 0.60, 0.73, 0.80, 0.87, 0.20, 0.07, 0.13 이 모두 k/15에 맞는지 확인
table6 = {
    "Close Drawer": {"Random": 0.20, "IRASim (ResNet50)": 0.60, "IRASim (MSE)": 0.87},
    "Place Mandarin on Green Plate": {"Random": 0.07, "IRASim (ResNet50)": 0.73, "IRASim (MSE)": 0.80},
    "Place Mandarin on Red Plate": {"Random": 0.13, "IRASim (ResNet50)": 0.60, "IRASim (MSE)": 0.87},
}
N15 = 15
fit = {}
for task, row in table6.items():
    for m, v in row.items():
        k = round(v * N15)
        fit[f"{task} | {m}"] = {"k_of_15": k, "value_from_k": round(k / N15, 3), "reported": v,
                               "consistent": abs(k / N15 - v) < 0.005}
results["table6_fit_to_15_trials"] = fit
pooled = {m: (sum(round(table6[t][m] * N15) for t in table6), 3 * N15) for m in ["Random", "IRASim (ResNet50)", "IRASim (MSE)"]}
results["table6_pooled_k_of_45"] = pooled


def fisher(a, b):
    ka, na = a
    kb, nb = b
    return float(f"{stats.fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]:.3g}")


results["table6_fisher_p"] = {
    "pooled MSE vs ResNet50": fisher(pooled["IRASim (MSE)"], pooled["IRASim (ResNet50)"]),
    "pooled MSE vs Random": fisher(pooled["IRASim (MSE)"], pooled["Random"]),
    "pooled ResNet50 vs Random": fisher(pooled["IRASim (ResNet50)"], pooled["Random"]),
    "Close Drawer MSE vs ResNet50": fisher((13, 15), (9, 15)),
    "Close Drawer MSE vs Random": fisher((13, 15), (3, 15)),
}
results["table6_pooled_wilson95"] = {m: wilson(k, n) for m, (k, n) in pooled.items()}

# ---------------------------------------------------------------- 표 5 (Push-T, 셀마다 100회 평균 IoU, SD 없음)
K = [1, 5, 10, 50]
push = {
    "P=0": [0.637, 0.679, 0.572, 0.418],
    "P=100": [0.637, 0.847, 0.878, 0.888],
    "P=200": [0.637, 0.866, 0.916, 0.912],
    "P=500": [0.637, 0.907, 0.906, 0.938],
    "P=1000": [0.637, 0.886, 0.945, 0.961],
}
viol = []
for row, vals in push.items():  # 행별: K가 늘수록 단조 증가?
    for a, b, ka, kb in zip(vals[:-1], vals[1:], K[:-1], K[1:]):
        if row != "P=0" and b < a:
            viol.append(f"{row}: K={ka}->{kb} {a:.3f} -> {b:.3f}")
col_viol = []
for j, k in enumerate(K):  # 열별: P가 늘수록 단조 증가? (P=0 제외 X: 전체 행)
    col = [push[r][j] for r in ["P=0", "P=100", "P=200", "P=500", "P=1000"]]
    for a, b, pa, pb in zip(col[:-1], col[1:], ["0", "100", "200", "500"], ["100", "200", "500", "1000"]):
        if b < a:
            col_viol.append(f"K={k}: P={pa}->{pb} {a:.3f} -> {b:.3f}")
results["table5_monotonicity_violations_in_K_for_P_gt_0"] = viol
results["table5_monotonicity_violations_in_P_by_K"] = col_viol
results["table5_P0_K50_minus_K1"] = round(push["P=0"][3] - push["P=0"][0], 3)
results["table5_best_cell"] = "P=1000, K=50, IoU 0.961 (20개 셀 중 최댓값 선택)"
results["table5_gain_from_best_cell"] = round(0.961 - 0.637, 3)

# ---------------------------------------------------------------- 부록 D/E: 모델 크기 (표 11)와 파라미터 산술
# DiT식 블록 한 개: attention(qkv 3d^2 + proj d^2+d), MLP(4d^2+4d + 4d^2+d), adaLN(6d^2+6d)
def dit_params(depth, d, in_ch=4, patch=2, freq=256, state_dim=7, state_mult=4):
    per_block = (3 * d * d + d * d + d) + (d * 4 * d + 4 * d + 4 * d * d + d) + (d * 6 * d + 6 * d)
    t_emb = freq * d + d + d * d + d
    x_emb = in_ch * patch * patch * d + d
    state = (state_dim * state_mult * d + state_mult * d) + (state_mult * d * d + d)
    final = d * 2 * d + 2 * d + d * (patch * patch * in_ch) + patch * patch * in_ch
    return depth * per_block + t_emb + x_emb + state + final


sizes = {"IRASim-S": (12, 384, 33e6), "IRASim-B": (12, 768, 132e6), "IRASim-L": (24, 1024, 461e6),
         "IRASim-XL": (28, 1152, 679e6)}
results["table11_param_check_millions"] = {
    k: {"formula_M": round(dit_params(dep, d) / 1e6, 1), "paper_M": round(p_ / 1e6, 1)}
    for k, (dep, d, p_) in sizes.items()
}

# ---------------------------------------------------------------- 학습 예산 (표 12, 표 10, main.py 설정)
steps_text = 300_000  # 본문, 그림 5, configs/*.yaml의 max_train_steps
steps_table10 = 3_000_000  # 표 10
results["training_steps"] = {"text_and_config": steps_text, "table10": steps_table10}
gpu_hours = {"RT-1": 2381, "Bridge": 2371, "Language-Table": 2369}
results["table12_gpu_hours_per_1k_steps"] = {k: round(v / (steps_text / 1000), 2) for k, v in gpu_hours.items()}
results["table12_wallclock_h_if_32_concurrent"] = {k: round(v / 32, 1) for k, v in gpu_hours.items()}
# 유효 배치: 코드는 local_batch_size(1) x gradient_accumulation_steps x GPU 수
results["effective_batch_by_config"] = {
    "rt1 (accum 2)": {"8 GPU": 16, "32 GPU": 64},
    "languagetable (accum 1)": {"8 GPU": 8, "32 GPU": 32, "64 GPU": 64},
    "paper Appendix E": 64,
}

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("wrote", OUT)

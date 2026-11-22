"""논문 표 값을 옮겨 두고 비율·차이·일관성을 다시 계산한다(수 초, 학습 없음).

표 값 출처: paper/paper.txt (arXiv:2210.02747v2)
  T1 = Table 1 (왼쪽: CIFAR-10, ImageNet 32/64; 오른쪽: ImageNet 128)
  T3 = Table 3 (학습 설정)
  T4 = Table 4 (NLL의 K별 값, dequantization 샘플 수)
  T2 = Table 2 (초해상도)
  v1 = paper/paper_v1.txt (arXiv:2210.02747v1) 의 본문과 표
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).with_name("recompute_results.json")

# T1 왼쪽: (NLL bpd, FID, NFE)
T1 = {
    "CIFAR-10": {
        "DDPM": (3.12, 7.48, 274), "SM": (3.16, 19.94, 242), "ScoreFlow": (3.09, 20.78, 428),
        "FM-Dif": (3.10, 8.06, 183), "FM-OT": (2.99, 6.35, 142)},
    "ImageNet-32": {
        "DDPM": (3.54, 6.99, 262), "SM": (3.56, 5.68, 178), "ScoreFlow": (3.55, 14.14, 195),
        "FM-Dif": (3.54, 6.37, 193), "FM-OT": (3.53, 5.02, 122)},
    "ImageNet-64": {
        "DDPM": (3.32, 17.36, 264), "SM": (3.40, 19.74, 441), "ScoreFlow": (3.36, 24.95, 601),
        "FM-Dif": (3.33, 16.88, 187), "FM-OT": (3.31, 14.45, 138)},
}
# T1 오른쪽: ImageNet 128 (FM-OT만 NLL 있음)
T1_128 = {"FM-OT": (2.90, 20.9), "BigGAN (FID)": 25.3, "PGMGAN (FID)": 21.7}
# T4: K별 NLL (K 마지막 열이 T1과 같은 K로 보임)
T4 = {
    "CIFAR-10": {"K": [1, 20, 50], "DDPM": [3.24, 3.14, 3.12], "SM": [3.28, 3.18, 3.16],
                 "ScoreFlow": [3.21, 3.11, 3.09], "FM-Dif": [3.23, 3.13, 3.10], "FM-OT": [3.11, 3.01, 2.99]},
    "ImageNet-32": {"K": [1, 5, 15], "DDPM": [3.62, 3.57, 3.54], "SM": [3.65, 3.59, 3.57],
                    "ScoreFlow": [3.63, 3.57, 3.55], "FM-Dif": [3.64, 3.58, 3.56], "FM-OT": [3.62, 3.56, 3.53]},
    "ImageNet-64": {"K": [1, 5, 10], "DDPM": [3.36, 3.33, 3.32], "SM": [3.43, 3.41, 3.40],
                    "ScoreFlow": [3.39, 3.37, 3.36], "FM-Dif": [3.37, 3.34, 3.33], "FM-OT": [3.35, 3.33, 3.31]},
}
# T3: 반복 수와 배치로 본 학습 규모 (데이터셋 크기는 공개 값)
T3 = {
    "CIFAR-10": {"epochs": 1000, "iters": 391_000, "batch": 256, "train_size": 50_000},
    "ImageNet-32": {"epochs": 200, "iters": 250_000, "batch": 1024, "train_size": 1_281_167},
    "ImageNet-64": {"epochs": 250, "iters": 157_000, "batch": 2048, "train_size": 1_281_167},
    "ImageNet-128": {"epochs": 571, "iters": 500_000, "batch": 1536, "train_size": 1_281_167},
}
# 본문 DN 비교: ImageNet-128 DN 4.36M it, batch 256 (v2 §6.1)
DN128 = {"iters": 4_360_000, "batch": 256}
# T2 초해상도: PSNR, SSIM, FID (SR3 vs FM-OT)
T2 = {"SR3": {"FID": 5.2, "PSNR": 26.4, "SSIM": 0.762}, "FM-OT": {"FID": 3.4, "PSNR": 24.7, "SSIM": 0.747}}
# v1 표 2 (NFE, 1e-5 tolerance): 열 CIFAR-10, ImageNet-32, ImageNet-64
V1_NFE = {"SM/Dif": (232, 157, 436), "FM/Dif": (179, 165, 179), "FM/OT": (156, 164, 162)}
V1_IN128 = {"NLL": 2.88, "FID": 21.52}

res = {}

# 1. NFE 비율 (OT/Dif)과 NLL·FID 차이 (T1)
res["nfe_ratio_ot_over_fmdif"] = {d: round(T1[d]["FM-OT"][2] / T1[d]["FM-Dif"][2], 3) for d in T1}
res["nfe_reduction_pct_vs_fmdif"] = {d: round(100 * (1 - T1[d]["FM-OT"][2] / T1[d]["FM-Dif"][2]), 1) for d in T1}
res["nfe_ratio_ot_over_ddpm"] = {d: round(T1[d]["FM-OT"][2] / T1[d]["DDPM"][2], 3) for d in T1}
res["fid_rel_change_pct_ot_vs_ddpm"] = {d: round(100 * (T1[d]["FM-OT"][1] / T1[d]["DDPM"][1] - 1), 1) for d in T1}
res["nll_gap_ot_minus_ddpm_bpd"] = {d: round(T1[d]["FM-OT"][0] - T1[d]["DDPM"][0], 3) for d in T1}
res["nll_gap_ot_minus_fmdif_bpd"] = {d: round(T1[d]["FM-OT"][0] - T1[d]["FM-Dif"][0], 3) for d in T1}
# 최고 NLL 모델 확인 (T1 왼쪽, 같은 데이터셋 안에서)
res["best_nll_model"] = {d: min(T1[d], key=lambda m: T1[d][m][0]) for d in T1}
res["best_fid_model"] = {d: min(T1[d], key=lambda m: T1[d][m][1]) for d in T1}
res["best_nfe_model"] = {d: min(T1[d], key=lambda m: T1[d][m][2]) for d in T1}

# 2. 논문 주장: "FM-OT consistently best in all measures"의 확인
res["claim_fmot_best_all_measures"] = {
    d: all(T1[d]["FM-OT"][0] <= T1[d][m][0] and T1[d]["FM-OT"][1] <= T1[d][m][1] and T1[d]["FM-OT"][2] <= T1[d][m][2]
           for m in T1[d]) for d in T1}

# 3. K 민감도 vs 방법 간 차이 (T4)
K_span = {}
for d in T4:
    K_span[d] = {m: round(max(v) - min(v), 3) for m, v in T4[d].items() if m != "K"}
res["nll_K_span_bpd"] = K_span
res["nll_K_span_max_over_methods_bpd"] = {d: max(K_span[d].values()) for d in T4}
# FM-OT vs DDPM 차이가 K 민감도 안에 드는지 (T4 최대 K 값 기준)
res["ImageNet32_fmot_vs_ddpm_at_maxK"] = round(T4["ImageNet-32"]["FM-OT"][-1] - T4["ImageNet-32"]["DDPM"][-1], 3)
res["CIFAR_fmot_vs_ddpm_at_maxK"] = round(T4["CIFAR-10"]["FM-OT"][-1] - T4["CIFAR-10"]["DDPM"][-1], 3)

# 4. T1 과 T4 최대 K 값 일치 점검 (T1 K는 명시되지 않음, 최대 K로 가정)
mismatch = {}
for d in T4:
    for m in T4[d]:
        if m == "K":
            continue
        t1 = T1[d][m][0]
        t4 = T4[d][m][-1]
        if abs(t1 - t4) > 1e-9:
            mismatch[f"{d}/{m}"] = {"T1": t1, "T4_maxK": t4, "diff": round(t1 - t4, 3)}
res["T1_vs_T4_maxK_mismatch"] = mismatch
# CIFAR 3열과 ImageNet 두 열은 모두 T1 값과 같은 K로 보임(CIFAR 50, IN32 15, IN64 10)
res["T1_vs_T4_check_note"] = "T1 NLL과 T4 마지막 열 비교. 불일치는 ImageNet-32 SM, FM-Dif 두 항목"

# 5. T3 반복 수와 epoch 일관성 (iters = epochs * ceil(N / batch))
cons = {}
for d, v in T3.items():
    steps_per_epoch = math.ceil(v["train_size"] / v["batch"])
    implied_iters = v["epochs"] * steps_per_epoch
    cons[d] = {"listed_iters": v["iters"], "implied_iters_from_epochs": implied_iters,
               "ratio_listed_over_implied": round(v["iters"] / implied_iters, 3)}
res["T3_iteration_consistency"] = cons

# 6. 학습 처리량 비교 (ImageNet-128, DN은 batch 256 * 4.36M it)
dn_img = DN128["iters"] * DN128["batch"]
fm_img = T3["ImageNet-128"]["iters"] * T3["ImageNet-128"]["batch"]
res["in128_throughput_fm_over_dn_pct"] = round(100 * fm_img / dn_img, 1)
res["in128_throughput_less_pct"] = round(100 * (1 - fm_img / dn_img), 1)
res["in128_dn_images_e9"] = round(dn_img / 1e9, 3)
res["in128_fm_images_e9"] = round(fm_img / 1e9, 3)

# 7. T2 초해상도: SR3 대비 FM-OT 차이
res["sr_fm_minus_sr3"] = {k: round(T2["FM-OT"][k] - T2["SR3"][k], 3) for k in ["FID", "PSNR", "SSIM"]}

# 8. v1 과 v2 비교 (NFE 표와 ImageNet-128 수치)
v1_ratio = {"IN32_FMOT_over_FMDif": round(V1_NFE["FM/OT"][1] / V1_NFE["FM/Dif"][1], 3),
            "IN64_FMOT_over_FMDif": round(V1_NFE["FM/OT"][2] / V1_NFE["FM/Dif"][2], 3)}
v2_ratio = {"IN32_FMOT_over_FMDif": round(T1["ImageNet-32"]["FM-OT"][2] / T1["ImageNet-32"]["FM-Dif"][2], 3),
            "IN64_FMOT_over_FMDif": round(T1["ImageNet-64"]["FM-OT"][2] / T1["ImageNet-64"]["FM-Dif"][2], 3)}
res["v1_vs_v2_nfe_ratio"] = {"v1": v1_ratio, "v2": v2_ratio}
res["v1_vs_v2_in128"] = {"v1_NLL": V1_IN128["NLL"], "v2_NLL": T1_128["FM-OT"][0],
                         "v1_FID": V1_IN128["FID"], "v2_FID": T1_128["FM-OT"][1],
                         "v1_FM_vs_BigGAN_fid": V1_IN128["FID"] < 25.3, "v2_FM_vs_BigGAN_fid": T1_128["FM-OT"][1] < 25.3}

OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=2, ensure_ascii=False))

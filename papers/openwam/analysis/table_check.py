"""OpenWAM (arXiv:2609.07398v1) 표·본문 수치 재계산.

실행: python3 analysis/table_check.py  → analysis/table_check_results.json
단일 스레드 numpy/scipy, 수 초. 수치는 논문 표·그림에서 옮겼고 출처를 주석으로 적었다.
선택 입력: paper/hf/alpha_pretrain_header.json(사전학습 체크포인트 safetensors 헤더, 가중치 아님)이
있으면 파라미터 수를 다시 세고 analysis/data/param_counts.json을 갱신한다. 없으면 그 파일을 읽는다.
analysis/data/robodojo_per_seed.json은 공식 저장소 assets/robodojo_verification/의 시드별 결과(Apache-2.0)다.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = {}


def r(x, k=3):
    return float(round(float(x), k))


def ztest(p1, n1, p2, n2):
    """두 비율 차이의 z와 양측 p (비대응, 합동 분산). p는 0–1."""
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return {"diff_pp": r(100 * (p1 - p2), 2), "se_pp": r(100 * se, 2), "z": r(z, 2),
            "p": float(f"{2 * stats.norm.sf(abs(z)):.3g}")}


def fisher(a, n1, b, n2):
    return float(f"{stats.fisher_exact([[a, n1 - a], [b, n2 - b]])[1]:.3g}")


# ---------------------------------------------------------------- C1. Table 4 데이터 구성
# (이름, fps, Full 프레임 M, Full 시간, Curated 프레임 M, Curated 시간, Share %) — Table 4
T4 = [
    ("Ego (WorldEngine, 비공개)", 30, 744.9, 6897, 155.7, 1442, 30.1),
    ("AgiBotWorld-Beta", 15, 124.5, 2306, 96.9, 1794, 18.6),
    ("RoboCOIN", 30, 104.5, 956, 74.1, 686, 14.3),
    ("DROID", 10, 46.3, 1285, 36.3, 1007, 7.0),
    ("InternData-A1", 30, 313.7, 2904, 155.5, 1440, 30.0),
]
T4_TOTAL = (1333.9, 14348, 518.5, 6369)
# 공개 사전학습 config(HF OpenWAM-Alpha-Pretrain-Foundation-Model/config.yaml)의 total_hours
CFG_HOURS = {"Ego (WorldEngine, 비공개)": 1439.026796296296, "AgiBotWorld-Beta": 1793.797222222222,
             "RoboCOIN": 686.119268518518, "DROID": 1007.053583333333, "InternData-A1": 1439.026796296296}
rows = []
cur_total = sum(t[4] for t in T4)
for name, fps, fm, fh, cm, ch, share in T4:
    rows.append({
        "source": name, "fps": fps,
        "full_hours_from_frames": r(fm * 1e6 / fps / 3600, 1), "full_hours_table": fh,
        "curated_hours_from_frames": r(cm * 1e6 / fps / 3600, 1), "curated_hours_table": ch,
        "config_total_hours": r(CFG_HOURS[name], 2),
        "config_frames_M": r(CFG_HOURS[name] * 3600 * fps / 1e6, 2),
        "share_from_frames_pct": r(100 * cm / cur_total, 2), "share_table_pct": share,
    })
real = [t for t in T4 if t[0] in ("AgiBotWorld-Beta", "RoboCOIN", "DROID")]
real_frames = sum(t[4] for t in real)
OUT["C1_table4"] = {
    "rows": rows,
    "totals_recomputed": [r(sum(t[2] for t in T4), 1), sum(t[3] for t in T4), r(cur_total, 1), sum(t[5] for t in T4)],
    "totals_table": T4_TOTAL,
    "frame_split_pct": {"ego": r(100 * 155.7 / cur_total, 2), "real_robot": r(100 * real_frames / cur_total, 2),
                        "synthetic": r(100 * 155.5 / cur_total, 2)},
    # 실제 로봇 40%를 정제 프레임 비율로 나눴다는 서술(5.2.1) 확인
    "real_robot_split_within_40pct": {t[0]: r(100 * t[4] / real_frames, 2) for t in real},
    "config_total_hours_sum": r(sum(CFG_HOURS.values()), 1),
    "note": "Ego와 InternData-A1 config 시간이 1439.0268로 완전히 같다(둘 다 30% 프레임 목표 = 같은 프레임 수, 같은 30 fps).",
}

# ---------------------------------------------------------------- C2. Table 9 연산량
ITERS, BATCH, GPUS, DAYS = 155_862, 3072, 128, 7
clips = ITERS * BATCH
gpu_h = GPUS * DAYS * 24
sec_per_step = DAYS * 86400 / ITERS
# 토큰 수 가정: 33프레임·stride 4 → 9 RGB 프레임 → Wan2.2-VAE(시간 4배, 공간 16배) → 3 latent 프레임,
# (1,2,2) patch → 384/32 × 320/32 = 12 × 10 = 120 토큰/latent 프레임. 행동 32 토큰(33프레임 창의 32 step).
vid_tok = 3 * (384 // 32) * (320 // 32)
act_tok = 32
flops_per_clip = 6 * (5.0e9 * vid_tok + 1.0e9 * act_tok)  # 학습 가능한 DiT·ActionDiT 행렬곱만, 주의 연산·umT5·VAE 제외
total_flops = flops_per_clip * clips
h200_bf16_dense = 989e12
OUT["C2_table9_compute"] = {
    "clips_seen": clips, "clips_seen_M": r(clips / 1e6, 1), "curated_frames_M": 518.5,
    "clips_per_frame": r(clips / 518.5e6, 3),
    "released_checkpoint_step": 154_000, "released_fraction_of_epoch": r(154_000 / ITERS, 4),
    "gpu_hours": gpu_h, "sec_per_step": r(sec_per_step, 2), "clips_per_sec_per_gpu": r(BATCH / sec_per_step / GPUS, 2),
    "assumed_tokens": {"video": vid_tok, "action": act_tok},
    "dit_matmul_flops_total": float(f"{total_flops:.3g}"),
    "mfu_dit_only_pct": r(100 * total_flops / (gpu_h * 3600 * h200_bf16_dense), 1),
    "note": ("창 stride 1이고 에피소드 끝은 마지막 프레임 반복으로 채우므로(robotwin.py 등) 창 수 ≈ 프레임 수다. "
             "478.8M 클립은 518.5M 프레임보다 7.7% 적다. 1 epoch의 정의(창 수 계산)가 논문에 없다. "
             "MFU는 주의 연산·umT5·VAE 인코딩을 빼고 DiT·ActionDiT 행렬곱만 센 하한 추정이다."),
}

# ---------------------------------------------------------------- C3. 파라미터 (safetensors 헤더)
hdr_path = ROOT / "paper" / "hf" / "alpha_pretrain_header.json"
pc_path = HERE / "data" / "param_counts.json"
if hdr_path.exists():
    hdr = json.loads(hdr_path.read_text())
    hdr.pop("__metadata__", None)
    groups, dtypes = Counter(), Counter()
    for k, v in hdr.items():
        n = math.prod(v["shape"])
        g = ".".join(k.split(".")[:2]) if k.startswith("video_backbone") else k.split(".")[0]
        groups[g] += n
        dtypes[v["dtype"]] += n
    pc = {"tensors": len(hdr), "dtype": dict(dtypes), "groups": dict(groups)}
    pc_path.write_text(json.dumps(pc, indent=1))
else:
    pc = json.loads(pc_path.read_text())
G = pc["groups"]
total = sum(G.values())
frozen = G["video_backbone.text_encoder"] + G["video_backbone.vae"]
OUT["C3_params"] = {
    "total_B": r(total / 1e9, 3), "bytes_bf16_GB": r(total * 2 / 1e9, 2), "file_size_GB": r(24_813_767_464 / 1e9, 2),
    "by_group_B": {k: r(v / 1e9, 4) for k, v in sorted(G.items(), key=lambda x: -x[1])},
    "trainable_B": r((total - frozen) / 1e9, 3), "frozen_B": r(frozen / 1e9, 3),
    "inference_without_text_encoder_GB_bf16": r((total - G["video_backbone.text_encoder"]) * 2 / 1e9, 2),
    "note": "논문 '5B DiT + 1B ActionDiT'와 맞는다. 모든 SFT 체크포인트도 같은 24.81 GB로, 고정된 umT5(5.68B)와 VAE까지 들어 있다.",
}

# ---------------------------------------------------------------- C4. HF 체크포인트 step ↔ Table 10
T10 = {"LIBERO": 10690, "VLABench": 6000, "RoboTwin-Full": 118655, "RoboTwin-Clean2Random": 10740,
       "RoboDojo": 60000, "RoboCasa365": 60000, "EBench": 100000, "RoboCasa-GR1": 100000,
       "Real-Single-Arm-Franka": 9860, "Real-RoboDojo (3 embodiment)": 30000, "Real-Dexterous-Hand-Wuji": 10925}
HF_STEPS = {"LIBERO": 10690, "VLABench": 6000, "RoboTwin-Full": 118655, "RoboTwin-Clean2Random": 10740,
            "RoboDojo": 60000, "RoboCasa365": 60000, "EBench": 100000, "RoboCasa-GR1": 100000,
            "Real-Single-Arm-Franka": 9860, "Real-RoboDojo (3 embodiment)": 30000, "Real-Dexterous-Hand-Wuji": 10925}
T10_BATCH = {"LIBERO": 256, "VLABench": 196, "RoboTwin-Full": 256, "RoboTwin-Clean2Random": 256, "RoboDojo": 256,
             "RoboCasa365": 1024, "EBench": 256, "RoboCasa-GR1": 256, "Real-Single-Arm-Franka": 256,
             "Real-RoboDojo (3 embodiment)": 256, "Real-Dexterous-Hand-Wuji": 256}
OUT["C4_sft_checkpoints"] = {
    "all_match_table10": all(T10[k] == HF_STEPS[k] for k in T10),
    "sft_clips_M": {k: r(T10[k] * T10_BATCH[k] / 1e6, 2) for k in T10},
    "robotwin_full_sft_vs_pretrain": r(T10["RoboTwin-Full"] * 256 / clips, 4),
    "pretrain_step_hf_vs_table9": [154_000, ITERS],
}

# ---------------------------------------------------------------- C5. Section 4 표·그림 내부 일관성
T1 = {"Single Vanilla": (85.20, 85.80, 85.50), "Single MoE": (86.22, 83.04, 84.63),
      "Dual JSA": (92.34, 92.38, 92.36), "Dual JCA": (87.86, 88.64, 88.25),
      "Dual Detached CA": (92.06, 91.64, 91.85), "Dual IDM": (87.76, 88.14, 87.95), "Tri JSA": (92.84, 92.36, 92.60)}
T2 = {"Isolated": (88.08, 86.74, 87.41), "Video Sees Action": (87.92, 87.34, 87.63),
      "Action Sees Video": (92.98, 91.80, 92.39), "Mutual": (92.50, 91.80, 92.15)}
FIG6 = {"Wan2.1-VACE-1.3B": (1.3, 90.14), "Cosmos-Predict2.5-2B": (2.0, 91.64),
        "Wan2.2-TI2V-5B": (5.0, 92.39), "Wan2.1-I2V-14B": (14.0, 93.79)}
FIG7 = {"DINOv3 w/o SVAE": 76.42, "V-JEPA 2.1 w/o SVAE": 80.91, "FLUX.2 VAE": 84.26,
        "V-JEPA 2.1 w/ SVAE": 88.46, "DINOv3 w/ SVAE": 90.18, "Wan2.2 VAE (DiT 무작위 초기화)": 90.30}
# Fig. 9 (RoboTwin2.0-Full 평균, mutual, 독립 noise 학습)
FIG9 = {"sync": 93.0,
        "VS action α=4": 91.0, "VS action α=8": 90.7, "VS action α=16": 89.7, "VS action α=32": 88.5,
        "VS video α=4": 91.8, "VS video α=8": 91.1, "VS video α=16": 91.9, "VS video α=32": 91.5,
        "LO action o=0.2": 92.2, "LO action o=0.4": 92.0, "LO action o=0.6": 92.3, "LO action o=0.8": 92.3,
        "LO video o=0.2": 91.7, "LO video o=0.4": 91.1, "LO video o=0.6": 91.5, "LO video o=0.8": 91.2}
# Fig. 10 (RoboTwin2.0-Clean2Random; ID = Clean, OOD = Randomized)
FIG10 = {"From Scratch": (87.00, 14.50), "Robot Only": (88.50, 23.80),
         "Ego→Robot (2단계)": (87.10, 26.50), "Ego+Robot (co-train)": (87.68, 26.62)}
# Fig. 11: Action Sees Video 절대값과 Mutual − ASV 차이
FIG11 = {"From Scratch · Full": (92.39, -0.24), "Pretrained · Full": (92.34, 0.16),
         "Pretrained · C2R ID": (86.98, 0.70), "Pretrained · C2R OOD": (25.70, 0.72)}

avg_err = {k: r((c + d) / 2 - a, 3) for k, (c, d, a) in {**T1, **T2}.items()}
logp = np.log([v[0] for v in FIG6.values()])
sr6 = np.array([v[1] for v in FIG6.values()])
slope = np.polyfit(logp, sr6, 1)[0]
OUT["C5_section4_consistency"] = {
    "table1_2_average_minus_mean": avg_err,
    "fig6_slope_pp_per_e_fold_params": r(slope, 3), "fig6_14B_minus_5B": r(93.79 - 92.39, 2),
    "fig6_vs_table2_ASV": [92.39, T2["Action Sees Video"][2]],
    "fig7_pretrained_init_gain_pp": r(92.39 - 90.30, 2),
    "fig11_mutual_values": {k: r(a + d, 2) for k, (a, d) in FIG11.items()},
    "fig11_vs_fig10_cotrain": {"ID": [r(86.98 + 0.70, 2), 87.68], "OOD": [r(25.70 + 0.72, 2), 26.62]},
    "fig11_from_scratch_vs_table2": [r(92.39 - 0.24, 2), T2["Mutual"][2]],
    "same_nominal_config_spread": {"Table1 Dual JSA": 92.36, "Table2 Mutual": 92.15, "Table2 ASV": 92.39,
                                   "Fig9 sync (mutual, 독립 noise)": 93.0},
    "note": ("Fig. 11의 pretrained C2R OOD Mutual = 25.70 + 0.72 = 26.42로, 같은 설정이어야 할 Fig. 10 co-train OOD 26.62와 0.20 다르다. "
             "ID는 87.68로 일치한다. Table 1의 Dual JSA 마스크는 명시되지 않았다."),
}

# ---------------------------------------------------------------- C6. 유의성 (RoboTwin: 과제 50 × 100 episode, seed 0)
N_COND = 50 * 100       # Clean 또는 Randomized 하나
N_AVG = 2 * N_COND      # 두 조건 평균
sig = {
    "T1 Dual JSA vs Single Vanilla (avg)": ztest(.9236, N_AVG, .8550, N_AVG),
    "T1 Tri JSA vs Dual JSA (avg)": ztest(.9260, N_AVG, .9236, N_AVG),
    "T1 Dual JSA vs Detached CA (avg)": ztest(.9236, N_AVG, .9185, N_AVG),
    "T2 ASV vs Isolated (avg)": ztest(.9239, N_AVG, .8741, N_AVG),
    "T2 ASV vs Mutual (avg)": ztest(.9239, N_AVG, .9215, N_AVG),
    "Fig6 14B vs 5B (avg)": ztest(.9379, N_AVG, .9239, N_AVG),
    "Fig7 Wan VAE scratch vs DINOv3 SVAE (avg)": ztest(.9030, N_AVG, .9018, N_AVG),
    "Fig7 pretrained 5B vs scratch 5B (avg)": ztest(.9239, N_AVG, .9030, N_AVG),
    "Fig9 sync vs best async LO action o=0.6 (avg)": ztest(.930, N_AVG, .923, N_AVG),
    "Fig9 sync vs VS action α=32 (avg)": ztest(.930, N_AVG, .885, N_AVG),
    "Fig9 VS video α=16 vs VS action α=4 (avg)": ztest(.919, N_AVG, .910, N_AVG),
    "Fig10 ID Robot Only vs Scratch": ztest(.8850, N_COND, .8700, N_COND),
    "Fig10 ID co-train vs Scratch": ztest(.8768, N_COND, .8700, N_COND),
    "Fig10 OOD Robot Only vs Scratch": ztest(.2380, N_COND, .1450, N_COND),
    "Fig10 OOD co-train vs Robot Only": ztest(.2662, N_COND, .2380, N_COND),
    "Fig10 OOD co-train vs 2단계": ztest(.2662, N_COND, .2650, N_COND),
    "Fig11 Pretrained Full Mutual-ASV": ztest(.9250, N_AVG, .9234, N_AVG),
    "Fig11 C2R ID Mutual-ASV": ztest(.8768, N_COND, .8698, N_COND),
    "Fig11 C2R OOD Mutual-ASV": ztest(.2642, N_COND, .2570, N_COND),
    "T16 ABot-M0.5 vs OpenWAM (Full avg)": ztest(.9410, N_AVG, .9360, N_AVG),
}
OUT["C6_significance_robotwin"] = {
    "assumption": ("공식 RoboTwin README: 과제당 100 episode, seed 0, 50과제. 조건 하나 N = 5,000, 평균 N = 10,000. "
                   "이항 SE는 과제 고정 시 episode 표본 오차의 상한이다. 학습 seed 분산(실행 1회)은 포함되지 않는다."),
    "tests": sig,
    "se_pp_at_92pct": {"one_condition": r(100 * math.sqrt(.92 * .08 / N_COND), 2),
                       "average": r(100 * math.sqrt(.92 * .08 / N_AVG), 2)},
}

# ---------------------------------------------------------------- C7. 부록 D 시뮬레이션 표
# Table 13 LIBERO (Spatial, Object, Goal, Long, Avg)
T13 = {"OpenVLA": (84.7, 88.4, 79.2, 53.7, 76.5), "π0": (98.0, 96.8, 94.4, 88.4, 94.4),
       "StarVLA": (97.8, 98.6, 96.2, 93.8, 96.6), "π0.5": (98.8, 98.2, 98.0, 92.4, 96.9),
       "GR00T-N1.6": (97.7, 98.5, 97.5, 94.4, 97.0), "OpenVLA-OFT": (97.6, 98.4, 97.9, 94.5, 97.1),
       "X-VLA": (98.2, 98.6, 97.8, 97.6, 98.1), "ABot-M0": (98.8, 99.8, 99.0, 96.6, 98.6),
       "Being-H0.5": (99.2, 99.6, 99.4, 97.4, 98.9), "Fast-WAM": (98.2, 100.0, 97.0, 95.2, 97.6),
       "Motus": (96.8, 99.8, 96.6, 97.6, 97.7), "ImageWAM": (97.2, 99.2, 98.8, 98.4, 98.4),
       "LingBot-VA": (98.5, 99.6, 97.2, 98.5, 98.5), "ABot-M0.5": (100.0, 99.8, 99.4, 98.4, 99.4),
       "OpenWAM-α": (99.6, 99.6, 99.8, 98.2, 99.3)}
T13_AVG_ONLY = {"Qwen-RobotManip": 99.2, "DiT4DiT": 98.6, "Being-H0.7": 99.2}
lib = {k: r(np.mean(v[:4]) - v[4], 3) for k, v in T13.items()}
lib_all = {**{k: v[4] for k, v in T13.items()}, **T13_AVG_ONLY}
# Table 5 LIBERO-Plus (Camera, Robot, Language, Light, Background, Noise, Layout, Avg)
T5 = {"π0": (13.8, 6.0, 58.8, 85.0, 81.4, 79.0, 68.9, 53.6), "OpenVLA-OFT": (56.4, 31.9, 79.5, 88.7, 93.3, 75.8, 74.2, 69.6),
      "StarVLA": (52.5, 49.8, 88.5, 95.7, 95.7, 73.0, 76.9, 74.1), "ABot-M0": (60.4, 67.9, 86.4, 96.2, 91.6, 86.4, 82.6, 80.5),
      "π0.5": (78.4, 73.6, 80.8, 96.2, 94.1, 89.0, 84.5, 84.4), "ACoT-VLA": (72.6, 82.6, 87.5, 97.7, 96.5, 87.8, 88.1, 86.6),
      "Qwen-RobotManip": (87.2, 75.5, 85.6, 96.6, 97.7, 97.7, 87.3, 89.0), "Fast-WAM": (16.4, 44.5, 68.9, 78.2, 53.7, 37.7, 60.7, 51.5),
      "Being-H0.7": (82.0, 59.0, 82.8, 97.8, 90.0, 93.5, 88.5, 82.1), "Cosmos-Policy": (75.8, 63.3, 81.7, 96.5, 88.9, 92.7, 82.2, 82.2),
      "ImageWAM": (80.8, 50.3, 91.4, 98.1, 85.5, 93.8, 80.5, 83.1), "ABot-M0.5": (70.5, 87.4, 88.6, 94.0, 89.7, 75.5, 85.2, 83.4),
      "OpenWAM-α": (33.8, 76.1, 88.0, 97.0, 87.1, 39.8, 77.5, 69.2)}
X5 = np.array([v[:7] for v in T5.values()])
y5 = np.array([v[7] for v in T5.values()])
# 합 1, 비음수 제약 최소제곱
cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
w5 = optimize.minimize(lambda w: ((X5 @ w - y5) ** 2).sum(), np.full(7, 1 / 7), bounds=[(0, 1)] * 7,
                       constraints=cons, method="SLSQP").x
TW_W = np.array([0.157, 0.1583, 0.1655, 0.1143, 0.113, 0.1613, 0.1306])  # thinkworld/ 분석의 과제 수 가중치
# Table 14 VLABench SR·PS·IS (5범주 + Avg)
T14 = {"π0": ([47.0, 21.2, 29.1, 17.3, 32.2], [62.7, 33.6, 43.0, 38.7, 42.5], [67.8, 44.0, 54.9, 58.0, 50.6], (29.4, 44.1, 55.0)),
       "π0.5": ([65.4, 38.2, 43.9, 48.2, 44.9], [77.8, 49.7, 57.3, 64.2, 62.3], [80.4, 52.0, 60.0, 67.0, 65.0], (48.1, 62.3, 64.9)),
       "ERVLA": ([69.7, 47.0, 44.0, 58.0, 47.4], [81.1, 61.0, 55.0, 70.2, 62.3], [84.2, 66.4, 57.2, 73.8, 70.6], (53.2, 65.9, 70.4)),
       "Xiaomi-Robotics-1": ([75.6, 53.0, 48.4, 55.8, 62.6], [85.0, 66.6, 58.3, 66.8, 74.9], [79.8, 66.4, 58.2, 70.2, 74.8], (59.1, 70.3, 69.9)),
       "Bridge-WA": ([78.0, 23.0, 51.1, 67.0, 45.0], [85.8, 28.8, 64.4, 80.3, 60.3], [85.0, 39.0, 74.2, 82.0, 76.0], (52.8, 64.0, 71.2)),
       "OpenWAM-α": ([83.4, 38.1, 58.0, 53.8, 61.4], [87.9, 45.9, 64.8, 64.5, 72.9], [75.2, 45.9, 57.9, 66.5, 71.6], (58.9, 67.2, 63.5))}
vla = {k: {m: [r(np.mean(v[i]), 2), v[3][i]] for i, m in enumerate(("SR", "PS", "IS"))} for k, v in T14.items()}
is_rank = sorted(T14, key=lambda k: -T14[k][3][2]).index("OpenWAM-α") + 1
# Table 15/16 RoboTwin
T15 = {"StarVLA": (46.5, 3.2, 24.9), "GR00T-N1.7": (43.6, 20.7, 32.2), "X-VLA": (68.0, 20.9, 44.5),
       "Spatial Forcing": (77.2, 26.7, 52.0), "ABot-M0": (70.7, 36.0, 53.4), "π0.5": (70.7, 46.0, 58.4),
       "GigaBrain-0.7": (66.8, 67.9, 67.4), "Qwen-RobotManip": (84.7, 69.4, 77.1), "AHA-WAM": (64.3, 3.2, 33.8),
       "Fast-WAM": (77.8, 1.9, 39.9), "X-WAM": (70.0, 25.8, 47.9), "4D-WAM": (81.5, 41.8, 61.7), "OpenWAM-α": (89.4, 48.7, 69.0)}
T16 = {"X-VLA": (72.80, 72.84, 72.82), "π0.5": (82.70, 76.80, 79.75), "ABot-M0": (86.06, 85.08, 85.57),
       "Qwen-VLA": (86.10, 87.20, 86.65), "StarVLA": (88.18, 88.32, 88.25), "Galaxea G0.5": (93.70, 92.80, 93.25),
       "Qwen-RobotManip": (93.70, 94.00, 93.85), "Motus": (88.66, 87.02, 87.84), "Fast-WAM": (91.90, 91.80, 91.85),
       "LingBot-VA": (92.93, 91.55, 92.24), "ImageWAM": (93.20, 93.56, 93.38), "LingBot-VA 2.0": (93.80, 93.40, 93.60),
       "ABot-M0.5": (94.00, 94.20, 94.10), "OpenWAM-α": (93.74, 93.46, 93.60)}
# Table 17 RoboDojo SR: (Gen-Std, Gen-Rand, Precision, Long-Horizon, Memory, Open, Avg)
T17 = {"StarVLA-α": (5.00, 0.00, 4.33, 6.50, 2.44, 0.58, 3.24), "X-VLA": (12.00, 1.00, 12.00, 9.75, 3.56, 0.50, 6.52),
       "π0.5": (15.00, 1.00, 5.50, 14.67, 4.56, 1.67, 6.91), "Spatial Forcing": (15.00, 4.00, 10.58, 14.58, 4.11, 1.58, 8.04),
       "Hy-Embodied-0.5-VLA": (17.00, 0.00, 8.00, 14.92, 12.11, 0.58, 8.80),
       "Xiaomi-Robotics-1": (28.00, 6.00, 18.83, 23.67, 6.56, 3.58, 13.93),
       "Galaxea G0.5": (20.00, 6.00, 20.42, 32.25, 7.33, 1.58, 14.88), "DM0.5": (18.00, 4.00, 16.75, 19.50, 47.44, 2.08, 19.34),
       "Fast-WAM": (2.00, 0.00, 0.00, 5.17, 3.44, 0.42, 2.03), "AHA-WAM": (6.00, 0.00, 2.42, 2.67, 2.78, 0.83, 2.39),
       "GigaWorld-Policy": (6.00, 0.00, 1.83, 8.92, 2.22, 0.50, 3.27), "X-WAM": (5.00, 1.00, 1.83, 9.08, 4.67, 0.25, 3.83),
       "OpenWAM-α": (25.56, 4.11, 9.25, 25.33, 9.11, 1.08, 11.92)}
dojo = {k: {"avg_5dim_merged_gen": r(((v[0] + v[1]) / 2 + sum(v[2:6])) / 5, 3), "avg_6col": r(np.mean(v[:6]), 3), "table": v[6]}
        for k, v in T17.items()}
# Table 18 EBench (TableTop SR, Score, PnP SR, Score, LH SR, Score, Overall SR, Score); 과제 수 7/10/9 (공식 ebench README)
T18 = {"π0": (15.7, 30.0, 35.0, 39.0, 17.0, 41.0, 23.6, 37.0), "X-VLA": (8.6, 24.0, 50.0, 54.0, 6.2, 25.0, 23.7, 36.0),
       "InternVLA-A1": (4.3, 11.0, 43.0, 47.0, 17.9, 46.0, 23.9, 36.0), "π0.5": (12.9, 32.0, 45.0, 50.0, 18.1, 39.0, 27.1, 41.0),
       "Qwen-RobotManip": (50.0, 70.0, 56.5, 60.0, 29.9, 55.0, 45.6, 60.0), "OpenWAM-α": (30.0, 44.2, 67.5, 72.0, 44.3, 72.6, 49.4, 64.7)}
wE = np.array([7, 10, 9]) / 26
eb = {k: {"SR": [r(wE @ np.array(v[0:6:2]), 2), v[6]], "Score": [r(wE @ np.array(v[1:6:2]), 2), v[7]]} for k, v in T18.items()}
# Table 19 RoboCasa365 (Atomic, Comp-Seen, Comp-Unseen, Avg)
T19 = {"Diffusion Policy": (15.7, 0.2, 1.3, 6.1), "π0": (36.3, 5.2, 0.7, 15.0), "π0.5": (39.6, 7.1, 1.2, 16.9),
       "GR00T-N1.5": (50.7, 14.8, 2.7, 23.9), "Qwen-RobotManip": (68.6, 20.1, 14.9, 35.9), "RLDX-1": (67.6, 27.9, 8.5, 36.0),
       "Xiaomi-Robotics-1": (80.2, 57.1, 32.1, 57.4), "GigaWorld-Policy": (44.4, 11.8, 2.9, 20.7),
       "ABot-M0.5": (75.9, 38.3, 2.7, 40.4), "OpenWAM-α": (69.7, 32.1, 8.9, 38.2)}
X19 = np.array([v[:3] for v in T19.values()])
y19 = np.array([v[3] for v in T19.values()])
w19 = optimize.minimize(lambda w: ((X19 @ w - y19) ** 2).sum(), np.full(3, 1 / 3), bounds=[(0, 1)] * 3,
                        constraints=cons, method="SLSQP").x
w19_task = np.array([18, 16, 16]) / 50
T20 = {"π0": 13.6, "π0.5": 37.0, "GR00T-N1.5": 48.0, "StarVLA": 48.8, "GR00T-N1.6": 49.9, "VP-VLA": 53.8,
       "Being-H0.5": 53.9, "Qwen-VLA-Instruct": 56.7, "RLDX-1": 58.7, "PhysBrain 1.0": 64.5, "UWM": 20.0,
       "Being-H0.7": 49.2, "DiT4DiT": 50.8, "LDA-1B": 55.4, "OpenWAM-α": 60.5}


def rank_of(d, key="OpenWAM-α"):
    order = sorted(d, key=lambda k: -d[k])
    return {"rank": order.index(key) + 1, "of": len(order), "best": order[0], "best_value": d[order[0]],
            "gap_to_best": r(d[key] - d[order[0]], 2)}


OUT["C7_appendixD"] = {
    "LIBERO_avg_minus_mean": lib, "LIBERO_rank": rank_of(lib_all),
    "LIBERO_Plus_weight_fit": {"weights": [r(x, 4) for x in w5], "rmse": r(np.sqrt(np.mean((X5 @ w5 - y5) ** 2)), 3),
                               "simple_mean_rmse": r(np.sqrt(np.mean((X5.mean(1) - y5) ** 2)), 3),
                               "thinkworld_weights_rmse": r(np.sqrt(np.mean((X5 @ TW_W - y5) ** 2)), 3),
                               "openwam_simple_mean": r(X5[-1].mean(), 2), "openwam_weighted": r(X5[-1] @ TW_W, 2)},
    "LIBERO_Plus_rank": rank_of({k: v[7] for k, v in T5.items()}),
    "VLABench_avg_recomputed_vs_table": vla, "VLABench_SR_rank": rank_of({k: v[3][0] for k, v in T14.items()}),
    "VLABench_IS_rank_openwam": is_rank,
    "RoboTwin_C2R_avg_check": {k: r((v[0] + v[1]) / 2 - v[2], 2) for k, v in T15.items()},
    "RoboTwin_C2R_rank": rank_of({k: v[2] for k, v in T15.items()}),
    "RoboTwin_C2R_best_other_WAM": "4D-WAM 61.7",
    "RoboTwin_Full_avg_check": {k: r((v[0] + v[1]) / 2 - v[2], 3) for k, v in T16.items()},
    "RoboTwin_Full_rank": rank_of({k: v[2] for k, v in T16.items()}),
    "RoboDojo_avg": dojo, "RoboDojo_rank": rank_of({k: v[6] for k, v in T17.items()}),
    "EBench_family_weighted_vs_table": eb, "EBench_rank": rank_of({k: v[6] for k, v in T18.items()}),
    "RoboCasa365_weight_fit": {"weights": [r(x, 4) for x in w19], "task_weights_18_16_16": list(w19_task),
                               "rmse_fit": r(np.sqrt(np.mean((X19 @ w19 - y19) ** 2)), 3),
                               "rmse_task_weights": r(np.sqrt(np.mean((X19 @ w19_task - y19) ** 2)), 3),
                               "simple_mean_rmse": r(np.sqrt(np.mean((X19.mean(1) - y19) ** 2)), 3)},
    "RoboCasa365_rank": rank_of({k: v[3] for k, v in T19.items()}),
    "RoboCasa_GR1_rank": rank_of(T20),
    "top_tier_claim_gaps_pp": {"LIBERO": r(99.3 - 99.4, 1), "VLABench SR": r(58.9 - 59.1, 1),
                               "RoboTwin-Full": r(93.60 - 94.10, 2), "RoboCasa365": r(38.2 - 57.4, 1),
                               "RoboCasa-GR1": r(60.5 - 64.5, 1)},
}

# RoboDojo 시드별 (공식 저장소 assets)
DIMS = {
    "precision": ["fasten_screws", "plug_in_charger", "insert_tubes", "pour_balls_into_vase", "play_Xylophone",
                  "deposit_coin", "insert_key", "build_tower"],
    "long_horizon": ["put_bottles_into_dustbin", "fill_pen_holder", "classify_objects", "play_tic_tac_toe",
                     "fill_egg_holder", "organize_table", "make_kong", "play_stacking_toy"],
    "memory": ["cover_blocks", "match_and_pick_from_conveyor", "swap_blocks", "swap_T", "press_by_number",
               "imitate_sorting_sequence"],
    "open": ["align_blocks", "general_pickup", "stack_blocks_by_language", "solve_equation",
             "classify_objects_by_language", "pick_from_conveyor_by_image", "store_tools_in_toolbox", "pour_by_language"],
}
dj = json.loads((HERE / "data" / "robodojo_per_seed.json").read_text())
gen_tasks = list(dj["generalization"].keys())
per_seed = {m: {} for m in ("sr", "score")}
for m in ("sr", "score"):
    gstd = np.array([dj["generalization"][t]["standard"][m] for t in gen_tasks]).mean(0)
    grnd = np.array([dj["generalization"][t]["random"][m] for t in gen_tasks]).mean(0)
    gm = np.array([dj["merged_all_tasks"][t][m] for t in gen_tasks]).mean(0)
    per_seed[m]["generalization_standard"] = gstd
    per_seed[m]["generalization_random"] = grnd
    per_seed[m]["generalization_merged"] = gm
    for d, ts in DIMS.items():
        per_seed[m][d] = np.array([dj["merged_all_tasks"][t][m] for t in ts]).mean(0)
    per_seed[m]["average"] = np.mean([per_seed[m][k] for k in ("generalization_merged", *DIMS)], axis=0)
dojo_seed = {}
for k in per_seed["sr"]:
    dojo_seed[k] = {"sr_mean": r(per_seed["sr"][k].mean(), 2), "sr_seed_sd": r(per_seed["sr"][k].std(ddof=1), 2),
                    "sr_seeds": [r(x, 2) for x in per_seed["sr"][k]],
                    "score_mean": r(per_seed["score"][k].mean(), 2), "published": dj["published_aggregates"][k]}
zero_tasks = [t for t, v in dj["merged_all_tasks"].items() if max(v["sr"]) == 0]
OUT["C8_robodojo_seeds"] = {"dims": dojo_seed, "n_tasks": len(dj["merged_all_tasks"]),
                            "tasks_zero_sr_all_seeds": len(zero_tasks), "episodes_per_task_per_seed": 50,
                            "gap_to_DM0.5_in_seed_sd": r((19.34 - dojo_seed["average"]["sr_mean"]) / dojo_seed["average"]["sr_seed_sd"], 1)}

# ---------------------------------------------------------------- C9. 실제 로봇
T6 = {"π0.5": [13, 7, 14, 15, 7, 10], "LingBot-VA": [16, 15, 17, 17, 13, 15], "OpenWAM-α": [17, 12, 20, 20, 13, 17]}
t6 = {k: {"sum": sum(v), "of": 120} for k, v in T6.items()}
# Table 7: 과제별 (Score, SR%) — ARX X5 / Piper / Piper X 각 6과제, 과제당 10회
T7 = {
    "X-VLA": [[(0, 0), (0, 0), (0, 0), (2, 0), (20.7, 10), (18, 0)], [(0, 0), (5.3, 0), (0, 0), (0, 0), (53, 50), (37, 0)],
              [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0.7, 0)]],
    "Xiaomi-Robotics-0": [[(24, 20), (0, 0), (3, 0), (4, 0), (40, 20), (19, 10)], [(0, 0), (0, 0), (0, 0), (0, 0), (23, 20), (22.7, 0)],
                          [(0, 0), (0.7, 0), (4, 0), (0, 0), (0, 0), (2.7, 0)]],
    "GalaxeaVLA (G0)": [[(1, 0), (0, 0), (3, 0), (6, 0), (0, 0), (10.3, 0)], [(0, 0), (32.7, 10), (13.3, 10), (0, 0), (56, 50), (30, 10)],
                        [(0, 0), (0, 0), (0, 0), (4, 0), (0, 0), (6, 0)]],
    "InternVLA-A1": [[(0, 0), (0, 0), (0, 0), (0, 0), (48, 20), (12, 0)], [(0, 0), (7.3, 0), (0, 0), (0, 0), (73, 70), (59, 40)],
                     [(0, 0), (0, 0), (4, 0), (2, 0), (0, 0), (10, 0)]],
    "π0.5": [[(24.6, 20), (1.8, 0), (25.8, 10), (47, 20), (40, 20), (26.8, 10)], [(10, 10), (28, 0), (10, 10), (0, 0), (72, 60), (72, 50)],
             [(0, 0), (14.8, 10), (0, 0), (29.5, 10), (7.5, 0), (3, 0)]],
    "OpenWAM-α": [[(31, 0), (0, 0), (13, 0), (52, 20), (100, 100), (38, 20)], [(8.3, 0), (60, 40), (36.7, 30), (0, 0), (100, 100), (75, 50)],
                  [(0, 0), (28, 10), (18, 0), (52.3, 20), (40, 40), (24.7, 10)]],
}
T7_TABLE = {"X-VLA": (7.6, 3.3), "Xiaomi-Robotics-0": (7.9, 3.9), "GalaxeaVLA (G0)": (9.0, 4.4), "InternVLA-A1": (12.0, 7.2),
            "π0.5": (22.9, 12.8), "OpenWAM-α": (37.6, 24.4)}
t7 = {}
for k, embs in T7.items():
    emb = [(np.mean([c[0] for c in e]), np.mean([c[1] for c in e])) for e in embs]
    t7[k] = {"emb_avg": [[r(a, 1), r(b, 1)] for a, b in emb],
             "overall": [r(np.mean([e[0] for e in emb]), 2), r(np.mean([e[1] for e in emb]), 2)], "table": T7_TABLE[k],
             "sr_successes_of_180": int(round(sum(c[1] for e in embs for c in e) / 10))}
best_score_tasks = sum(
    1 for ei in range(3) for ti in range(6)
    if T7["OpenWAM-α"][ei][ti][0] >= max(T7[k][ei][ti][0] for k in T7) and T7["OpenWAM-α"][ei][ti][0] > 0)
tie_or_win_sr = sum(1 for ei in range(3) for ti in range(6)
                    if T7["OpenWAM-α"][ei][ti][1] >= max(T7[k][ei][ti][1] for k in T7))
strict_win_sr = sum(1 for ei in range(3) for ti in range(6)
                    if T7["OpenWAM-α"][ei][ti][1] > max(T7[k][ei][ti][1] for k in T7 if k != "OpenWAM-α"))
# Table 8: (ID SR, n, OOD SR, n, ID score num, den, OOD score num, den)
T8 = {"Stack Toy Tower": {"π0.5": (1, 10, 1, 10, 16, 30, 12, 30), "OpenWAM-α": (4, 10, 3, 10, 21, 30, 17, 30)},
      "Collect Shuttlecocks": {"π0.5": (2, 10, 3, 15, 11, 35, 17, 54), "OpenWAM-α": (6, 10, 4, 15, 29, 35, 30, 57)},
      "Put Away Clothes": {"π0.5": (6, 10, 9, 20, 26, 30, 45, 60), "OpenWAM-α": (10, 10, 17, 20, 30, 30, 55, 60)},
      "Twist off Bottle Cap": {"π0.5": (3, 10, 6, 20, 8, 10, 13, 20), "OpenWAM-α": (7, 10, 16, 20, 10, 10, 18, 20)}}
t8 = {}
for task, d in T8.items():
    a, b = d["OpenWAM-α"], d["π0.5"]
    t8[task] = {"ID_fisher_p": fisher(a[0], a[1], b[0], b[1]), "OOD_fisher_p": fisher(a[2], a[3], b[2], b[3]),
                "OOD_score_denominators": [b[7], a[7]]}
pool = {m: [sum(T8[t][m][i] for t in T8) for i in range(4)] for m in ("π0.5", "OpenWAM-α")}
OUT["C9_real_robot"] = {
    "T6_sums": t6, "T6_fisher_vs_LingBot": fisher(99, 120, 93, 120), "T6_fisher_vs_pi05": fisher(99, 120, 66, 120),
    "T6_task_wins_vs_LingBot": sum(o > l for o, l in zip(T6["OpenWAM-α"], T6["LingBot-VA"])),
    "T6_task_ties_vs_LingBot": sum(o == l for o, l in zip(T6["OpenWAM-α"], T6["LingBot-VA"])),
    "T7": t7, "T7_openwam_best_score_tasks_of_18": best_score_tasks,
    "T7_openwam_best_or_tied_sr_of_18": tie_or_win_sr, "T7_openwam_strict_best_sr_of_18": strict_win_sr,
    "T7_fisher_sr_vs_pi05": fisher(t7["OpenWAM-α"]["sr_successes_of_180"], 180, t7["π0.5"]["sr_successes_of_180"], 180),
    "T8_per_task": t8,
    "T8_pooled": {"ID": [f"{pool['OpenWAM-α'][0]}/{pool['OpenWAM-α'][1]}", f"{pool['π0.5'][0]}/{pool['π0.5'][1]}",
                         fisher(pool['OpenWAM-α'][0], pool['OpenWAM-α'][1], pool['π0.5'][0], pool['π0.5'][1])],
                  "OOD": [f"{pool['OpenWAM-α'][2]}/{pool['OpenWAM-α'][3]}", f"{pool['π0.5'][2]}/{pool['π0.5'][3]}",
                          fisher(pool['OpenWAM-α'][2], pool['OpenWAM-α'][3], pool['π0.5'][2], pool['π0.5'][3])]},
}

# ---------------------------------------------------------------- C10. 본문 수치
OUT["C10_text_claims"] = {
    "17_physical_platforms": 1 + 15 + 1,
    "21_robot_embodiments": 1 + 15 + 1 + 4,
    "frames_518M_hours_6400": [518.5, 6369],
    "raw_1.33B_14300h": [1333.9, 14348],
    "speedup_fig4_RTX5090_ms": {"Vanilla": 149, "MoE": 156, "Dual JSA": 175, "Dual JCA": 180, "IDM": 251, "Tri JSA": 216},
    "alpha_170ms_vs_dual_jsa_175ms": True,
}

(HERE / "table_check_results.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in OUT.items() if k in ("C2_table9_compute", "C3_params")}, ensure_ascii=False, indent=1)[:3000])
print("wrote", HERE / "table_check_results.json")

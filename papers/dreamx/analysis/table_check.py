"""DreamX(arXiv:2611.05377v1) 표·본문 수치 재계산 (CPU, 단일 스레드). 실행: python3 analysis/table_check.py

- C1: Table 1 평균(회귀 4열 평균, 분류 3열 평균)과 최강 기준선 대비 개선 폭
- C2: Table 2 상대 개선
- C3: Table 3·5 subgoal 성공률 정의 재계산, reach/push 분리 Fisher 검정, 지연 비
- C4: Table 6(사람 영상만) vs Table 5(로봇 시연) Upper 비교
- C5: Table 7 CALVIN Avg. Len. = Σ(≥k 성공률)
- C6: 학습 데이터 시간 합(Fig. 9)
- C7: 압축률 "1,200 → 4 tokens, 300×"를 토큰·실수 개수로 다시 셈
- C8: Table 8 섭동 수치 정합성
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 3), round(c + h, 3)]


# Table 1 (LARYBench): (CALVIN, VLABench, RoboCOIN, AgiBot, Avg) / (Atomic robot, Composite human, Composite robot, Avg)
T1 = {
    "LAPA": ((0.96, 0.95, 0.96, 1.00, 0.97), (22.27, 14.61, 23.64, 20.17)),
    "UniVLA": ((0.82, 0.74, 0.94, 0.97, 0.87), (18.62, 19.08, 18.56, 18.75)),
    "villa-X": ((0.86, 0.72, 0.94, 0.97, 0.87), (15.00, 17.80, 29.90, 20.90)),
    "LAPA-MAGVIT2": ((0.59, 0.36, 0.80, 0.83, 0.65), (33.12, 59.70, 29.53, 40.78)),
    "LAPA-SigLIP2": ((0.57, 0.30, 0.86, 0.88, 0.65), (46.83, 54.74, 29.44, 43.67)),
    "LAPA-DINOv2": ((0.55, 0.26, 0.85, 0.86, 0.63), (58.04, 55.86, 34.19, 49.36)),
    "LAPA-DINOv3": ((0.50, 0.25, 0.82, 0.84, 0.60), (56.27, 64.19, 27.04, 49.17)),
    "MoTo": ((0.43, 0.13, 0.54, 0.68, 0.44), (41.57, 55.11, 56.34, 51.01)),
    "AdaWorld": ((0.64, 0.26, 0.94, 0.94, 0.69), (31.58, 49.8, 51.48, 44.29)),
    "VideoWorld2": ((0.72, 0.41, 0.82, 0.85, 0.70), (14.50, 30.24, 37.36, 27.37)),
    "DiLA": ((0.43, 0.09, 0.51, 0.52, 0.38), (58.51, 65.47, 62.76, 62.25)),
    "DreamX": ((0.15, 0.05, 0.26, 0.22, 0.17), (68.03, 72.72, 68.26, 69.67)),
    "DreamX+FAE-32": ((0.23, 0.07, 0.31, 0.26, 0.22), (65.90, 69.49, 66.06, 67.15)),
}
REG_COLS = ("CALVIN", "VLABench", "RoboCOIN", "AgiBotWorld-Beta")
CLS_COLS = ("Atomic robot", "Composite human", "Composite robot")


def main() -> None:
    out: dict = {}

    # C1 Table 1 평균과 개선 폭
    c1 = {}
    for m, (reg, cls) in T1.items():
        ra = sum(reg[:4]) / 4
        ca = sum(cls[:3]) / 3
        # 각 칸이 소수 둘째 자리 반올림이므로 실제 평균은 ra ± 0.005. 표 값(±0.005)과 구간이 겹치면 정합
        reg_exact = abs(ra - reg[4]) <= 0.005 + 1e-9
        reg_possible = abs(ra - reg[4]) <= 0.010
        c1[m] = {"reg_avg_recomputed": round(ra, 4), "reg_avg_table": reg[4], "reg_ok": reg_exact,
                 "reg_within_rounding": reg_possible,
                 "cls_avg_recomputed": round(ca, 2), "cls_avg_table": cls[3], "cls_ok": abs(ca - cls[3]) <= 0.006}
    base = [m for m in T1 if not m.startswith("DreamX")]
    best_reg = {c: min(T1[m][0][i] for m in base) for i, c in enumerate(REG_COLS)}
    best_cls = {c: max(T1[m][1][i] for m in base) for i, c in enumerate(CLS_COLS)}
    gains = {}
    for name in ("DreamX", "DreamX+FAE-32"):
        reg, cls = T1[name]
        gains[name] = {
            "reg_rel_drop_vs_best_baseline": {c: round(1 - reg[i] / best_reg[c], 3) for i, c in enumerate(REG_COLS)},
            "cls_gain_pp_vs_best_baseline": {c: round(cls[i] - best_cls[c], 2) for i, c in enumerate(CLS_COLS)},
        }
    out["C1_table1"] = {"rows": c1, "all_ok": all(v["reg_ok"] and v["cls_ok"] for v in c1.values()),
                        "boundary_cases": [m for m, v in c1.items() if not v["reg_ok"]],
                        "best_baseline_per_column_reg": best_reg, "best_baseline_per_column_cls": best_cls,
                        "gains": gains,
                        "note": "LARYBench 탐침(probe)은 벤치마크마다 따로 학습. 표현이 embodiment 사이에 정렬되었는지는 재지 않음(03 §2)"}

    # C2 Table 2 (H&R zero-shot, 1,008 clips)
    t2 = {"AdaWorld": (170.19, 0.344, 22.95, 0.846), "VideoWorld2": (241.92, 0.299, 22.53, 0.881), "DreamX": (57.44, 0.179, 26.52, 0.917)}
    out["C2_table2"] = {
        "FVD_ratio_best_baseline_over_dreamx": round(170.19 / 57.44, 2),
        "LPIPS_rel_drop_vs_best": round(1 - 0.179 / 0.299, 3),
        "PSNR_gain_dB_vs_best": round(26.52 - 22.95, 2),
        "SSIM_gain_vs_best": round(0.917 - 0.881, 3),
        "n_clips": 1008, "frames_per_clip_evaluated": 19,
        "note": "시드·신뢰구간 없음. 기준선의 학습 데이터가 DreamX(사람 469 h 포함)와 다름",
    }

    # C3 Table 3·5 실물 서랍 닫기 (설정당 10회)
    reach = {"Mimic-Video": (80, 100, 100), "DreamX": (100, 100, 100)}
    push = {"Mimic-Video": (20, 0, 0), "DreamX": (80, 100, 100)}
    sr = {m: [round((reach[m][i] + push[m][i]) / 2, 1) for i in range(3)] for m in reach}
    avg = {m: round(sum(v) / 3, 2) for m, v in sr.items()}
    n = 10
    pk = {m: sum(round(p / 100 * n) for p in push[m]) for m in push}
    rk = {m: sum(round(p / 100 * n) for p in reach[m]) for m in reach}
    _, p_push = stats.fisher_exact([[pk["DreamX"], 30 - pk["DreamX"]], [pk["Mimic-Video"], 30 - pk["Mimic-Video"]]])
    _, p_reach = stats.fisher_exact([[rk["DreamX"], 30 - rk["DreamX"]], [rk["Mimic-Video"], 30 - rk["Mimic-Video"]]])
    _, p_upper_push = stats.fisher_exact([[8, 2], [2, 8]])
    out["C3_real_policy"] = {
        "sr_per_config_recomputed": sr, "avg_recomputed": avg, "table3": {"Mimic-Video": 50.0, "DreamX": 96.7},
        "push_successes_of_30": pk, "reach_successes_of_30": rk,
        "push_wilson95": {m: wilson(pk[m], 30) for m in pk},
        "fisher_p_push_pooled": float(f"{p_push:.2e}"), "fisher_p_reach_pooled": round(p_reach, 3),
        "fisher_p_upper_push_8v2": round(p_upper_push, 4),
        "latency_ratio": round(28226 / 1474, 2),
        "note": "차이는 전부 push에서 나온다. Mimic-Video는 거의 항상 서랍에 닿지만 거의 밀지 못함. 학습 시연 45개, 설정당 10회, 체크포인트·시드 1개",
    }

    # C4 Table 6(Upper를 사람 영상만으로) vs Table 5(로봇 시연) Upper
    _, p6 = stats.fisher_exact([[10, 0], [8, 2]])
    out["C4_human_only_upper"] = {"human_only_push": "10/10", "robot_demo_push": "8/10", "fisher_p": round(p6, 3),
                                  "note": "'사람 영상만으로도 100%'는 표본 10회 안에서 로봇 시연(8/10)과 구분되지 않음. 사람 영상 개수는 보고 안 됨"}

    # C5 Table 7 CALVIN
    t7 = {"w/o latent": (83.4, 61.7, 42.5, 29.8, 20.4, 2.378), "DreamX policy": (85.7, 67.0, 49.5, 36.4, 25.7, 2.643)}
    c5 = {m: {"avg_len_recomputed": round(sum(v[:5]) / 100, 3), "table": v[5]} for m, v in t7.items()}
    # 평가 시퀀스 수는 논문에 없음. CALVIN 관례 1,000개를 가정하면 ≥1 성공 857 vs 834
    _, p_k1 = stats.fisher_exact([[857, 143], [834, 166]])
    _, p_k5 = stats.fisher_exact([[257, 743], [204, 796]])
    out["C5_table7_calvin"] = {"avg_len": c5, "gain": round(2.643 - 2.378, 3),
                               "fisher_p_if_1000_sequences": {"k>=1": round(p_k1, 3), "k>=5": round(p_k5, 4)},
                               "note": "평가 시퀀스 수 미보고. 1,000개(관례)라고 가정한 검정. Avg. Len. 2.64는 CALVIN ABC→D 최신 정책(4 이상)보다 낮은 수준이라 '보조 신호로서의 효과'만 보여 줌"}

    # C6 학습 데이터 (Fig. 9)
    hours = {"EgoVerse": 59.0, "EgoDex": 410.1, "Bridge": 47.9, "DROID": 350.5, "MolmoAct2": 502.3, "AgiBot": 818.2}
    tot = sum(hours.values())
    human = hours["EgoVerse"] + hours["EgoDex"]
    out["C6_data_hours"] = {"total_h": round(tot, 1), "human_h": round(human, 1), "human_share": round(human / tot, 3),
                            "robot_share": round(1 - human / tot, 3), "agibot_share": round(hours["AgiBot"] / tot, 3),
                            "note": "AgiBot(37.4%)이 LARYBench 회귀 벤치마크 AgiBotWorld-Beta와 같은 출처. 학습/평가 분리 여부는 논문에 없음"}

    # C7 압축률: Cosmos-Predict2.5 2B, 480×640, VAE 8× 공간·4× 시간 + 2×2 patch (VAE 사양은 사전 지식: Wan2.1 VAE, 16채널)
    h, w = 480, 640
    vae_s, patch, vae_ch, D = 8, 2, 16, 2048
    tokens_per_latent_frame = (h // vae_s // patch) * (w // vae_s // patch)
    q_tokens = 4
    out["C7_compression"] = {
        "tokens_per_latent_frame": tokens_per_latent_frame,
        "token_ratio": tokens_per_latent_frame / q_tokens,
        "floats_backbone_tokens_per_frame": tokens_per_latent_frame * D,
        "floats_latent_action_per_frame": q_tokens * D,
        "floats_vae_latent_per_frame": (h // vae_s) * (w // vae_s) * vae_ch,
        "ratio_vs_vae_latent": round((h // vae_s) * (w // vae_s) * vae_ch / (q_tokens * D), 1),
        "floats_fae32_per_frame": q_tokens * 32,
        "ratio_fae32_vs_vae_latent": round((h // vae_s) * (w // vae_s) * vae_ch / (q_tokens * 32), 1),
        "latent_frames_for_29_frames": 1 + (29 - 1) // 4,
        "max_query_tokens": q_tokens * (1 + (29 - 1) // 4),
        "assumption": "VAE 8× 공간·4× 시간·16채널, patch 2는 Cosmos-Predict2.5(Wan2.1 VAE) 사양을 사전 지식으로 둔 것. 1,200 토큰과 2–8 latent 프레임은 논문 수치와 일치",
        "note": "300×는 토큰 개수 비. 각 latent-action 토큰은 백본 폭(2,048)을 그대로 가져 프레임당 8,192개 실수. VAE latent(76,800) 대비로는 9.4×이고, 진짜 '작은 코드'는 FAE-32(128개)",
    }

    # C8 Table 8 섭동
    t8 = {"0.1": (1.000, 0.141, 0.0054, 0.0030), "0.3": (0.995, 0.211, 0.0077, 0.0034), "1.0": (0.946, 0.481, 0.0215, 0.0077), "zero": (0.456, 0.866, 0.0758, 0.0227)}
    out["C8_table8"] = {
        "static_over_whole_rgb": {k: round(v[3] / v[2], 2) for k, v in t8.items()},
        "monotone_flow_dev": all(t8[a][1] < t8[b][1] for a, b in (("0.1", "0.3"), ("0.3", "1.0"), ("1.0", "zero"))),
        "n": "선별한 고움직임 클립 8개 × decoder seed 2",
        "note": "α=1(채널 표준편차 크기 잡음)에서도 움직임 크기는 0.946로 거의 유지 → 디코더가 latent의 세부에 둔감. 0 query에서도 움직임이 46% 남음 → 첫 프레임과 생성 prior만으로도 상당한 움직임이 나옴",
    }

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:280])
    print("wrote", path)


if __name__ == "__main__":
    main()

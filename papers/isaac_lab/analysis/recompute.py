"""Isaac Lab 기술 보고서(arXiv:2511.04831 v1)의 수치를 다시 계산하고 그림·본문 일관성을 검사한다.

논문에는 표가 없고 처리량은 로그축 그림으로만 나온다. 그래서 그림 값은 PDF를 300 dpi로
렌더링한 뒤 눈으로 읽은 근사치다. 로그축 한 decade가 약 203 px이므로 읽기 오차는 대략 ±3%다.
이 값으로 계산하는 것은 비율과 지수뿐이며, 절대값 주장은 본문에 적힌 수치와만 비교한다.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json (수 초, numpy)
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"

# ---- 그림 13: 단일 상태 기반 학습 처리량 (FPS, 단위 10^6). 축은 로그, decade = 203 px ----
# 판독 방법: y(px) → 10^(-1 + (300 - y)/203) 로 환산 (0.10 눈금 y=300, 1.00 눈금 y=97)
N_ENVS = [2048, 4096, 8192, 16384]
FIG13A = {  # Dexsuite Lift (DextrAH 계열 들기 과제)
    "L40 1-GPU": [0.0332, 0.0489, 0.0657, 0.0805],
    "GeForce 5090 1-GPU": [0.0501, 0.0721, 0.108, 0.152],
    "RTX PRO 6000 1-GPU": [0.0381, 0.0593, 0.0843, 0.121],
    "RTX PRO 6000 2-GPU": [0.0721, 0.116, 0.159, 0.248],
    "RTX PRO 6000 4-GPU": [0.141, 0.232, 0.322, 0.473],
    "RTX PRO 6000 8-GPU": [0.277, 0.447, 0.636, 0.925],
}
FIG13B = {  # Open Drawer (Franka 서랍 열기). 같은 y축 눈금을 가정한다(그림 B에는 눈금 숫자가 없음)
    "L40 1-GPU": [0.0712, 0.107, 0.142, 0.163],
    "GeForce 5090 1-GPU": [0.130, 0.202, 0.281, 0.365],
    "RTX PRO 6000 1-GPU": [0.0628, 0.107, 0.159, 0.214],
    "RTX PRO 6000 2-GPU": [0.124, 0.212, 0.317, 0.429],
    "RTX PRO 6000 4-GPU": [0.242, 0.418, 0.606, 0.805],
    "RTX PRO 6000 8-GPU": [0.479, 0.824, 1.21, 1.61],
}

# ---- 그림 18a: Warp RayCaster 자산 수에 따른 처리량 (두 GPU는 서로 다른 y축) ----
ASSETS = [1, 2, 4, 8, 16, 32]
FIG18A_BLACKWELL_K_FPS = [45.0, 43.8, 42.6, 40.1, 36.8, 30.3]   # 왼쪽 축, 단위 10^3 FPS
FIG18A_ADA_M_RAYS = [73.2, 72.1, 71.3, 69.5, 66.8, 61.9]         # 오른쪽 축, 단위 10^6 rays/s

# ---- 그림 18c: 메시 면 수에 따른 처리량 (Blackwell, 단위 10^3 FPS). x축 최댓값은 약 20,000 면 ----
FIG18C_FACES_MAX_READ = 20000
FIG18C_BLACKWELL_K_FPS = [19.1, 19.1, 19.1, 19.0, 18.9]  # 면 수 약 100 ~ 20k 다섯 점
FIG18C_ADA_K_FPS = [10.6, 10.6, 10.6, 10.6, 10.6]


def digitize_check():
    """판독 공식의 역산 검사: 눈금 두 개(0.10, 1.00)가 맞는지 확인한다."""
    y_at_tick = {0.10: 300.0, 1.00: 97.0}
    out = {}
    for val, y in y_at_tick.items():
        back = 10 ** (-1 + (300.0 - y) / 203.0)
        out[str(val)] = round(back, 4)
    return out


def scaling_block(table, label):
    """같은 환경 수에서 N-GPU 처리량을 N × 1-GPU로 나눈 효율을 계산한다."""
    one = np.array(table["RTX PRO 6000 1-GPU"])
    res = {}
    for n in (2, 4, 8):
        many = np.array(table[f"RTX PRO 6000 {n}-GPU"])
        eff = many / (n * one)
        res[f"eff_{n}gpu_per_env"] = {str(e): round(float(x), 3) for e, x in zip(N_ENVS, eff)}
    allv = [v for key in (f"eff_{n}gpu_per_env" for n in (2, 4, 8)) for v in res[key].values()]
    res["eff_range_all_gpu_counts_and_envs"] = [round(min(allv), 3), round(max(allv), 3)]
    res["label"] = label
    return res


def main():
    results = {"method": "그림 13, 18의 로그축 판독 (300 dpi PDF, 눈금 2개로 보정, 근사 ±3%)"}

    # V1. 본문 수치와 그림 판독 비교
    results["V1_text_vs_figure"] = {
        "dextrah_8gpu_16384_text": 900000,
        "dextrah_8gpu_16384_read": round(FIG13A["RTX PRO 6000 8-GPU"][3] * 1e6),
        "franka_8gpu_16384_text": 1600000,
        "franka_8gpu_16384_read": round(FIG13B["RTX PRO 6000 8-GPU"][3] * 1e6),
        "verdict": "두 값 모두 본문 하한(900k, 1.6M)과 읽기 오차 안에서 맞음. 1.6M은 경계 근처라 ±3% 밖에서는 판정 불가",
    }

    # V2. 다중 GPU 스케일링 효율 (그림 13)
    results["V2_scaling_efficiency"] = {
        "dexsuite_lift": scaling_block(FIG13A, "Dexsuite Lift"),
        "open_drawer": scaling_block(FIG13B, "Open Drawer"),
        "reading_error_note": "2-GPU의 1.00 근처 값은 판독 오차 안이다. 8-GPU 효율은 Dexsuite 0.91–0.96, Open Drawer 0.94–0.95로 '거의 선형'과 맞는다. 환경 수를 GPU당으로 보든 전체로 보든 같은 격자에서 비교하면 결론이 같다.",
    }

    # V3. "5090은 DextrAH에서 약 25% 빠르다" 문장의 기준선
    a16 = {k: v[3] for k, v in FIG13A.items()}
    b16 = {k: v[3] for k, v in FIG13B.items()}
    results["V3_5090_claim_baseline"] = {
        "dexsuite_5090_vs_rtx1": round(a16["GeForce 5090 1-GPU"] / a16["RTX PRO 6000 1-GPU"], 3),
        "dexsuite_5090_vs_rtx2": round(a16["GeForce 5090 1-GPU"] / a16["RTX PRO 6000 2-GPU"], 3),
        "open_drawer_5090_vs_rtx2": round(b16["GeForce 5090 1-GPU"] / b16["RTX PRO 6000 2-GPU"], 3),
        "verdict": "단일 RTX PRO 6000 대비 약 +26%로 본문 '약 25%'와 맞음. 2-GPU 대비는 0.61로 본문 문장의 기준으로 맞지 않음. 기준선이 본문에 없음",
    }

    # V4. 논문 식 (1)의 FPS와 3.53% 주장
    results["V4_direct_vs_manager_3_53pct"] = {
        "reported": 3.53,
        "recomputable": False,
        "reason": "점별 값, 반복 횟수, 평균 방식(환경 수 평균인지 전체 평균인지)이 없다. 그림 16 좌측 패널은 로그축 눈금 숫자가 100000 하나뿐이라 점 값 복원도 불가하다. 두 곡선은 전 구간에서 거의 겹친다(정성 확인).",
    }

    # V5. 그림 18a의 자산 수 지수 (처리량 ∝ N^b)
    n = np.array(ASSETS, dtype=float)
    blk = np.array(FIG18A_BLACKWELL_K_FPS)
    ada = np.array(FIG18A_ADA_M_RAYS)
    b_blk = float(np.polyfit(np.log(n), np.log(blk), 1)[0])
    b_ada = float(np.polyfit(np.log(n), np.log(ada), 1)[0])
    results["V5_asset_scaling_exponent"] = {
        "blackwell_exponent": round(b_blk, 3),
        "ada_exponent": round(b_ada, 3),
        "blackwell_drop_pct_1_to_32": round(100 * (blk[-1] / blk[0] - 1), 1),
        "ada_drop_pct_1_to_32": round(100 * (ada[-1] / ada[0] - 1), 1),
        "verdict": "자산 32배에서 처리량은 약 33%(Blackwell)와 15%(Ada) 줄었다. 본문 '준선형 감소(sub-linear)'와 맞다. 두 GPU의 y축과 단위가 달라 절대 비교는 안 된다",
    }

    # V6. 그림 18c의 면 수 범위 (본문: 20k–200k)
    blk_c = np.array(FIG18C_BLACKWELL_K_FPS)
    results["V6_mesh_range"] = {
        "text_claim_faces": "20,000–200,000",
        "figure_x_axis_max_read": FIG18C_FACES_MAX_READ,
        "blackwell_change_pct_over_range": round(100 * (blk_c[-1] / blk_c[0] - 1), 2),
        "verdict": "그림 x축은 약 20k에서 끝난다. 200k 점은 없다. 캡션과 본문의 '20–200k'는 그림과 맞지 않는다. 면 수 영향은 작다(약 -1%)는 본문과 맞다",
    }

    # V7. 통계 정보 (반복, 분산, 검정)
    results["V7_statistics_available"] = {
        "repetitions_reported": False,
        "error_band_definition_in_caption": False,
        "confidence_interval_reported": False,
        "significance_test_reported": False,
        "note": "그림 13, 14, 16은 선 둘레에 음영 띠가 있지만 캡션에 무엇인지(표준편차, 신뢰구간, 시드 범위) 없다. 따라서 정량 CI와 검정은 계산할 수 없다",
    }

    # V8. 보고된 sim-to-real 수치 (모두 인용)
    results["V8_sim_to_real_numbers"] = {
        "factory_zero_shot_success_pct": [83, 99],
        "factory_source": "Tang et al. 2023 (IndustReal), 본문 6.4절",
        "automate_success_pct": 80,
        "automate_source": "Tang et al. 2024 (AutoMate), 본문 6.4절 (약 80%)",
        "spot_speed_mps": 5.2,
        "spot_source": "Miller et al. 2025 (RAI), 본문 6.1절 (자체 실험 아님)",
        "own_sim_to_real_experiment": False,
        "note": "이 보고서의 자체 sim-to-real 실험은 없다. 인용 수치는 원 논문에서 확인해야 한다",
    }

    # V9. 처리량 정의 (식 1, 식 2)와 env 수 해석
    results["V9_metric_definitions"] = {
        "eq1": "FPS = #environment steps / (simulation time + learning time)",
        "eq2": "FPS = #rendering steps / simulation time (센서 절)",
        "rays_per_s": "Warp RayCaster 절 (그림 18)은 rays/s를 쓴다",
        "code_fps_fields_rl_games": ["Environment only FPS", "Environment + Inference FPS", "Environment + Inference + Policy update FPS"],
        "env_count_per_gpu_convention": "benchmark_non_rl.py는 effective FPS = fps × num_envs × world_size를 계산한다. 환경 수는 프로세스(GPU)당 값으로 보인다. 논문은 이를 명시하지 않는다",
    }

    # V10. 그림 16 캡션의 좌/우 설명
    results["V10_fig16_caption"] = {
        "caption_says": "Right: single GPU setups. Left: multi-GPU setups",
        "figure_shows": "좌측 패널은 L40, 5090, RTX PRO 6000 (각 단일 GPU) 비교, 가운데 패널은 1·2·4·8 GPU 비교",
        "verdict": "캡션의 좌/우 설명이 그림 배치와 맞지 않는다",
    }

    results["digitize_check_ticks"] = digitize_check()
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print(json.dumps({k: results[k] for k in ("V1_text_vs_figure", "V3_5090_claim_baseline")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Table I·본문·프로젝트 페이지 수치 재계산 (CPU). 실행: python3 analysis/table_check.py

- C1–C6: 본문 주장(약 100배, 44%p, 약 2배, 40%p 이상)을 Table I 값으로 다시 계산
- C7: 50개 기준 성공 개수가 정수인지(표 내부 일관성)
- C8: 50개 표본의 Wilson 95% 신뢰구간, 최강 기준선과의 Fisher 정확검정(대응 정보가 없어 비대응 근사)
- C9: 성공 1건당 GPU-h
- C10: 처리량(env step/s), replay buffer 메모리(코드 문서의 1.76 GB/백만 행)
- C11: 규모 확장(프로젝트 페이지: 1,000개 중 600개 이상, 60 GPU-h)과 50개 벤치마크의 효율 비교
"""
import json
import math
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
N = 50  # 50-motion benchmark (Sec. IV-A b)

# Table I: (SR_SPIDER, SR_MT-Obj, SR_MT, MOPE mm, MORE rad, GPU-h)
TABLE = {
    "xhand": {
        "Do as I Do": (0.36, 0.04, 0.00, 47.32, 0.28, 66),
        "DexMachina": (0.48, 0.32, 0.12, 37.49, 0.29, 447),
        "CHORD": (0.46, 0.02, 0.00, 55.70, 0.16, 2847),
        "FlashDexRetarget": (0.90, 0.86, 0.86, 10.95, 0.25, 29),
    },
    "sharpa": {
        "Do as I Do": (0.64, 0.10, 0.10, 58.33, 0.24, 62),
        "CHORD": (0.50, 0.22, 0.00, 60.07, 0.19, 3314),
        "FlashDexRetarget": (0.86, 0.80, 0.80, 16.45, 0.32, 44),
    },
}
METRICS = ("SR_SPIDER", "SR_MT-Obj", "SR_MT")
ENV_STEPS = {"xhand": 300e6, "sharpa": 400e6}  # Sec. IV-B
FDR = "FlashDexRetarget"


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(c - h, 3), round(c + h, 3)


def main() -> None:
    out: dict = {}

    # C1–C6 본문 주장
    x, s = TABLE["xhand"], TABLE["sharpa"]
    out["C1_compute_ratio_chord_over_fdr"] = {
        "xhand": round(x["CHORD"][5] / x[FDR][5], 1),
        "sharpa": round(s["CHORD"][5] / s[FDR][5], 1),
        "claim": "approximately 100x lower training compute (Abstract, Sec. I)",
    }
    out["C2_spider_gain_vs_chord_pp"] = {
        "xhand": round(100 * (x[FDR][0] - x["CHORD"][0])),
        "sharpa": round(100 * (s[FDR][0] - s["CHORD"][0])),
        "claim": "44-percentage-point improvement (Abstract)",
    }
    out["C3_min_spider_gain_vs_any_baseline_pp_xhand"] = {
        "value": round(100 * min(x[FDR][0] - v[0] for k, v in x.items() if k != FDR)),
        "claim": "at least a 40-percentage-point higher success rate than the baselines (Sec. IV-B)",
    }
    out["C4_compute_ratio_doasido_over_fdr"] = {
        "xhand": round(x["Do as I Do"][5] / x[FDR][5], 2),
        "sharpa": round(s["Do as I Do"][5] / s[FDR][5], 2),
        "claim": "approximately 2x less compute than Do as I Do (Sec. IV-B, XHand)",
    }
    out["C5_fdr_less_compute_than_all_baselines"] = {
        hand: all(t[FDR][5] < v[5] for k, v in t.items() if k != FDR) for hand, t in TABLE.items()
    }
    out["C6_sharpa_text_values"] = {
        "SR_SPIDER": [s[FDR][0], s["Do as I Do"][0], s["CHORD"][0]],
        "SR_MT": [s[FDR][2], s["Do as I Do"][2], s["CHORD"][2]],
        "claim": "86% vs 64% / 50% under SR_SPIDER; 80% vs 10% / 0% under SR_MT (Sec. IV-B)",
    }

    # C7 표 내부 일관성: SR x 50이 정수이고 SR_MT <= SR_MT-Obj
    c7 = {}
    for hand, t in TABLE.items():
        for m, v in t.items():
            counts = [v[i] * N for i in range(3)]
            c7[f"{hand}/{m}"] = {
                "counts": [round(c) for c in counts],
                "integer": all(abs(c - round(c)) < 1e-9 for c in counts),
                "MT_le_MTObj": v[2] <= v[1],
            }
    out["C7_internal_consistency"] = c7

    # C8 신뢰구간과 검정 (같은 50개 모션이지만 모션별 결과가 없어 비대응 Fisher 검정으로 근사)
    c8 = {}
    for hand, t in TABLE.items():
        for i, metric in enumerate(METRICS):
            k_f = round(t[FDR][i] * N)
            best = max((v[i], k) for k, v in t.items() if k != FDR)
            k_b = round(best[0] * N)
            _, p = stats.fisher_exact([[k_f, N - k_f], [k_b, N - k_b]])
            c8[f"{hand}/{metric}"] = {
                "fdr": [k_f, wilson(k_f, N)],
                "best_baseline": [best[1], k_b, wilson(k_b, N)],
                "fisher_p": float(f"{p:.2e}"),
            }
    out["C8_ci_and_tests"] = c8

    # C9 성공 1건당 GPU-h
    c9 = {}
    for hand, t in TABLE.items():
        for m, v in t.items():
            row = {}
            for i, metric in enumerate(METRICS):
                k = round(v[i] * N)
                row[metric] = round(v[5] / k, 2) if k else None
            row["per_motion_attempt"] = round(v[5] / N, 2)
            c9[f"{hand}/{m}"] = row
    out["C9_gpu_hours_per_success"] = c9

    # C10 처리량과 메모리
    out["C10_throughput_and_memory"] = {
        "env_steps_per_s": {h: round(ENV_STEPS[h] / (TABLE[h][FDR][5] * 3600)) for h in TABLE},
        "iterations_per_s_at_2048_envs": {
            h: round(ENV_STEPS[h] / (TABLE[h][FDR][5] * 3600) / 2048, 2) for h in TABLE
        },
        "replay_buffer_gb": {
            "50M_rows": round(50 * 1.76, 1),
            "10M_rows_default_flashsac": round(10 * 1.76, 1),
            "source": "docs/guide/flashdexretarget.md: 1.76 GB per million rows; ~16 GB GPU besides the buffer",
        },
        "rtx3090_vram_gb": 24,
        "note": "50M 행 버퍼는 3090 VRAM에 들어가지 않으므로 buffer_device_type=cpu(호스트 RAM 약 100 GB)가 필요",
    }

    # C11 규모 확장 (프로젝트 페이지 수치, 본문 Fig. 6에는 숫자가 없음)
    fdr50 = round(x[FDR][1] * N)
    out["C11_scaling_efficiency"] = {
        "50_motion_xhand_successes_per_gpu_hour": round(fdr50 / x[FDR][5], 2),
        "1000_motion_successes_per_gpu_hour_lower_bound": round(600 / 60, 1),
        "ratio": round((600 / 60) / (fdr50 / x[FDR][5]), 1),
        "caveat": "1,000개 실험은 600M step 중 처음 60 GPU-h 구간, 50개 실험은 300M step 전체. 성공률도 60%대 vs 86%",
        "hours_for_600M_steps_at_xhand_throughput": round(600e6 / (ENV_STEPS["xhand"] / (x[FDR][5] * 3600)) / 3600, 1),
    }

    # C12 MOPE/MORE는 SR_SPIDER 통과 궤적만의 평균이라 표본이 다름
    out["C12_mope_more_subset_sizes"] = {
        f"{hand}/{m}": {"n_spider_success": round(v[0] * N), "MOPE_mm": v[3], "MORE_rad": v[4]}
        for hand, t in TABLE.items() for m, v in t.items()
    }

    # C13 레퍼런스 길이 대비 학습량: scripts/retarget/50_motion_benchmark.sh의 클립별 프레임 수(60 Hz)
    frames = {"taco": 3170, "oakink2": 12852, "hot3d": 10365}
    total = sum(frames.values())
    out["C13_reference_length_vs_budget"] = {
        "benchmark_frames": frames,
        "total_frames": total,
        "total_minutes_at_60hz": round(total / 60 / 60, 2),
        "mean_clip_seconds": round(total / N / 60, 1),
        "env_steps_per_reference_frame_xhand": round(ENV_STEPS["xhand"] / total),
        "gpu_seconds_per_reference_second_xhand": round(x[FDR][5] * 3600 / (total / 60)),
    }

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:300])
    print("wrote", path)


if __name__ == "__main__":
    main()

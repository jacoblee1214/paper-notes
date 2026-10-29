"""본문·그림·Table I 수치 재계산 (CPU). 실행: python3 analysis/table_check.py

그림 판독값은 analysis/fig_digitize.py의 결과(fig_digitize_results.json)를 읽는다. 없으면 먼저 그것을 실행한다.

- C1–C3: 계산량 비교(39.3배)와 그림 5·6의 checkpoint 대응
- C4–C5: 그림 5(a) 판독값과 본문 "0.3% 이하"의 관계, 분리 학습 81.4%의 분해
- C6: 그림 5(b) 표준편차가 이항 잡음으로 설명되는지
- C7–C9: 처리량, 그림 7의 계산량 범위, 평가 표본 수
- C10: Table I 다중 부품 성공률의 기하 감소 적합과 20개 외삽
- C11: 관측·망 차원(150 = 5 × 30, 278 = 128 + 150)과 SimBa 몸통 파라미터 수
- C12–C13: 독립 MuJoCo 구현 README의 표 내부 일관성
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIG = json.loads((ROOT / "analysis" / "fig_digitize_results.json").read_text())

# 본문 수치 (Sec. V)
SHARED_GPUH, SEP_TOTAL_GPUH, N_TASKS = 410, 16_100, 20
ENVS, ITERS, STEPS_PER_ITER, ENVS_PER_GPU = 64_000, 25_000, 32, 4_000
FIG5B = {"shared_sparse": (98.2, 0.986), "separate_sparse": (81.4, 1.732),
         "shared_dense": (17.8, 0.010), "separate_dense": (78.3, 0.573)}
TABLE1 = [99, 97, 95, 93, 92, 90, 89, 87, 86, 84]  # 1–10 active pairs
FIG7_TEXT = {"sparse_beta": {"1K": 38.6, "4K": 78.9, "8K": 86.9, "256K": 97.4},
             "sparse_uniform": {"256K": 84.3}, "dense_beta": {"256K": 77.4}, "dense_uniform": {"256K": 34.3}}
EVAL_TRIALS = 5 * 32 * 20  # 5 seeds × 32 trials × 20 tasks (Fig. 7 caption)
CONTROL_HZ, EP_SECONDS_PER_PAIR = 25, 15


def binom_sd(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)


def main() -> None:
    out: dict = {}
    c6 = DIG["fig6_curves"]

    # C1 39.3배
    out["C1_compute_ratio"] = {"separate_total_over_shared": round(SEP_TOTAL_GPUH / SHARED_GPUH, 2),
                               "claim": "39.3x more device-hours (Sec. V-A)"}
    # C2 분리 정책 하나의 비용 = 공유 정책의 몇 배인가 → 50K iteration을 시사
    per = SEP_TOTAL_GPUH / N_TASKS
    out["C2_per_specialist"] = {
        "gpu_h_each": round(per, 1), "ratio_to_shared_run": round(per / SHARED_GPUH, 2),
        "if_25K_iterations_like_shared": {"total_gpu_h": SHARED_GPUH * N_TASKS,
                                          "ratio": round(SHARED_GPUH * N_TASKS / SHARED_GPUH, 1)},
        "note": "같은 64K env·같은 iteration이면 정책 하나당 비용이 공유 정책과 같아야 한다. 805 GPU-h는 약 2배라 "
                "분리 정책이 50K iteration(그림 6의 끝)까지 돌았을 때의 비용으로 보인다(가설: 비용이 iteration에 비례).",
    }
    # C3 그림 5(b) 막대 = 그림 6의 어느 시점인가
    out["C3_fig5b_vs_fig6"] = {
        k: {"fig5b": FIG5B[k][0], "fig6_25K": c6[k]["25K"], "fig6_50K": c6[k]["50K"],
            "matches": "25K" if abs(FIG5B[k][0] - c6[k]["25K"]) < abs(FIG5B[k][0] - c6[k]["50K"]) else "50K"}
        for k in FIG5B
    }
    out["C3_reframed_comparison"] = {
        "equal_25K_iterations": {"shared": FIG5B["shared_sparse"][0], "separate_sparse": c6["separate_sparse"]["25K"],
                                 "separate_dense": c6["separate_dense"]["25K"], "compute_ratio": 20.0},
        "separate_run_to_50K": {"shared_at_25K": FIG5B["shared_sparse"][0], "separate_sparse_50K": c6["separate_sparse"]["50K"],
                                "separate_dense_50K": c6["separate_dense"]["50K"],
                                "compute_ratio": round(SEP_TOTAL_GPUH / SHARED_GPUH, 1)},
    }

    # C4 그림 5(a) 판독 행 평균 vs 그림 5(b), 그리고 본문 "<=0.3%" 칸
    cells = DIG["fig5a_cells"]
    out["C4_heatmap_vs_text"] = {
        "row_means_digitized": DIG["fig5a_row_means"],
        "fig5b_reported": {k: v[0] for k, v in FIG5B.items()},
        "separate_sparse_N4_R4_P4": {t: cells["separate_sparse"][t] for t in ("N4", "R4", "P4")},
        "text": "From randomly initialized states, separately-trained policies have <=0.3% success on M4 nut, 4 mm peg, rod insertion",
        "reading": "heatmap은 모든 reset 종류의 평균이고 본문은 무작위 초기 상태만의 값으로 보인다. 그렇다면 이 세 과제의 "
                   "reset bank 가운데 약 30–50%는 손에 쥔 상태·근접·부분 조립처럼 분리 정책도 풀 수 있는 쉬운 시작점이다.",
    }
    # C5 81.4%의 분해: 판독값으로 '어려운 칸'을 빼면 나머지 평균은?
    sep = cells["separate_sparse"]
    hard = [t for t, v in sep.items() if v < 60]
    rest = [v for t, v in sep.items() if t not in hard]
    out["C5_separate_sparse_decomposition"] = {
        "cells_below_60": {t: sep[t] for t in hard},
        "mean_of_rest": round(float(np.mean(rest)), 1), "n_rest": len(rest),
        "shared_sparse_same_tasks": {t: cells["shared_sparse"][t] for t in hard},
    }

    # C6 그림 5(b) SD가 이항 잡음과 맞는가: 시드당 시행 수 n을 역산
    out["C6_sd_plausibility"] = {}
    for k, (m, sd) in FIG5B.items():
        p = m / 100
        n_implied = p * (1 - p) / (sd / 100) ** 2
        out["C6_sd_plausibility"][k] = {"mean": m, "sd": sd, "n_trials_per_seed_implied": round(n_implied),
                                        "binom_sd_if_640_trials": round(100 * binom_sd(p, 640), 2)}
    out["C6_note"] = "그림 7 기준(시드당 32 trial × 20 과제 = 640)이면 17.8%의 이항 SD는 1.5%다. 0.010%는 1,460만 trial이 필요해 단위 오기(0.010 = 1.0%)로 보인다(가설)."

    # C7 처리량
    steps = ENVS * ITERS * STEPS_PER_ITER
    gpus = ENVS // ENVS_PER_GPU
    out["C7_throughput"] = {"env_steps_64K_25K": f"{steps:.3e}", "gpus": gpus,
                            "wall_clock_h": round(SHARED_GPUH / gpus, 1),
                            "env_steps_per_gpu_s": round(steps / (SHARED_GPUH * 3600)),
                            "per_env_steps_per_s": round(steps / (SHARED_GPUH * 3600) / ENVS_PER_GPU, 1),
                            "gpu_type": "본문에 없음"}
    # C8 그림 7은 계산량이 고정되지 않은 비교다(iteration 고정, env 수를 늘림)
    out["C8_fig7_compute_range"] = {
        env: {"gpus": max(1, env // ENVS_PER_GPU), "env_steps": f"{env * ITERS * STEPS_PER_ITER:.2e}",
              "gpu_h_if_linear": round(max(1, env // ENVS_PER_GPU) * SHARED_GPUH / gpus)}
        for env in (1_000, 4_000, 8_000, 64_000, 256_000)
    }
    # C9 그림 7 평가 표본과 이항 표준오차
    out["C9_fig7_eval_noise"] = {
        "trials_per_point": EVAL_TRIALS,
        "se_pp": {f"{k}@{e}": round(100 * binom_sd(v / 100, EVAL_TRIALS), 2) for k, d in FIG7_TEXT.items() for e, v in d.items()},
        "beta_minus_uniform_256K_sparse_pp": round(FIG7_TEXT["sparse_beta"]["256K"] - FIG7_TEXT["sparse_uniform"]["256K"], 1),
        "note": "SD는 평가 시드 5개의 것이다. 학습 시드는 설정마다 하나로 보이며, 학습 재현성은 보고되지 않았다.",
    }

    # C10 Table I: 기하 감소 적합 s_k = s_1 · r^(k-1)
    k = np.arange(1, 11)
    s = np.array(TABLE1) / 100
    r_fit = float(np.exp(np.polyfit(k - 1, np.log(s), 1)[0]))
    ratios = [round(TABLE1[i] / TABLE1[i - 1], 3) for i in range(1, 10)]
    out["C10_table1"] = {
        "consecutive_ratios": ratios, "per_pair_ratio_fit": round(r_fit, 4),
        "independent_99_model_at_10": round(100 * 0.99 ** 10, 1),
        "extrapolate_20_pairs_fit": round(100 * s[0] * r_fit ** 19, 1),
        "two_groups_of_10": round(100 * s[-1] ** 2, 1),
        "episode_steps": {"per_pair": CONTROL_HZ * EP_SECONDS_PER_PAIR, "10_pairs": 10 * CONTROL_HZ * EP_SECONDS_PER_PAIR},
        "missing": "시행 수, 표준편차, 활성 부품을 어떻게 골랐는지(고정·무작위)가 없다",
    }

    # C11 관측·망 차원 (Fig. 3)
    proprio = 7 + 6 + 9 + 8
    trunk_in = 128 + 150

    def simba_trunk(d_in, d_out, h=256, e=1024, blocks=2):
        p = d_in * h + h                        # 입력 사영
        p += blocks * (2 * h + h * e + e + e * h + h)  # 블록: LN(2h) + 확장 + 축소
        p += 2 * h + h * d_out + d_out           # 마지막 LN + 출력
        return p

    out["C11_architecture"] = {"proprio_per_frame": proprio, "history_frames_if_150": 150 / proprio,
                               "trunk_input": trunk_in, "trunk_input_matches_fig3": trunk_in == 278,
                               "actor_trunk_params": simba_trunk(278, 8), "critic_trunk_params": simba_trunk(278, 1),
                               "point_encoder": "3 × n_pts → 256 → 128, n_pts 미기재 → 파라미터 수 미정"}

    # C12 독립 MuJoCo 구현(suniique/assembly-nist @ eb5d15e) README의 부품별 표 → 전체(20과제 가중) 재계산
    counts = {"Rods": 4, "Pegs": 4, "Gears": 3, "Nuts": 4, "Connectors": 5}
    readme = {"10K": {"Rods": 98.6, "Pegs": 98.1, "Gears": 98.3, "Nuts": 98.0, "Connectors": 99.0, "Overall": 98.4},
              "12.5K": {"Rods": 98.6, "Pegs": 98.6, "Gears": 99.0, "Nuts": 97.5, "Connectors": 99.6, "Overall": 98.7}}
    out["C12_independent_impl_consistency"] = {
        it: {"weighted_overall": round(sum(v[g] * c for g, c in counts.items()) / 20, 2), "reported": v["Overall"]}
        for it, v in readme.items()
    }
    out["C13_independent_impl_note"] = ("README의 'Original AlphaNIST 86.9'는 AlphaNIST 그림 7의 8K env 값이다. "
                                        "시뮬레이터·자산·접촉 모델이 달라 직접 비교가 아니라고 README도 적었다.")

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for key, v in out.items():
        print(key, json.dumps(v, ensure_ascii=False)[:260])
    print("wrote", path)


if __name__ == "__main__":
    main()

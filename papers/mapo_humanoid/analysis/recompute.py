"""MAPO (arXiv:2611.02613 v1) 논문 표 수치를 다시 계산하고 검정한다.

입력: 논문 표 1(쪽 7)과 부록 D "Seed Variance"(쪽 21), 표 C.1·C.2(쪽 18–19)의 값을 아래에 그대로 옮겼다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 결과에 표시한다)
- 논문은 학습·평가 모션 수를 적지 않는다. 정수 검사(V1)는 "성공률 = 성공한 모션 수 / 분모 M"을 가정하고,
  분모가 모션 수인지 모션 수 x 롤아웃 5인지는 구분하지 않는다.
- 부록 D의 "mean ± standard deviation across the three training seeds"는 시드 3개의 표본 SD(ddof=1)로 읽는다.
  모집단 SD(ddof=0)로 읽은 결과도 함께 낸다.
- 시드 간 검정은 Welch t 검정(시드 3개, 시드마다 값이 없어 요약 통계로 계산)이다. 시드 수가 3이라 검정력이 낮다.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}

METRICS = ["success", "eg_pos", "el_pos", "eacc", "evel", "ert_h", "ert_vel", "ert_r", "ert_p"]
LOWER_BETTER = {m: (m != "success") for m in METRICS}

# 표 1(a) 학습 집합 (AMASS + LAFAN1), 기본 전문가 수 N = 6 (본문 6.1절)
T1A = {
    "Monolithic PPO":      [81.78, 165.7, 96.50, 5.787, 16.35, 0.060, 0.532, 0.169, 0.136],
    "PPO with MoE":        [83.38, 153.1, 87.98, 5.499, 14.30, 0.058, 0.501, 0.120, 0.105],
    "MAPO with fixed router": [83.78, 152.7, 86.65, 5.250, 14.42, 0.052, 0.487, 0.144, 0.122],
    "MAPO":                [86.60, 147.4, 78.10, 4.576, 11.78, 0.048, 0.406, 0.138, 0.117],
}
# 표 1(a) 전문가 수 절제
T1A_N = {
    4:  {"PPO with MoE": [83.96, 156.5, 85.01, 5.301, 13.75, 0.054, 0.490, 0.131, 0.119],
         "MAPO":         [86.87, 143.6, 74.61, 4.672, 11.60, 0.050, 0.407, 0.119, 0.108]},
    8:  {"PPO with MoE": [82.28, 164.4, 93.47, 5.841, 15.51, 0.064, 0.538, 0.126, 0.110],
         "MAPO":         [87.85, 134.8, 71.65, 4.120, 10.06, 0.045, 0.360, 0.138, 0.115]},
    16: {"PPO with MoE": [82.95, 159.6, 94.08, 6.175, 15.54, 0.061, 0.528, 0.118, 0.107],
         "MAPO":         [87.73, 136.9, 72.58, 4.549, 11.02, 0.048, 0.375, 0.128, 0.110]},
}
# 표 1(b) 시험 집합 (OMOMO)
T1B = {
    "Monolithic PPO":      [84.36, 227.2, 86.43, 6.014, 12.44, 0.035, 0.356, 0.157, 0.134],
    "PPO with MoE":        [82.78, 241.4, 93.90, 6.451, 13.77, 0.041, 0.397, 0.140, 0.130],
    "MAPO with fixed router": [85.21, 221.6, 87.01, 5.937, 12.30, 0.031, 0.331, 0.153, 0.136],
    "MAPO":                [85.45, 225.8, 85.29, 5.764, 11.76, 0.030, 0.326, 0.151, 0.129],
    "HOVER":               [83.39, 272.2, 122.81, 6.001, 12.46, 0.058, 0.382, 0.193, 0.163],
    "TWIST 2":             [57.85, 386.6, 104.72, 4.661, 11.87, 0.042, 0.355, 0.080, 0.074],
}
# 표 1에서 밑줄·굵은 글씨로 표시된 칸(그림 판독): {metric: (best, second)}
BOLD_A = {"success": ("MAPO", "MAPO with fixed router"), "eg_pos": ("MAPO", "MAPO with fixed router"),
          "el_pos": ("MAPO", "MAPO with fixed router"), "eacc": ("MAPO", "MAPO with fixed router"),
          "evel": ("MAPO", "PPO with MoE"), "ert_h": ("MAPO", "MAPO with fixed router"),
          "ert_vel": ("MAPO", "MAPO with fixed router"), "ert_r": ("PPO with MoE", "MAPO"),
          "ert_p": ("PPO with MoE", "MAPO")}
BOLD_B = {"success": ("MAPO", "MAPO with fixed router"), "eg_pos": ("MAPO with fixed router", "MAPO"),
          "el_pos": ("MAPO", "Monolithic PPO"), "eacc": ("TWIST 2", "MAPO"),
          "evel": ("MAPO", "TWIST 2"), "ert_h": ("MAPO", "MAPO with fixed router"),
          "ert_vel": ("MAPO", "MAPO with fixed router"), "ert_r": ("TWIST 2", "PPO with MoE"),
          "ert_p": ("TWIST 2", "MAPO")}

# 부록 D "Seed Variance" (쪽 21): 시드 3개의 평균 ± SD
SEED = {"MAPO": {"success": (87.03, 0.64), "eg_pos": (147.9, 1.5)},
        "PPO with MoE": {"success": (82.89, 0.43), "eg_pos": (154.1, 1.7)}}
SEED_COUNTS = {"mapo_only": [182, 155, 191], "ppo_only": [1, 1, 0]}


def best_second(table, mi):
    vals = {k: v[mi] for k, v in table.items()}
    rev = not LOWER_BETTER[METRICS[mi]]
    order = sorted(vals, key=lambda k: vals[k], reverse=rev)
    return order[0], order[1]


def v1_denominators():
    """성공률(소수 둘째 자리)이 k/M으로 나올 수 있는 최소 분모 M."""
    def feasible(vals, M):
        for v in vals:
            ok = False
            k0 = int(round(v / 100 * M))
            for k in (k0 - 1, k0, k0 + 1):
                if 0 <= k <= M and round(100.0 * k / M + 1e-9, 2) == round(v, 2):
                    ok = True
                    break
            if not ok:
                return False
        return True

    train = [v[0] for v in T1A.values()] + [d[m][0] for d in T1A_N.values() for m in d]
    test = [v[0] for v in T1B.values()]
    out = {}
    for name, vals in (("train", train), ("test", test), ("train+test", train + test)):
        ms = [M for M in range(10, 60001) if feasible(vals, M)]
        out[name] = {"n_values": len(vals), "min_denominator": ms[0] if ms else None,
                     "first10_denominators": ms[:10], "all_ok_from": None}
        # 모든 분모가 가능해지는 지점: 0.01% 간격보다 촘촘해지면 어떤 값도 가능
    out["note"] = "분모가 약 10,000 이상이면 소수 둘째 자리 성공률은 어떤 값도 정수 비율로 가능해 이 검사로는 걸러지지 않는다."
    return out


def v2_deltas():
    out = {"train_N6": {}, "train_N": {}, "test": {}}
    a = T1A["MAPO"]
    b = T1A["PPO with MoE"]
    for i, m in enumerate(METRICS):
        out["train_N6"][m] = {"mapo": a[i], "ppo_moe": b[i], "diff": round(a[i] - b[i], 4),
                              "rel_pct": round(100 * (a[i] - b[i]) / b[i], 2)}
    for n, d in T1A_N.items():
        out["train_N"][n] = {m: round(d["MAPO"][i] - d["PPO with MoE"][i], 4) for i, m in enumerate(METRICS)}
    a = T1B["MAPO"]
    for ref in ("PPO with MoE", "Monolithic PPO", "MAPO with fixed router", "HOVER", "TWIST 2"):
        b = T1B[ref]
        out["test"][ref] = {m: round(a[i] - b[i], 4) for i, m in enumerate(METRICS)}
    return out


def v3_best_cells():
    """굵은 글씨(최선) 칸 수, 본문 "대부분 지표에서 최저 오차" 주장 검사."""
    out = {}
    for name, table in (("train", T1A), ("test", T1B)):
        # 표의 굵은 글씨와 우리 계산이 같은지
        mism = []
        for i, m in enumerate(METRICS):
            b, s = best_second(table, i)
            bold = (BOLD_A if name == "train" else BOLD_B)[m]
            if (b, s) != bold:
                mism.append({"metric": m, "computed": [b, s], "table": list(bold)})
        wins = {}
        for i, m in enumerate(METRICS):
            b, _ = best_second(table, i)
            wins[b] = wins.get(b, 0) + 1
        err_wins_mapo = sum(1 for i, m in enumerate(METRICS) if m != "success" and best_second(table, i)[0] == "MAPO")
        out[name] = {"best_cells_per_method": wins, "mapo_error_metric_wins_of_8": err_wins_mapo,
                     "mapo_success_best": best_second(table, 0)[0] == "MAPO", "bold_mismatch_vs_computed": mism}
    # 시험 집합에서 HOVER·TWIST 2를 뺀 네 행 비교
    four = {k: T1B[k] for k in ("Monolithic PPO", "PPO with MoE", "MAPO with fixed router", "MAPO")}
    out["test_four_rows"] = {"mapo_error_metric_wins_of_8": sum(
        1 for i, m in enumerate(METRICS) if m != "success" and best_second(four, i)[0] == "MAPO")}
    return out


def v4_n_ablation_claim():
    """본문: "MAPO가 모든 전문가 수에서 PPO with MoE보다 낫다. 단 root roll, pitch는 제외"."""
    rows = []
    allN = {6: {"PPO with MoE": T1A["PPO with MoE"], "MAPO": T1A["MAPO"]}, **T1A_N}
    for n, d in sorted(allN.items()):
        worse = [m for i, m in enumerate(METRICS)
                 if ((d["MAPO"][i] < d["PPO with MoE"][i]) if not LOWER_BETTER[m] else (d["MAPO"][i] > d["PPO with MoE"][i]))]
        rows.append({"N": n, "metrics_where_mapo_is_worse": worse})
    return rows


def welch_summary(m1, s1, n1, m2, s2, n2):
    se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    t = (m1 - m2) / se
    df = se ** 4 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), df)
    # 차이의 95% 구간
    h = stats.t.ppf(0.975, df) * se
    return {"diff": round(m1 - m2, 4), "t": round(t, 3), "df": round(df, 2), "p": float(p),
            "ci95": [round(m1 - m2 - h, 3), round(m1 - m2 + h, 3)]}


def v5_seed_variance():
    out = {}
    for metric in ("success", "eg_pos"):
        (m1, s1), (m2, s2) = SEED["MAPO"][metric], SEED["PPO with MoE"][metric]
        sample = welch_summary(m1, s1, 3, m2, s2, 3)
        # 모집단 SD로 읽으면 표본 SD = SD * sqrt(3/2)
        k = math.sqrt(3 / 2)
        pop = welch_summary(m1, s1 * k, 3, m2, s2 * k, 3)
        out[metric] = {"sample_sd_reading": sample, "population_sd_reading": pop}
    # 표 1 단일 시드 값과 시드 평균의 관계: 나머지 두 시드 평균과, 이 값이 가능한 최소 SD
    chk = {}
    for method, tab_idx in (("MAPO", None), ("PPO with MoE", None)):
        for metric, mi in (("success", 0), ("eg_pos", 1)):
            single = (T1A["MAPO"] if method == "MAPO" else T1A["PPO with MoE"])[mi]
            mean, sd = SEED[method][metric]
            other_mean = (3 * mean - single) / 2
            # 나머지 두 시드가 같은 값일 때 표본 SD(ddof=1)가 최소
            sd_min = math.sqrt(((single - mean) ** 2 + 2 * (other_mean - mean) ** 2) / 2)
            chk[f"{method}.{metric}"] = {"table1_value": single, "seed_mean": mean, "seed_sd": sd,
                                          "other_two_mean": round(other_mean, 3), "min_possible_sd_ddof1": round(sd_min, 3),
                                          "consistent": sd >= sd_min - 1e-9}
    out["table1_vs_seed_stats"] = chk
    # 표 1 값이 시드 중 하나라면, 나머지 두 시드의 차이가 가질 수 있는 범위(보고된 SD의 반올림 구간, ddof=1)
    gaps = {}
    for key, v in chk.items():
        sd, single, mean = v["seed_sd"], v["table1_value"], v["seed_mean"]
        dec = 2 if sd < 1 else 1
        half = 0.5 * 10 ** (-dec)
        ss_lo, ss_hi = 2 * (sd - half) ** 2, 2 * (sd + half) ** 2   # 표본 분산 x 2 = 제곱합
        base = (single - mean) ** 2 + 2 * ((3 * mean - single) / 2 - mean) ** 2
        lo = math.sqrt(max(0.0, 2 * (ss_lo - base))) if ss_lo >= base else 0.0
        hi = math.sqrt(2 * (ss_hi - base)) if ss_hi >= base else None
        gaps[key] = {"other_two_seed_gap_range": [round(lo, 3), None if hi is None else round(hi, 3)]}
    out["other_two_seed_gap_given_rounded_sd"] = gaps
    # 표 1 값이 3시드 평균과 SD의 몇 배 떨어져 있는지
    out["table1_z_vs_seed_mean"] = {
        k: round((v["table1_value"] - v["seed_mean"]) / v["seed_sd"], 2) for k, v in chk.items()}
    return out


def v6_counts():
    c = SEED_COUNTS
    mean = float(np.mean(c["mapo_only"]))
    sd = float(np.std(c["mapo_only"], ddof=1))
    out = {"mapo_only_counts": c["mapo_only"], "mean": round(mean, 2), "sd_ddof1": round(sd, 2),
           "net_counts": [a - b for a, b in zip(c["mapo_only"], c["ppo_only"])]}
    # 성공률 차이(포인트)와 모션 수 N을 연결하는 가정표: 순 차이 모션 수 = diff_pp/100 * N
    diffs = {"table1_single_seed": 86.60 - 83.38, "three_seed_mean": 87.03 - 82.89}
    grid = {}
    for N in (3000, 5000, 8000, 11000, 12000):
        grid[N] = {k: {"net_gap_motions": round(d / 100 * N, 1),
                       "share_explained_by_consistent_set": round(181 / (d / 100 * N), 2)} for k, d in diffs.items()}
    out["assumed_N_grid"] = grid
    out["N_where_consistent_set_equals_gap"] = {k: round(181 / (d / 100), 0) for k, d in diffs.items()}
    out["note"] = ("AMASS 원논문(arXiv:1904.03278)은 11,000개 이상 모션이라고 쓴다. 이 논문이 쓴 모션 수는 본문에 없다. "
                   "일관된 승리 집합(182 - 1 = 181개)이 성공률 차이로 설명되는 몫은 N이 작을수록 커진다.")
    # 일관 승리 집합 크기의 시드 간 변동: 포아송 근사
    out["poisson_sd_if_independent"] = round(math.sqrt(mean), 2)
    return out


def v7_fig_reads():
    """그림에서 읽은 값(근사)과 그 계산. 판독 오차는 눈금 한 칸 이내로 본다."""
    out = {}
    # 그림 4(a): 30,000 반복에서 MAPO ≈ 28, PPO with MoE ≈ 15.5 (그림 판독)
    out["reward_ratio_fig4a"] = {"mapo": 28.0, "ppo_moe": 15.5, "ratio": round(28.0 / 15.5, 2),
                                 "note": "프로젝트 페이지는 'about 1.8x'라고 쓴다. 그림 판독 값 기준 1.8배와 맞는다."}
    # 그림 4(c): MAPO - PPO 전문가쌍 거리 차이 행렬 (상삼각 15칸)
    M = {(1, 2): -2.0, (1, 3): -0.3, (1, 4): -0.5, (1, 5): 1.0, (1, 6): -0.5,
         (2, 3): 0.8, (2, 4): 0.3, (2, 5): 4.0, (2, 6): 2.5,
         (3, 4): 2.4, (3, 5): 3.9, (3, 6): 2.1,
         (4, 5): 4.3, (4, 6): 3.0, (5, 6): 5.3}
    v = np.array(list(M.values()))
    out["pairwise_distance_diff_fig4c"] = {"pairs": 15, "positive": int((v > 0).sum()), "negative": int((v < 0).sum()),
                                           "mean": round(float(v.mean()), 2), "median": round(float(np.median(v)), 2),
                                           "pairs_with_E1_negative": int(sum(1 for k, x in M.items() if 1 in k and x < 0)),
                                           "note": "그림 판독 값. 단위와 색 막대 눈금(±4)을 넘는 5.3이 있다."}
    # 그림 4(b): 정렬된 혼합 가중치 평균 (N = 6)
    top = {"MAPO": (0.62, 0.18), "PPO": (0.42, 0.31)}
    rest = {k: round(1 - a - b, 2) for k, (a, b) in top.items()}
    out["mixing_weights_fig4b"] = {"top1_top2": top, "remaining_four_sum": rest,
                                   "uniform_weight_1_over_6": round(1 / 6, 3), "note": "그림 판독 값(눈금 0.2 간격)"}
    # 부록 D: 클리핑 비율 (본문 수치)
    out["clip_fractions_text"] = {"vanilla_pg_mapo_router_criterion": 0.05, "vanilla_pg_ppo_criterion": 0.50,
                                  "mapo_training_mapo_router_criterion": 0.02, "mapo_training_ppo_criterion": 0.24,
                                  "ratio_vanilla": 10.0, "ratio_mapo_training": 12.0,
                                  "mapo_training_expert_criterion_fig_read": 0.20}
    return out


def v8_compute():
    """표 C.2와 본문에서 계산 가능한 학습 규모(가정은 문장에 표시)."""
    envs, steps, mb, ep = 4096, 24, 4, 5
    batch = envs * steps
    updates_per_iter = mb * ep
    iters_main, iters_ablation, iters_fig2 = 30000, 15000, 10000
    env_steps_main = batch * iters_main
    out = {"batch_per_iter": batch, "minibatch_size": batch // mb, "grad_updates_per_iter": updates_per_iter,
           "grad_updates_main_run": updates_per_iter * iters_main,
           "env_steps_main_run": env_steps_main, "env_steps_ablation_run": batch * iters_ablation,
           "assumption": "그림 4의 'Training steps' 30,000을 PPO 반복(iteration) 횟수로 가정(rsl_rl 관행, 논문에 정의 없음)"}
    wall = {}
    for fps in (20000, 40000, 60000):
        wall[fps] = round(env_steps_main / fps / 3600, 1)
    out["wallclock_hours_main_run_if_env_steps_per_s"] = wall
    # 보고된 실행 수(표 1과 그림 5, D.4에서 세어 본 하한)
    runs = {"table1_main_teacher_runs": 5 + 6, "extra_seeds_mapo_and_ppo_moe": 4,
            "fig5_curves_lower_bound": 4 + 4 * 2 + 4 + 6}
    out["run_count_lower_bound"] = runs
    out["run_count_total_approx"] = sum(runs.values())
    return out


def v9_reward_table():
    """표 C.1: 보상 항 가중치와 지수 보상의 오차-보상 대응."""
    pos = {"root_ori": 1.0, "rel_body_pos": 20.0, "rel_body_ori": 1.0, "body_lin_vel": 1.0, "body_ang_vel": 1.0}
    neg = {"action_rate": -0.2, "joint_limit": -10.0, "undesired_contact": -0.1}
    out = {"positive_weight_sum": sum(pos.values()), "negative_weights": neg,
           "rel_body_pos_share_of_positive_weight": round(20.0 / sum(pos.values()), 3)}
    # 상대 몸체 위치 보상 exp(-e^2/0.05): 오차 e(m)에서의 값 (e는 몸체 평균 L2 제곱 오차의 제곱근으로 가정)
    out["rel_body_pos_reward_at_error_m"] = {str(e): round(math.exp(-e * e / 0.05), 4) for e in (0.05, 0.1, 0.2, 0.3, 0.5)}
    out["success_threshold_m"] = 0.5
    out["note"] = ("같은 크기(0.5 m)의 루트 상대 몸체 오차에서 이 보상 항은 0.007이다. 성공 기준의 오차(평균 위치 오차, 전역인지 루트 상대인지 본문에 없음)와 "
                   "보상 항의 오차(루트 상대)는 정의가 같다고 볼 수 없어 크기 비교만 한다.")
    return out


def v10_test_power():
    """시험 집합 성공률 차이(MAPO - Monolithic = 1.09 %p, MAPO - PPO MoE = 2.67 %p)가 구분되려면 필요한 모션 수.
    시험 모션 수 N은 본문에 없다. 같은 모션에 대한 짝 비교(McNemar)로 보고, 두 정책이 갈리는 모션 비율 d를 가정한다."""
    out = {}
    for ref, diff in (("Monolithic PPO", 85.45 - 84.36), ("PPO with MoE", 85.45 - 82.78), ("MAPO with fixed router", 85.45 - 85.21)):
        rows = {}
        for d in (0.05, 0.10, 0.20):
            # 필요 N (양측 5%, 검정력 80%): n_disc = N*d, 승-패 차이 = N*diff/100; 정규 근사로 N ≈ (z_a + z_b)^2 * d / (diff/100)^2
            za, zb = 1.959964, 0.841621
            N = (za + zb) ** 2 * d / ((diff / 100) ** 2)
            rows[str(d)] = int(math.ceil(N))
        out[ref] = {"diff_pp": round(diff, 2), "required_N_by_discordant_fraction": rows}
    # 가정 N에서의 짝 비교 p 값 (d = 0.10 가정)
    p_tab = {}
    for N in (500, 1000, 2000, 5000):
        row = {}
        for ref, diff in (("Monolithic PPO", 1.09), ("PPO with MoE", 2.67)):
            d = 0.10
            se = math.sqrt(d / N - (diff / 100) ** 2 / N)
            z = (diff / 100) / se
            row[ref] = round(2 * stats.norm.sf(abs(z)), 4)
        p_tab[str(N)] = row
    out["assumed_p_discordant_0.10"] = p_tab
    out["assumption"] = "N(시험 모션 수)와 d(두 정책의 성공·실패가 갈리는 모션 비율)는 논문에 없다. d = 0.05, 0.10, 0.20을 가정."
    return out


def v11_grad_scale():
    """MoE 정책 한 갱신에서 N 증가에 따른 파라미터 수 (표 C.2, 관측 차원 O는 논문에 없어 입력층만 O의 함수로 표기)."""
    def mlp(sizes):
        return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))

    out = {}
    for O in (200, 300, 400):
        act = 29
        enc = mlp([O, 512, 256])
        router = lambda n: mlp([256, 256, 128, n])
        expert = mlp([256, 128, 128, act])
        row = {}
        for n in (2, 4, 6, 8, 16):
            row[str(n)] = enc + router(n) + n * expert
        out[f"obs_dim_{O}"] = row
    out["note"] = ("표 C.2의 층 크기에서 계산. 관측 차원 O, 행동 차원 29(G1)는 가정. 전문가 수를 키우면 파라미터가 선형 증가해 "
                   "표 1의 N 절제는 용량과 섞여 있다. 단일 정책(Monolithic PPO)의 층 크기는 논문에 없다.")
    return out


def main():
    RES["V1_denominators"] = v1_denominators()
    RES["V2_deltas"] = v2_deltas()
    RES["V3_best_cells"] = v3_best_cells()
    RES["V4_n_ablation_claim"] = v4_n_ablation_claim()
    RES["V5_seed_variance"] = v5_seed_variance()
    RES["V6_counts"] = v6_counts()
    RES["V7_figure_reads"] = v7_fig_reads()
    RES["V8_compute"] = v8_compute()
    RES["V9_reward_table"] = v9_reward_table()
    RES["V10_test_power"] = v10_test_power()
    RES["V11_param_count"] = v11_grad_scale()
    OUT.write_text(json.dumps(RES, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    for k, v in RES.items():
        print("==", k)
        print(json.dumps(v, indent=1, ensure_ascii=False, default=float)[:1800])


if __name__ == "__main__":
    main()

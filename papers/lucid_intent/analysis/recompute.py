"""LUCID (arXiv:2606.11628 v1) 논문 그림·표의 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문의 그림 12·13 막대 라벨(시도 수), 그림 5·6의 판독 값, 부록 표 1–11의 값을 아래에 옮겨 적었다.
      외부 데이터와 가중치는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 성공 수는 그림 12(웹 과제, 칸당 10회)와 그림 13(자체 수집 과제, 칸당 15회) 막대에 적힌 분수다.
- 그림 6의 성공률(30회 중)과 held-out 손실은 그림에서 읽은 값이다(판독 오차: 성공률 ±1 %p, 손실 ±0.01).
  성공률은 k/30 격자에서 가장 가까운 값으로 맞췄다(1/30, 3/30, 12/30, 19/30, 22/30).
- 그림 7의 에피소드 보상은 곡선 끝을 읽은 값(판독 오차 ±2)이다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}
Z = 1.959964


def wilson(k, n, z=Z):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2, z=Z):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def fisher(k1, n1, k2, n2):
    return stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


# ---------------------------------------------------------------------------
# 1. 그림 12: 웹 과제, 칸당 10회. 순서: success, unrecoverable, perception loss, incorrect behavior
# ---------------------------------------------------------------------------
FIG12 = {
    "stirring": {
        "ours": [[6, 2, 2, 0], [8, 2, 0, 0], [5, 2, 0, 3]],
        "open": [[4, 6, 0, 0], [3, 7, 0, 0], [0, 8, 0, 2]],
        "overall_ours": [19, 6, 2, 3], "overall_open": [7, 21, 0, 2],
    },
    "wiping": {
        "ours": [[9, 1, 0, 0], [7, 1, 0, 2], [10, 0, 0, 0]],
        "open": [[3, 2, 0, 5], [2, 4, 0, 4], [8, 0, 0, 2]],
        "overall_ours": [26, 2, 0, 2], "overall_open": [13, 6, 0, 11],
    },
    "binning": {
        "ours": [[7, 3, 0, 0], [9, 0, 1, 0], [5, 3, 2, 0]],
        "open": [[4, 3, 3, 0], [1, 6, 0, 3], [0, 8, 0, 2]],
        "overall_ours": [21, 6, 3, 0], "overall_open": [5, 17, 3, 5],
    },
}
# 그림 13: 자체 수집 과제, 칸당 15회. 순서: success, 나머지는 모두 incorrect behavior
FIG13 = {"push_t": {"hand": 12, "gripper": 10}, "cable": {"hand": 7, "gripper": 9}}


def block_web():
    out = {"sum_check": {}, "per_task": {}, "per_scenario": [], "pooled": {}, "failure_tally": {}}
    tot = {"ours": [0, 0], "open": [0, 0]}
    cats_o = np.zeros(4, int)
    cats_p = np.zeros(4, int)
    pvals = []
    labels = []
    strata = []
    for task, d in FIG12.items():
        for arm in ("ours", "open"):
            s = np.sum(d[arm], axis=0).tolist()
            out["sum_check"][f"{task}_{arm}"] = {"sum_of_scenarios": s, "reported_overall": d["overall_" + arm],
                                                  "match": s == d["overall_" + arm],
                                                  "row_sums_are_10": all(sum(r) == 10 for r in d[arm])}
        ko = sum(r[0] for r in d["ours"])
        kp = sum(r[0] for r in d["open"])
        tot["ours"][0] += ko
        tot["open"][0] += kp
        tot["ours"][1] += 30
        tot["open"][1] += 30
        cats_o += np.sum(d["ours"], axis=0)
        cats_p += np.sum(d["open"], axis=0)
        dd = newcombe_diff(ko, 30, kp, 30)
        out["per_task"][task] = {"ours": [ko, 30, ko / 30, *wilson(ko, 30)], "open": [kp, 30, kp / 30, *wilson(kp, 30)],
                                 "diff_pp": [100 * dd[0], 100 * dd[1], 100 * dd[2]], "fisher_p": fisher(ko, 30, kp, 30)}
        for i in range(3):
            a, b = d["ours"][i][0], d["open"][i][0]
            p = fisher(a, 10, b, 10)
            out["per_scenario"].append({"task": task, "scenario": i + 1, "ours": a, "open": b, "fisher_p": p})
            pvals.append(p)
            labels.append((task, i + 1))
            strata.append((a, 10 - a, b, 10 - b))
    adj = holm(pvals)
    for row, a in zip(out["per_scenario"], adj):
        row["holm_p"] = a
    ko, n = tot["ours"]
    kp, _ = tot["open"]
    out["pooled"] = {"ours": [ko, n, ko / n, *wilson(ko, n)], "open": [kp, n, kp / n, *wilson(kp, n)],
                     "paper_claim": "73% vs 28%", "fisher_p": fisher(ko, n, kp, n),
                     "diff_pp": [100 * x for x in newcombe_diff(ko, n, kp, n)]}
    # Mantel-Haenszel (9개 칸을 층으로): 시나리오 이질성에 덜 의존
    num = den = var = 0.0
    orn = ord_ = 0.0
    for a, b, c, d_ in strata:
        n_ = a + b + c + d_
        r1, r2, c1, c2 = a + b, c + d_, a + c, b + d_
        num += a - r1 * c1 / n_
        var += r1 * r2 * c1 * c2 / (n_ ** 2 * (n_ - 1))
        orn += a * d_ / n_
        ord_ += b * c / n_
    chi = (abs(num) - 0.5) ** 2 / var
    out["mantel_haenszel_9_strata"] = {"chi2_cc": chi, "p": float(stats.chi2.sf(chi, 1)),
                                       "common_or": orn / ord_ if ord_ else None}
    out["failure_tally"] = {"categories": ["success", "unrecoverable", "perception_loss", "incorrect"],
                            "ours": cats_o.tolist(), "open": cats_p.tolist(),
                            "ours_fail_share": [float(x) / (90 - cats_o[0]) for x in cats_o[1:]],
                            "open_fail_share": [float(x) / (90 - cats_p[0]) for x in cats_p[1:]]}
    # 시나리오 간 이질성 (ours, 과제별 3x2)
    het = {}
    rng = np.random.default_rng(0)
    for task, d in FIG12.items():
        for arm in ("ours", "open"):
            ks = [r[0] for r in d[arm]]
            tab = np.array([[k, 10 - k] for k in ks])
            chi2, p, _, _ = stats.chi2_contingency(tab, correction=False)
            # 정확 검정에 가까운 순열 검정 (행 주변합 10, 열 주변합 고정)
            pool = np.array([1] * sum(ks) + [0] * (30 - sum(ks)))
            obs = chi2
            cnt = 0
            B = 20000
            for _ in range(B):
                rng.shuffle(pool)
                t = np.array([[pool[:10].sum(), 10 - pool[:10].sum()], [pool[10:20].sum(), 10 - pool[10:20].sum()],
                              [pool[20:].sum(), 10 - pool[20:].sum()]])
                try:
                    c2 = stats.chi2_contingency(t, correction=False)[0]
                except ValueError:
                    continue
                cnt += c2 >= obs - 1e-12
            het[f"{task}_{arm}"] = {"ks": ks, "chi2": chi2, "p_chi2": p, "p_perm": cnt / B}
    out["scenario_heterogeneity"] = het
    return out


def block_self():
    out = {}
    for task, d in FIG13.items():
        k1, k2 = d["hand"], d["gripper"]
        out[task] = {"hand": [k1, 15, *wilson(k1, 15)], "gripper": [k2, 15, *wilson(k2, 15)],
                     "fisher_p": fisher(k1, 15, k2, 15), "diff_pp": [100 * x for x in newcombe_diff(k1, 15, k2, 15)]}
    kh = sum(d["hand"] for d in FIG13.values())
    kg = sum(d["gripper"] for d in FIG13.values())
    out["pooled"] = {"hand": [kh, 30, kh / 30, *wilson(kh, 30)], "gripper": [kg, 30, kg / 30, *wilson(kg, 30)],
                     "fisher_p": fisher(kh, 30, kg, 30), "diff_pp": [100 * x for x in newcombe_diff(kh, 30, kg, 30)]}
    # 같은 집계 19/30에서 검출 가능한 최소 차이 (80% 검정력, Fisher, 모의)
    rng = np.random.default_rng(1)

    def power(p1, p2, n, B=1500):
        hit = 0
        for _ in range(B):
            a = rng.binomial(n, p1)
            b = rng.binomial(n, p2)
            hit += fisher(a, n, b, n) < 0.05
        return hit / B

    base30, base15 = 19 / 30, 0.7
    out["power_fisher_n30_base63pct"] = {f"{dd:.2f}": power(base30, base30 - dd, 30) for dd in (0.2, 0.3, 0.4, 0.5)}
    out["power_fisher_n10_base80pct"] = {f"{dd:.2f}": power(0.8, 0.8 - dd, 10) for dd in (0.4, 0.5, 0.6, 0.7)}
    out["power_fisher_n15_base70pct"] = {f"{dd:.2f}": power(base15, base15 - dd, 15) for dd in (0.2, 0.3, 0.4, 0.5, 0.6)}
    return out


# ---------------------------------------------------------------------------
# 2. 데이터 스케일링 (그림 6, 판독) 과 부록 C.3 멱법칙
# ---------------------------------------------------------------------------
CLIPS = np.array([1000, 2000, 5000, 10000, 20000])
SUCC = np.array([1, 3, 12, 19, 22])          # 30회 중 (그림 판독, k/30 격자)
LOSS = np.array([3.175, 3.079, 2.978, 2.899, 2.815])   # 그림 판독 (±0.01)


def logistic_fit(x, k, n):
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(50):
        eta = X @ b
        p = 1 / (1 + np.exp(-eta))
        W = n * p * (1 - p)
        g = X.T @ (k - n * p)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, g)
        b += step
        if np.max(abs(step)) < 1e-10:
            break
    cov = np.linalg.inv(H)
    return b, np.sqrt(np.diag(cov))


def block_scaling():
    out = {"clips": CLIPS.tolist(), "succ_k_of_30": SUCC.tolist(), "loss_read": LOSS.tolist()}
    out["wilson"] = [[float(k) / 30, *wilson(int(k), 30)] for k in SUCC]
    pairs = {}
    for i in range(4):
        pairs[f"{CLIPS[i]}_vs_{CLIPS[i+1]}"] = fisher(int(SUCC[i + 1]), 30, int(SUCC[i]), 30)
    pairs["1000_vs_20000"] = fisher(int(SUCC[4]), 30, int(SUCC[0]), 30)
    pairs["5000_vs_20000"] = fisher(int(SUCC[4]), 30, int(SUCC[2]), 30)
    out["fisher_adjacent"] = pairs
    # Cochran-Armitage (log10 clips 점수)
    s = np.log10(CLIPS)
    n = np.full(5, 30.0)
    k = SUCC.astype(float)
    N = n.sum()
    pbar = k.sum() / N
    T = np.sum(s * (k - n * pbar))
    V = pbar * (1 - pbar) * (np.sum(n * s ** 2) - np.sum(n * s) ** 2 / N)
    z = T / math.sqrt(V)
    out["cochran_armitage"] = {"z": z, "p_two_sided": 2 * stats.norm.sf(abs(z))}
    b, se = logistic_fit(s, k, n)
    out["logistic_on_log10_clips"] = {"intercept": b[0], "slope_per_decade": b[1], "slope_se": se[1],
                                      "slope_ci95": [b[1] - Z * se[1], b[1] + Z * se[1]],
                                      "clips_at_50pct": float(10 ** (-b[0] / b[1]))}
    # 그림 12 binning 20k (21/30) 와 그림 6 20k (22/30) 의 차이
    out["fig6_20k_vs_fig12_binning"] = {"fig6": 22, "fig12": 21, "fisher_p_if_independent_runs": fisher(22, 30, 21, 30),
                                        "note": "같은 30회라면 그림 6 20k 점은 22/30(73.3%), 그림 12·그림 5의 binning은 21/30(70.0%)"}
    # 손실: 로그-선형과 고정 지수 멱법칙 (부록 C.3)
    A = np.column_stack([np.ones(5), s])
    coef, res, *_ = np.linalg.lstsq(A, LOSS, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((LOSS - pred) ** 2))
    ss_tot = float(np.sum((LOSS - LOSS.mean()) ** 2))
    out["loss_loglinear"] = {"intercept": coef[0], "slope_per_decade": coef[1], "r2": 1 - ss_res / ss_tot,
                             "extrap_at_20M": float(coef[0] + coef[1] * math.log10(2e7)),
                             "extrap_at_20k": float(coef[0] + coef[1] * math.log10(2e4))}
    fits = {}
    for alpha in (0.05, 0.10, 0.15, 0.20, 0.25):
        Xa = np.column_stack([np.ones(5), CLIPS.astype(float) ** (-alpha)])
        (c, a), *_ = np.linalg.lstsq(Xa, LOSS, rcond=None)
        r = LOSS - Xa @ np.array([c, a])
        fits[str(alpha)] = {"c": float(c), "a": float(a), "sse": float(np.sum(r ** 2)),
                            "max_abs_resid": float(np.max(abs(r))),
                            "pred_20M": float(c + a * (2e7) ** (-alpha)),
                            "asymptote_c": float(c)}
    out["powerlaw_fixed_alpha"] = fits
    # 손실-성공 관계: 성공 50% 가 되는 손실 (로지스틱 on 손실)
    bl, sel = logistic_fit(LOSS, k, n)
    out["logistic_on_loss"] = {"slope_per_loss_unit": bl[1], "slope_se": sel[1],
                               "loss_at_50pct": float(-bl[0] / bl[1])}
    out["loss_total_drop_1k_to_20k"] = float(LOSS[0] - LOSS[-1])
    out["loss_relative_drop_pct"] = float(100 * (LOSS[0] - LOSS[-1]) / LOSS[0])
    return out


# ---------------------------------------------------------------------------
# 3. 부록 표 값에서 유도되는 양 (일관성 검사)
# ---------------------------------------------------------------------------
def block_tables():
    out = {}
    # 표 1, 부록 A: 예측 지평
    T, hz = 8, 8.0
    out["horizon_s"] = {"T": T, "ref_rate_hz": hz, "horizon": T / hz, "paper": "1 s"}
    out["window_frames"] = {"F": 2, "T": 8, "total_frames": 10, "span_s_at_8hz": 10 / 8.0,
                            "stride_frames": 2, "stride_s": 2 / 8.0}
    # 표 3: 참조 궤적 길이
    seg_min = 2.5 + 0.0 + 3.0 + 2.5
    seg_max = 2.5 + 7.5 + 3.0 + 2.5
    out["episode_length_s"] = {"min": seg_min, "max": seg_max, "steps_at_50hz": [seg_min * 50, seg_max * 50]}
    # 행동 차원 (B.2, 표 4, 표 7)
    out["action_dim"] = {"arm": 6, "hand": 16, "eig": 5, "sum": 6 + 16 + 5, "table4_prev_action": 27, "table7_mu": 27}
    # 행동 적분의 최대 속도 (B.2): q_target 이 한 스텝에 움직이는 최대량 = alpha * eta * |a| (|a| <= 1, tanh)
    eta, a_arm, a_hnd, hz_ctl = 0.1, 0.15, 0.75, 50
    out["max_target_speed"] = {"arm_rad_per_step": a_arm * eta, "arm_rad_per_s": a_arm * eta * hz_ctl,
                               "hand_rad_per_step_per_channel": a_hnd * eta, "hand_rad_per_s_per_channel": a_hnd * eta * hz_ctl,
                               "note": "tanh 출력 |a|<=1일 때 관절별 목표 이동 상한. 손은 eigen 항 E a_eig + a_hnd 의 합이라 채널당 상한은 더 클 수 있음"}
    # 학습 예산 (표 8, 7절 B.5, 그림 7)
    t_env, t_h, t_mb = 20480, 32, 32768
    s_env, s_h, s_mb = 2048, 16, 8192
    out["teacher_budget"] = {"steps_per_iter": t_env * t_h, "minibatches_per_iter": t_env * t_h / t_mb,
                             "iters_for_1B_steps": 1e9 / (t_env * t_h), "grad_steps_per_iter": 4 * t_env * t_h / t_mb}
    out["student_budget"] = {"steps_per_iter": s_env * s_h, "minibatches_per_iter": s_env * s_h / s_mb,
                             "iters_for_100M_steps": 1e8 / (s_env * s_h),
                             "lambda_anneal_1000_iters_env_steps": 1000 * s_env * s_h,
                             "anneal_share_of_100M": 1000 * s_env * s_h / 1e8,
                             "note": "epoch = PPO 반복(iteration)이라고 가정. 논문은 'epoch'의 정의를 적지 않음"}
    # 표 4 관측 차원: 학생 (원본 표의 dim x history). N=16, T=8
    N, Tn = 16, 8
    student = {"prev_action": 27 * 8, "joint_pos": 22 * 8, "prev_target": 22 * 8, "eigen_proj": 5 * 8,
               "palm_pose": 7 * 8, "palm_lookahead": 7 * Tn, "points": 3 * N, "points_lookahead": 3 * N * Tn,
               "segment_progress": 1}
    out["student_obs_dims_N16_T8"] = {"non_image_total": sum(student.values()), "parts": student,
                                      "wrist_depth_pixels": 80 * 60}
    # 보상 가중치 (표 6): 양의 항 합과 종료 항
    pos = {"lookahead_tracking": 8.0, "trajectory_success": 25.0, "tracking_progress": 4.0, "palm_following": 3.5,
           "good_finger_contact": 1.5, "finger_manipulation": 6.0}
    out["reward_weights"] = {"positive_per_step_sum": sum(pos.values()), "timeout_bonus": 200.0, "early_termination": -100.0,
                             "contact_factor_max_leap": 1.0 * (2.0 + 1.8 + 1.4 + 0.4),
                             "contact_factor_max_gripper": 2.0 + 2.0,
                             "note": "c(f)=s_thumb*(2.0 s0+1.8 s1+1.4 s2+0.4 s_palm), 각 s는 [0,1]"}
    # 그림 7 (판독): 최종 에피소드 보상. 보상 가중치와 비교
    out["fig7_read_final_reward"] = {"teacher_ours": 67, "teacher_mlp": 27, "teacher_no_eigen": 13,
                                     "student_ours": 61, "student_dagger_bc": 47, "student_no_wrist": 42,
                                     "note": "표 6의 timeout 보너스(+200)보다 작다 -> 에피소드 보상이 정규화(스텝 평균 등)되었거나 대다수가 조기 종료"}
    out["fig7_ratios"] = {"teacher_ours_over_mlp": 67 / 27, "teacher_ours_over_no_eigen": 67 / 13,
                          "student_ours_over_dagger": 61 / 47, "student_ours_over_no_wrist": 61 / 42}
    # 배포 (표 11): 의도 모델 1 Hz, 지평 1 s
    out["deploy"] = {"intent_hz": 1, "horizon_s": 1.0, "policy_hz": 50, "tracker_hz": 30,
                     "policy_steps_per_intent_cycle": 50, "veo_generation_s_per_rollout": [120, 180]}
    return out


def main():
    RES["web"] = block_web()
    RES["self_collected"] = block_self()
    RES["scaling"] = block_scaling()
    RES["tables"] = block_tables()
    # 요약 출력
    w = RES["web"]
    print("== 웹 과제 (그림 12) ==")
    for k, v in w["sum_check"].items():
        print(k, "합 일치" if v["match"] else "불일치", v["sum_of_scenarios"], v["reported_overall"])
    print("풀링", w["pooled"])
    for t, v in w["per_task"].items():
        print(t, f"ours {v['ours'][0]}/30 open {v['open'][0]}/30 Fisher p={v['fisher_p']:.2g} diff={v['diff_pp']}")
    for r in w["per_scenario"]:
        print(r)
    print("MH", w["mantel_haenszel_9_strata"])
    print("실패 분포", w["failure_tally"])
    print("이질성", json.dumps(w["scenario_heterogeneity"], indent=0)[:900])
    print("== 자체 수집 ==", json.dumps(RES["self_collected"], indent=0)[:1500])
    print("== 스케일링 ==", json.dumps(RES["scaling"], indent=0)[:3500])
    print("== 표 유도값 ==", json.dumps(RES["tables"], indent=0)[:3500])
    OUT.write_text(json.dumps(RES, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

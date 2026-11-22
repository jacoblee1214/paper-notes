"""WAM-OPD (arXiv:2608.22364 v2) 보고 수치 재계산과 일관성 검사 (numpy, scipy, 1초 미만).

입력은 논문 표 2, 3, 4, 그림 4의 값을 아래에 옮긴 것이다. 시행 수: 시뮬레이션 60, 실로봇 36.
결과: analysis/recompute_results.json
실행: python3 analysis/recompute.py
"""
import json
from math import comb, sqrt
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"

# ---- 논문 값 (표 2, 3) ----
T2 = {  # 과제: (Teacher, Student, WAM-OPD, Δ 보고)
    "handover_mic": (80.0, 1.7, 35.0, 33.3),
    "put_object_cabinet": (51.7, 3.3, 16.7, 13.3),
    "place_fan": (96.7, 21.7, 50.0, 28.3),
    "place_a2b_left": (90.0, 11.7, 53.3, 41.7),
    "place_a2b_right": (90.0, 16.7, 65.0, 48.3),
    "place_bread_basket": (88.3, 25.0, 60.0, 35.0),
    "place_cans_plasticbox": (100.0, 73.3, 100.0, 26.7),
    "stamp_seal": (96.7, 16.7, 81.7, 65.0),
    "rotate_qrcode": (88.3, 23.3, 58.3, 35.0),
    "place_container_plate": (98.3, 93.3, 98.3, 5.0),
    "move_pillbottle_pad": (98.3, 66.7, 85.0, 18.3),
    "place_object_stand": (96.7, 51.7, 85.0, 33.3),
}
T2_AVG = (89.6, 33.8, 65.7, 31.9)
T3 = {
    "put_gripper_parts": (36.1, 5.6, 8.3, 2.8),
    "shake_bottle": (94.4, 88.9, 94.4, 5.6),
    "insert_screwdriver_tape": (66.7, 52.8, 83.3, 30.6),
    "put_tape_platform": (80.6, 58.3, 72.2, 13.9),
}
T3_AVG = (69.4, 51.4, 64.6, 13.2)
# 그림 4 (A2B Right, Bread Basket, Cans) : action-only, video-only, joint. 본문 증가폭: action 20.0/33.3/20.0, video 43.3/26.7/21.7
FIG4 = {
    "place_a2b_right": (36.7, 60.0, 65.0),
    "place_bread_basket": (58.3, 51.7, 60.0),
    "place_cans_plasticbox": (93.3, 95.0, 100.0),
}
FIG4_TEXT_GAIN = {"action": (20.0, 33.3, 20.0), "video": (43.3, 26.7, 21.7), "best_single_margin": (5.0, 1.7, 5.0)}
# 표 4: Student(Flash-WAM), Joint/Teacher histories, Joint/Student histories
T4 = {
    "place_a2b_right": (16.7, 26.7, 65.0),
    "place_bread_basket": (25.0, 75.0, 60.0),
    "place_cans_plasticbox": (73.3, 100.0, 100.0),
}
# v1 (arXiv v1, 표 1): 12쌍(장면 6 x 잡음 은행 2). Released, WAM-OPD
V1 = {"handover_mic": (0, 7), "put_object_cabinet": (2, 4)}
# 저장소 문서(docs/deployment/REAL_ROBOT_DEPLOYMENT.md): place_a2b_left formal12
REPO_DOC_A2B_LEFT = (0, 5)
# 공개된 LingBot-VA 논문(arXiv:2601.21998) 부록 표 S1의 RoboTwin 2.0 Easy 성공률(%)
LINGBOT_EASY = {
    "handover_mic": 94, "put_object_cabinet": 85, "place_fan": 99, "place_a2b_left": 97,
    "place_a2b_right": 97, "place_bread_basket": 97, "place_cans_plasticbox": 100, "stamp_seal": 96,
    "rotate_qrcode": 96, "place_container_plate": 99, "move_pillbottle_pad": 99, "place_object_stand": 99,
}


def to_count(pct, n):
    """퍼센트(소수 첫째 자리 반올림)가 되는 정수 k 후보."""
    return [k for k in range(n + 1) if round(100 * k / n + 1e-9, 1) == round(pct, 1)]


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def exact_sign_p(c, b):
    """쌍체 이산 쌍 c(구조), b(퇴보)의 양측 정확 McNemar(이항) p."""
    m = c + b
    if m == 0:
        return 1.0
    k = min(c, b)
    p = sum(comb(m, i) for i in range(k + 1)) / 2 ** m * 2
    return min(1.0, p)


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def table(T, n, name):
    rows, feas = {}, {}
    for t, (te, st, op, d) in T.items():
        ks = [to_count(x, n) for x in (te, st, op)]
        feas[t] = [len(k) == 1 for k in ks]
        kt, ks_, ko = (k[0] for k in ks)
        rows[t] = dict(
            kT=kt, kS=ks_, kO=ko, n=n,
            delta_count_pct=round(100 * (ko - ks_) / n, 2),
            delta_reported=d,
            delta_ok=round(100 * (ko - ks_) / n, 1) == d,
            student_ci=[round(x, 1) for x in wilson(ks_, n)],
            opd_ci=[round(x, 1) for x in wilson(ko, n)],
            fisher_p=float(stats.fisher_exact([[ko, n - ko], [ks_, n - ks_]])[1]),
            rescues_min=max(0, ko - ks_),  # b=0일 때의 최소 구조 수
            paired_best_case_p=exact_sign_p(max(0, ko - ks_), 0) if ko != ks_ else 1.0,
            b_max=min(ks_, n - ko),
        )
        c_max = rows[t]["b_max"] + (ko - ks_)
        rows[t]["paired_worst_case_p"] = exact_sign_p(c_max, rows[t]["b_max"]) if c_max >= 0 else 1.0
    return rows, feas


def avg_check(rows, n, avg_rep):
    kT = sum(r["kT"] for r in rows.values())
    kS = sum(r["kS"] for r in rows.values())
    kO = sum(r["kO"] for r in rows.values())
    N = n * len(rows)
    a = [100 * kT / N, 100 * kS / N, 100 * kO / N, 100 * (kO - kS) / N]
    return dict(counts=[kT, kS, kO], N=N, avg=[round(x, 3) for x in a], avg_reported=list(avg_rep),
                ok=[round(x, 1) == r for x, r in zip(a, avg_rep)],
                avg_of_rounded=[round(float(np.mean([v[i] for v in (T2 if N == 720 else T3).values()])), 3) for i in range(3)])


def main():
    res = {}
    # ---- 표 2 ----
    r2, f2 = table(T2, 60, "t2")
    res["table2"] = dict(rows=r2, integer_feasible_all=all(all(v) for v in f2.values()))
    res["table2"]["avg"] = avg_check(r2, 60, T2_AVG)
    ps = [r["fisher_p"] for r in r2.values()]
    for (t, r), a in zip(r2.items(), holm(ps)):
        r["fisher_p_holm"] = a
    kS = res["table2"]["avg"]["counts"][1]
    kO = res["table2"]["avg"]["counts"][2]
    N = 720
    res["table2"]["pooled"] = dict(
        student_ci=[round(x, 1) for x in wilson(kS, N)], opd_ci=[round(x, 1) for x in wilson(kO, N)],
        fisher_p=float(stats.fisher_exact([[kO, N - kO], [kS, N - kS]])[1]),
        improved_tasks=sum(r["kO"] > r["kS"] for r in r2.values()),
        sign_test_p=2 * 0.5 ** 12,
        wilcoxon_p=float(stats.wilcoxon([r["kO"] - r["kS"] for r in r2.values()]).pvalue),
        task_cluster_bootstrap_ci=None,
    )
    d = np.array([100 * (r["kO"] - r["kS"]) / 60 for r in r2.values()])
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, 12, 12)].mean() for _ in range(20000)]
    res["table2"]["pooled"]["task_cluster_bootstrap_ci"] = [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)]
    res["table2"]["pooled"]["task_t_ci"] = [round(float(x), 1) for x in stats.t.interval(0.95, 11, loc=d.mean(), scale=stats.sem(d))]
    # 격차-이득
    gap = np.array([100 * (r["kT"] - r["kS"]) / 60 for r in r2.values()])
    gain = d
    res["table2"]["gap_gain"] = dict(
        pearson=float(stats.pearsonr(gap, gain)[0]), spearman=float(stats.spearmanr(gap, gain)[0]),
        spearman_p=float(stats.spearmanr(gap, gain)[1]),
        closure={t: round(float(g / gp), 3) for t, g, gp in zip(r2, gain, gap)},
        mean_closure=round(float(np.mean(gain / gap)), 3), pooled_closure=round(float(gain.sum() / gap.sum()), 3),
        teacher_below_student_or_equal=[t for t, r in r2.items() if r["kT"] <= r["kO"]],
        opd_equals_teacher=[t for t, r in r2.items() if r["kT"] == r["kO"]],
        opd_above_teacher=[t for t, r in r2.items() if r["kO"] > r["kT"]],
    )
    # 단일 과제 검출 한계(80% 검정력, 비쌍체, 학생 p0 기준): 필요한 차이
    def mde(p0, n, alpha=0.05, power=0.8):
        za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
        lo, hi = 0.0, 1 - p0
        for _ in range(60):
            mid = (lo + hi) / 2
            p1 = p0 + mid
            nn = ((za * sqrt(2 * (p0 + mid / 2) * (1 - p0 - mid / 2)) + zb * sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2) / mid ** 2
            if nn > n:
                lo = mid
            else:
                hi = mid
        return 100 * hi
    res["detection_limits_pp"] = {
        "sim_task_n60_p0_0.5": round(mde(0.5, 60), 1), "sim_task_n60_p0_0.2": round(mde(0.2, 60), 1),
        "sim_pooled_n720_p0_0.34": round(mde(0.338, 720), 1), "real_task_n36_p0_0.5": round(mde(0.5, 36), 1),
        "real_pooled_n144_p0_0.51": round(mde(0.514, 144), 1),
        "ablation_n60_p0_0.6": round(mde(0.6, 60), 1),
    }
    # 표 2 Teacher 대 공개 LingBot-VA Easy
    lb = {}
    for t, r in r2.items():
        pub = LINGBOT_EASY[t] / 100
        lb[t] = dict(teacher_k=r["kT"], teacher_pct=round(100 * r["kT"] / 60, 1), published_pct=LINGBOT_EASY[t],
                     diff_pp=round(100 * r["kT"] / 60 - LINGBOT_EASY[t], 1),
                     binom_lower_p=float(stats.binomtest(r["kT"], 60, pub, alternative="less").pvalue))
    res["teacher_vs_published_lingbot"] = dict(rows=lb, published_mean=round(float(np.mean(list(LINGBOT_EASY.values()))), 2),
                                               teacher_mean=round(res["table2"]["avg"]["avg"][0], 2))

    # ---- 표 3 ----
    r3, f3 = table(T3, 36, "t3")
    res["table3"] = dict(rows=r3, integer_feasible_all=all(all(v) for v in f3.values()))
    res["table3"]["avg"] = avg_check(r3, 36, T3_AVG)
    ps3 = [r["fisher_p"] for r in r3.values()]
    for (t, r), a in zip(r3.items(), holm(ps3)):
        r["fisher_p_holm"] = a
    kS3, kO3 = res["table3"]["avg"]["counts"][1], res["table3"]["avg"]["counts"][2]
    # 층화(Cochran-Mantel-Haenszel) 검정
    tabs = [np.array([[r["kO"], 36 - r["kO"]], [r["kS"], 36 - r["kS"]]]) for r in r3.values()]
    num = sum(t[0, 0] - t[0].sum() * t[:, 0].sum() / t.sum() for t in tabs)
    var = sum(t[0].sum() * t[1].sum() * t[:, 0].sum() * t[:, 1].sum() / (t.sum() ** 2 * (t.sum() - 1)) for t in tabs)
    cmh = (abs(num) - 0.5) ** 2 / var
    res["table3"]["pooled"] = dict(
        student_ci=[round(x, 1) for x in wilson(kS3, 144)], opd_ci=[round(x, 1) for x in wilson(kO3, 144)],
        fisher_p=float(stats.fisher_exact([[kO3, 144 - kO3], [kS3, 144 - kS3]])[1]),
        cmh_chi2=float(cmh), cmh_p=float(stats.chi2.sf(cmh, 1)),
        gain_pp=round(100 * (kO3 - kS3) / 144, 2),
        two_prop_ci_pp=[round(100 * x, 1) for x in (
            (kO3 / 144 - kS3 / 144) - 1.96 * sqrt(kO3 / 144 * (1 - kO3 / 144) / 144 + kS3 / 144 * (1 - kS3 / 144) / 144),
            (kO3 / 144 - kS3 / 144) + 1.96 * sqrt(kO3 / 144 * (1 - kO3 / 144) / 144 + kS3 / 144 * (1 - kS3 / 144) / 144))],
    )
    # 실로봇 대 시뮬레이션 총 이득 비교(합산 상대 위험)
    # 실로봇 개선 3/4 과제 유의하지 않음
    res["table3"]["sign_test_4_of_4_p"] = 2 * 0.5 ** 4

    # ---- 그림 4 ----
    f4 = {}
    for i, (t, (a, v, j)) in enumerate(FIG4.items()):
        s = T2[t][1]
        ka, kv, kj, ks_ = (to_count(x, 60)[0] for x in (a, v, j, s))
        best = max(ka, kv)
        f4[t] = dict(
            counts=dict(action=ka, video=kv, joint=kj, student=ks_),
            joint_minus_best_single_pp=round(100 * (kj - best) / 60, 1),
            text_margin=FIG4_TEXT_GAIN["best_single_margin"][i],
            margin_ok=round(100 * (kj - best) / 60, 1) == FIG4_TEXT_GAIN["best_single_margin"][i],
            action_gain_pp=round(100 * (ka - ks_) / 60, 1), action_gain_text=FIG4_TEXT_GAIN["action"][i],
            video_gain_pp=round(100 * (kv - ks_) / 60, 1), video_gain_text=FIG4_TEXT_GAIN["video"][i],
            joint_vs_best_fisher_p=float(stats.fisher_exact([[kj, 60 - kj], [best, 60 - best]])[1]),
            joint_vs_best_paired_best_case_p=exact_sign_p(kj - best, 0) if kj > best else 1.0,
            video_vs_action_fisher_p=float(stats.fisher_exact([[kv, 60 - kv], [ka, 60 - ka]])[1]),
            joint_equals_table2=to_count(T2[t][2], 60)[0] == kj,
            joint_equals_table4_student_hist=to_count(T4[t][2], 60)[0] == kj,
            student_equals_table4=to_count(T4[t][0], 60)[0] == ks_,
        )
    res["fig4"] = f4
    # 합산: joint 대 최고 단일(3과제 합)
    kj = sum(v["counts"]["joint"] for v in f4.values())
    kb = sum(max(v["counts"]["action"], v["counts"]["video"]) for v in f4.values())
    res["fig4_pooled"] = dict(joint=kj, best_single=kb, N=180, diff_pp=round(100 * (kj - kb) / 180, 2),
                              fisher_p=float(stats.fisher_exact([[kj, 180 - kj], [kb, 180 - kb]])[1]),
                              note="최고 단일은 과제별 사후 선택이라 편향(유리한 쪽 선택)이 있다")
    # 필요한 시행 수: 60% 부근에서 5pp 차이를 80% 검정력으로
    za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
    p0, p1 = 0.60, 0.65
    n_need = ((za * sqrt(2 * 0.625 * 0.375) + zb * sqrt(p0 * 0.4 + p1 * 0.35)) ** 2) / 0.05 ** 2
    res["fig4_n_needed_5pp_at_60"] = round(n_need)

    # ---- 표 4 ----
    t4 = {}
    for t, (s, th, sh) in T4.items():
        ks_, kt, kh = (to_count(x, 60)[0] for x in (s, th, sh))
        t4[t] = dict(student=ks_, teacher_hist=kt, student_hist=kh,
                     fisher_student_vs_teacher_hist=float(stats.fisher_exact([[kh, 60 - kh], [kt, 60 - kt]])[1]),
                     diff_pp=round(100 * (kh - kt) / 60, 1),
                     teacher_hist_vs_flash_fisher=float(stats.fisher_exact([[kt, 60 - kt], [ks_, 60 - ks_]])[1]))
    res["table4"] = t4

    # ---- v1 대 v2, 저장소 문서 ----
    v = {}
    for t, (kr, ko) in V1.items():
        k2s, k2o = r2[t]["kS"], r2[t]["kO"]
        v[t] = dict(v1_released=kr, v1_opd=ko, v1_n=12, v2_student=k2s, v2_opd=k2o, v2_n=60,
                    v1_opd_pct=round(100 * ko / 12, 1), v2_opd_pct=round(100 * k2o / 60, 1),
                    ext48_released_if_v1_pairs_in_v2=k2s - kr, ext48_opd_if_same_adapter=k2o - ko,
                    fisher_v1_opd_vs_v2_opd=float(stats.fisher_exact([[ko, 12 - ko], [k2o, 60 - k2o]])[1]),
                    v1_train_traj=8, v2_train_traj=24)
    res["v1_vs_v2"] = v
    kr, ko = REPO_DOC_A2B_LEFT
    res["a2b_left_repo_doc_vs_paper"] = dict(
        formal12_released=kr, formal12_adapted=ko, paper_student=r2["place_a2b_left"]["kS"], paper_opd=r2["place_a2b_left"]["kO"],
        ext48_released=r2["place_a2b_left"]["kS"] - kr, ext48_adapted=r2["place_a2b_left"]["kO"] - ko,
        ext48_released_pct=round(100 * (r2["place_a2b_left"]["kS"] - kr) / 48, 1),
        ext48_adapted_pct=round(100 * (r2["place_a2b_left"]["kO"] - ko) / 48, 1),
        formal12_adapted_pct=round(100 * ko / 12, 1),
        fisher_formal12_vs_ext48_adapted=float(stats.fisher_exact([[ko, 12 - ko], [r2["place_a2b_left"]["kO"] - ko, 48 - (r2["place_a2b_left"]["kO"] - ko)]])[1]),
        note="docs/deployment/REAL_ROBOT_DEPLOYMENT.md의 5/12와 논문 표 2를 formal12 + extension48 가정으로 맞춘 것")

    # ---- 저장소 훈련 규모 ----
    res["train_scale"] = dict(
        trajectories=dict(train=24, calibration=12, screening=4, heldout_formal=6),
        a2b_left_optimizer_steps_doc=219, per_epoch_steps=73, labels_per_epoch_est=73 * 4,
        labels_per_train_traj_est=round(73 * 4 / 24, 1),
        lora_trainable_params_doc=20.15e6,
    )
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    # 요약 출력
    print("표 2 정수 가능:", res["table2"]["integer_feasible_all"], "평균 일치:", res["table2"]["avg"]["ok"], res["table2"]["avg"]["counts"])
    print("표 3 정수 가능:", res["table3"]["integer_feasible_all"], "평균 일치:", res["table3"]["avg"]["ok"], res["table3"]["avg"]["counts"])
    print("Δ 일치(표 2):", {t: r["delta_ok"] for t, r in r2.items() if not r["delta_ok"]}, "(불일치만)")
    print("Δ 일치(표 3):", {t: (r["delta_count_pct"], r["delta_reported"]) for t, r in r3.items() if not r["delta_ok"]})
    print("pooled2", res["table2"]["pooled"])
    print("pooled3", res["table3"]["pooled"])
    print("gap", res["table2"]["gap_gain"])
    print("det", res["detection_limits_pp"])
    print("fig4", json.dumps(res["fig4"], indent=0)[:1500])
    print("pooled4", res["fig4_pooled"], res["fig4_n_needed_5pp_at_60"])
    print("t4", t4)
    print("v1v2", v)
    print("a2b", res["a2b_left_repo_doc_vs_paper"])
    print("teacher", {t: (x["diff_pp"], round(x["binom_lower_p"], 4)) for t, x in lb.items()}, res["teacher_vs_published_lingbot"]["published_mean"], res["teacher_vs_published_lingbot"]["teacher_mean"])


if __name__ == "__main__":
    main()

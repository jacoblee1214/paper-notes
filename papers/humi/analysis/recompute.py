"""HuMI(arXiv:2602.06643 v3) 보고 수치 재계산.

논문 본문·부록에 적힌 성공 횟수, 퍼센트, 시간, 하이퍼파라미터에서 계산할 수 있는 것을 다시 계산한다.
결과는 analysis/recompute_results.json 에 저장한다. 수 초 안에 끝나고 numpy, scipy만 쓴다.

실행: python3 analysis/recompute.py
"""
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def clopper(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def fisher(k1, n1, k2, n2):
    return stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def newcombe(k1, n1, k2, n2):
    """두 비율 차(p1-p2)의 Newcombe-Wilson 95% 구간."""
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    return (p1 - p2 - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            p1 - p2 + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def power_two_prop(p1, p2, n1, n2, alpha=0.05, sims=20000, seed=0):
    """Fisher 정확 검정의 검정력을 모의로 추정한다."""
    rng = np.random.default_rng(seed)
    hit = 0
    cache = {}
    for _ in range(sims):
        a = rng.binomial(n1, p1)
        b = rng.binomial(n2, p2)
        key = (a, b)
        if key not in cache:
            cache[key] = fisher(a, n1, b, n2)
        hit += cache[key] < alpha
    return hit / sims


# ---------------------------------------------------------------- K1 보고된 성공 횟수와 퍼센트
cells = {
    # 이름: (성공, 시도, 논문 퍼센트, 위치)
    "knee_humi": (17, 20, 85, "§4.1.1, 그림 6c"),
    "knee_no_hil": (1, 10, 10, "§4.1.1"),
    "knee_ee_only": (0, 10, 0, "§4.1.1"),
    "sword_humi": (17, 20, 85, "§4.1.2"),
    "sword_no_varspeed": (5, 10, 50, "§4.1.2"),
    "toss_humi": (15, 20, 75, "§4.1.3"),
    "toss_executed_ref": (4, 10, 40, "§4.1.3"),
    "walk_ee_pelvis": (15, 20, 75, "§4.1.4"),
    "walk_ee_only": (6, 10, 60, "§4.1.4"),
    "walk_ee_pelvis_feet": (7, 10, 70, "§4.1.4"),
    "walk_absolute_pelvis": (0, 10, 0, "§4.1.4"),
    "squat_gen_unseen": (14, 20, 70, "§4.2"),
    "squat_in_domain_C1": (17, 20, 85, "부록 C.1"),
    "squat_no_adaptive_ee": (5, 10, 50, "부록 C.1"),
}
k1 = {}
for name, (k, n, pct, where) in cells.items():
    lo, hi = wilson(k, n)
    clo, chi = clopper(k, n)
    k1[name] = dict(k=k, n=n, reported_pct=pct, computed_pct=round(100 * k / n, 2), match=abs(100 * k / n - pct) < 0.5,
                    wilson=[round(lo, 4), round(hi, 4)], clopper=[round(clo, 4), round(chi, 4)], where=where)
res["K1_cells"] = k1

# ---------------------------------------------------------------- K2 절제 대 본 방법 (Fisher, Holm)
pairs = [
    ("knee: 인간 개입 없음", "knee_humi", "knee_no_hil"),
    ("knee: 전신 명세 없음(EE-only)", "knee_humi", "knee_ee_only"),
    ("sword: 가변 속도 없음", "sword_humi", "sword_no_varspeed"),
    ("toss: 실행 자세 참조", "toss_humi", "toss_executed_ref"),
    ("walk: EE-only", "walk_ee_pelvis", "walk_ee_only"),
    ("walk: EE+pelvis+feet", "walk_ee_pelvis", "walk_ee_pelvis_feet"),
    ("walk: 절대 골반 추적", "walk_ee_pelvis", "walk_absolute_pelvis"),
    ("squat: 적응형 EE 보상 없음", "squat_in_domain_C1", "squat_no_adaptive_ee"),
]
rows = []
for label, a, b in pairs:
    ka, na = cells[a][:2]
    kb, nb = cells[b][:2]
    p = fisher(ka, na, kb, nb)
    lo, hi = newcombe(ka, na, kb, nb)
    rows.append(dict(label=label, humi=f"{ka}/{na}", ablation=f"{kb}/{nb}", diff_pp=round(100 * (ka / na - kb / nb), 1),
                     diff_ci_pp=[round(100 * lo, 1), round(100 * hi, 1)], fisher_p=p))
adj = holm([r["fisher_p"] for r in rows if "EE+pelvis+feet" not in r["label"]])
j = 0
for r in rows:
    if "EE+pelvis+feet" in r["label"]:
        r["holm_p"] = None  # 이 비교는 '차이 없음'을 말하려는 것이라 Holm 묶음에서 뺀다
    else:
        r["holm_p"] = adj[j]
        j += 1
res["K2_ablations"] = rows

# ---------------------------------------------------------------- K3 키포인트 인터페이스 사이의 비교 ('동등' 서술 검토)
kp = []
for label, (a, b) in {
    "EE+pelvis 대 EE+pelvis+feet": ("walk_ee_pelvis", "walk_ee_pelvis_feet"),
    "EE-only 대 EE+pelvis": ("walk_ee_only", "walk_ee_pelvis"),
    "EE-only 대 EE+pelvis+feet": ("walk_ee_only", "walk_ee_pelvis_feet"),
}.items():
    ka, na = cells[a][:2]
    kb, nb = cells[b][:2]
    lo, hi = newcombe(ka, na, kb, nb)
    kp.append(dict(label=label, a=f"{ka}/{na}", b=f"{kb}/{nb}", fisher_p=fisher(ka, na, kb, nb),
                   diff_ci_pp=[round(100 * lo, 1), round(100 * hi, 1)]))
res["K3_keypoint_interfaces"] = kp
# 15/20 대 10/10 시도에서 75% 대 60%를 검출할 검정력
res["K3_power_75_vs_60_n20_n10"] = power_two_prop(0.75, 0.60, 20, 10)
res["K3_power_75_vs_60_n20_n20"] = power_two_prop(0.75, 0.60, 20, 20)
res["K3_power_85_vs_50_n20_n10"] = power_two_prop(0.85, 0.50, 20, 10)
# 75% 대 60%를 80% 검정력으로 검출하는 데 필요한 팔당 시도 수(대략)
need = None
for n in range(20, 400, 5):
    if power_two_prop(0.75, 0.60, n, n, sims=2000, seed=1) >= 0.8:
        need = n
        break
res["K3_n_per_arm_for_80pct_power_75_vs_60"] = need

# ---------------------------------------------------------------- K4 일반화 70% 와 정의역 내 성공률
k, n = 14, 20
lo, hi = wilson(k, n)
gen = dict(wilson=[round(lo, 3), round(hi, 3)], clopper=[round(x, 3) for x in clopper(k, n)])
gen["vs_in_domain_17_20_fisher_p"] = fisher(14, 20, 17, 20)
gen["vs_in_domain_diff_ci_pp"] = [round(100 * x, 1) for x in newcombe(14, 20, 17, 20)]
# 환경 4곳 x 5회 군집. 환경 간 상관(ICC)을 가정해 설계효과로 구간을 넓힌다.
for icc in (0.0, 0.1, 0.3):
    deff = 1 + (5 - 1) * icc
    n_eff = n / deff
    lo, hi = wilson(round(0.7 * n_eff), max(1, round(n_eff)))
    gen[f"icc_{icc}_n_eff"] = round(n_eff, 1)
    gen[f"icc_{icc}_wilson"] = [round(lo, 3), round(hi, 3)]
res["K4_generalization"] = gen

# ---------------------------------------------------------------- K5 데이터 수집 효율 정수 사슬
d = {}
d["v3_valid_demos"] = dict(humi=60, twist2=18, ratio=round(60 / 18, 3))
d["v1v2_episodes"] = dict(humi=62, twist2=28, ratio_raw=round(62 / 28, 3))
d["acceptance_v1v2_reported"] = dict(humi=96.7, twist2=64.3)
d["acceptance_recomputed_60_62"] = round(100 * 60 / 62, 3)
d["acceptance_recomputed_18_28"] = round(100 * 18 / 28, 3)
d["acceptance_29_30"] = round(100 * 29 / 30, 3)
d["acceptance_note"] = "96.7%는 60/62=96.77%를 올림하면 96.8%라 맞지 않고 내림(절단)이거나 다른 분모. 64.3%는 18/28과 일치"
d["time_per_valid_demo_ratio"] = round((15 / 60) / (15 / 18), 4)
d["time_per_valid_demo_reported_pct"] = 30.0
d["humi_valid_per_min"] = 60 / 15
d["twist2_valid_per_min"] = 18 / 15
d["main_exp_100_demos_minutes_at_session_rate"] = round(100 / (60 / 15), 2)
d["main_exp_reported_minutes"] = 22
d["main_exp_implied_rate_per_min"] = round(100 / 22, 3)
d["twist2_expected_demos_in_22min"] = round(22 * 18 / 15, 1)
d["bimanual_pnp_100demos"] = dict(humi_min=6.3, twist2_min=18.5, ratio=round(18.5 / 6.3, 3), humi_per_min=round(100 / 6.3, 2))
d["knee_50_demos_15min_seconds_per_demo"] = 15 * 60 / 50
d["knee_twist2"] = 0
# 60 대 18 의 이항 검정(한 세션 단위라 독립 시행이 아니다. 개수 수준의 참고값)
d["poisson_conditional_binom_p_60_of_78"] = stats.binomtest(60, 78, 0.5).pvalue
lo_p = stats.beta.ppf(0.025, 60, 19)
hi_p = stats.beta.ppf(0.975, 61, 18)
d["poisson_rate_ratio_exact_ci"] = [round(lo_p / (1 - lo_p), 2), round(hi_p / (1 - hi_p), 2)]
# 매칭 예산 비교(85% 대 50%): TWIST2 시도 수가 안 적혀 있어 경우별로 계산
mb = {}
for label, (n1, k1_, n2, k2_) in {"20대 10(17/20 대 5/10)": (20, 17, 10, 5), "20대 20(17/20 대 10/20)": (20, 17, 20, 10),
                                  "10대 10(9/10 대 5/10)": (10, 9, 10, 5)}.items():
    mb[label] = dict(fisher_p=fisher(k1_, n1, k2_, n2))
d["matched_budget_fisher"] = mb
res["K5_data_efficiency"] = d

# ---------------------------------------------------------------- K6 칼 끝 추적 오차 (그림 6b: 15.7±7.5, 21.2±6.4 mm)
te = {}
for label, (n1, n2) in {"성공 롤아웃만 17 대 5": (17, 5), "전체 20 대 10": (20, 10)}.items():
    # 막대 위 오차 막대의 뜻(SD, SE)이 안 적혀 있다. SD라고 가정한 Welch 검정
    t, p = stats.ttest_ind_from_stats(15.7, 7.5, n1, 21.2, 6.4, n2, equal_var=False)
    te[label + " (SD 가정)"] = dict(t=round(t, 3), p=round(p, 4))
    # SE라고 가정하면 SD = SE*sqrt(n)
    t2, p2 = stats.ttest_ind_from_stats(15.7, 7.5 * math.sqrt(n1), n1, 21.2, 6.4 * math.sqrt(n2), n2, equal_var=False)
    te[label + " (SE 가정)"] = dict(t=round(t2, 3), p=round(p2, 4))
te["cohen_d_sd_assumed"] = round((21.2 - 15.7) / math.sqrt((7.5 ** 2 + 6.4 ** 2) / 2), 3)
te["relative_sd_humi"] = round(7.5 / 15.7, 3)
res["K6_tracking_error"] = te

# ---------------------------------------------------------------- K7 IK 미리보기 효과
ik = {}
ik["feasible_without_preview_6_of_20_wilson"] = [round(x, 3) for x in wilson(6, 20)]
ik["infeasible_14_20_pct"] = 70.0
ik["hand_pelvis_distance_cm"] = dict(before=76.7, after=90.7, change_cm=round(90.7 - 76.7, 1), change_pct=round(100 * (90.7 / 76.7 - 1), 1))
ik["with_preview_100pct_n"] = "미기재 (그림 10a)"
res["K7_ik_preview"] = ik

# ---------------------------------------------------------------- K8 보상 모양 (σ 선택이 신호에 주는 영향)
rew = {}
errs = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
for name, sigma in {"전신 위치(σ=0.3 m)": 0.3, "EE 최대 허용(σ=0.1 m)": 0.1, "EE 최소 허용(σ=0.01 m)": 0.01}.items():
    rew[name] = {f"{e*100:g}cm": float(np.exp(-(e ** 2) / sigma ** 2)) for e in errs}


def sigma_of_speed(v, smin, smax, vmin=0.05, vmax=0.10):
    a = np.clip((v - vmin) / (vmax - vmin), 0, 1)
    return smin + (smax - smin) * a


rew["sigma_p(v) m"] = {f"{v:g}m/s": float(sigma_of_speed(v, 0.01, 0.1)) for v in (0.0, 0.05, 0.075, 0.10, 0.3)}
rew["sigma_theta(v) deg"] = {f"{v:g}m/s": float(sigma_of_speed(v, 5, 20)) for v in (0.0, 0.05, 0.075, 0.10, 0.3)}
# 5 cm 오차에서 전신(σ=0.3)이 주는 보상과 σ=0.01이 주는 보상
rew["note"] = "σ=0.3 m에서는 5 cm 오차의 보상이 0.97이라 센티미터 정밀도를 밀어 주지 못한다. σ=0.01 m에서는 3 cm부터 보상이 1e-4 미만이라 신호가 사라진다."
res["K8_reward_shape"] = rew

# ---------------------------------------------------------------- K9 가변 속도 분포
def trunc_normal_samples(mu, sd, lo, hi, size, rng):
    out = np.empty(size)
    filled = 0
    while filled < size:
        x = rng.normal(mu, sd, size * 2)
        x = x[(x >= lo) & (x <= hi)]
        take = min(len(x), size - filled)
        out[filled:filled + take] = x[:take]
        filled += take
    return out


rng = np.random.default_rng(0)
vs = {}
for sd in (1e-4, 0.3, 1.0):
    a, b = (0.25 - 1) / sd, (1.25 - 1) / sd
    dist = stats.truncnorm(a, b, loc=1.0, scale=sd)
    vs[f"sd={sd:g}"] = dict(mean_rate=round(float(dist.mean()), 4), sd_rate=round(float(dist.std()), 4),
                            p_rate_ge_0p9=round(float(1 - dist.cdf(0.9)), 4), p_rate_le_0p5=round(float(dist.cdf(0.5)), 4))
# 같은 sd=1.0 에서 속도를 0.02 s 마다 독립으로 뽑을 때와 0.25 s 마다 뽑아 보간할 때, 1초 구간 평균 속도의 산포
def window_avg_std(interval_s, horizon_s=1.0, dt=0.02, n_paths=4000, sd=1.0):
    steps = int(horizon_s / dt)
    knots = int(math.ceil(horizon_s / interval_s)) + 2
    t = np.arange(steps) * dt
    avgs = np.empty(n_paths)
    for i in range(n_paths):
        r = trunc_normal_samples(1.0, sd, 0.25, 1.25, knots, rng)
        avgs[i] = np.interp(t, np.arange(knots) * interval_s, r).mean()
    return float(avgs.mean()), float(avgs.std())


m02, s02 = window_avg_std(0.02)
m25, s25 = window_avg_std(0.25)
vs["window_1s_avg_rate_resample_0.02s"] = dict(mean=round(m02, 3), sd=round(s02, 3))
vs["window_1s_avg_rate_resample_0.25s_interp"] = dict(mean=round(m25, 3), sd=round(s25, 3))
vs["note"] = "논문 부록 E는 0.02 s마다 뽑는다고 쓰지만 공식 코드(playback_rate_delta=0.25)는 0.25 s 간격 격자를 선형 보간한다. 1초 구간 평균 속도의 산포가 달라진다."
res["K9_variable_speed"] = vs

# ---------------------------------------------------------------- K10 학습 예산
tb = {}
tb["teacher_iters"] = 50000
tb["steps_per_env"] = 24
tb["num_envs"] = 4096
tb["teacher_env_steps"] = 50000 * 24 * 4096
tb["student_env_steps"] = 5000 * 24 * 4096
tb["curriculum_iters_teacher"] = [240000 // 24, 360000 // 24]
tb["curriculum_frac_of_teacher_run"] = [round(240000 / 24 / 50000, 3), round(360000 / 24 / 50000, 3)]
tb["curriculum_iters_student_in_code"] = [24000 // 24, 36000 // 24]
tb["control_hz"] = 1 / (0.005 * 4)
tb["teacher_sim_hours_at_50Hz"] = round(tb["teacher_env_steps"] / 50 / 3600, 0)
res["K10_training_budget"] = tb

# ---------------------------------------------------------------- K11 시간 기준 일관성
tm = {}
tm["highlevel_action_horizon_s"] = 48 / 20
tm["client_chunk_length_s"] = 2.0
tm["waypoint_spacing_s"] = 2.0 / 10
tm["waypoint_spacing_control_steps"] = (2.0 / 10) / 0.02
tm["policy_period_s"] = 1 / 5
tm["history_proprio_span_s"] = 15 / 20
tm["lowlevel_history_s"] = 25 * 0.02
tm["dataset_hz"] = 59.94
tm["obs_hz_from_downsample3"] = 59.94 / 3
res["K11_timing"] = tm

# ---------------------------------------------------------------- K12 HF 공개 데이터와 논문 시연 수
hf = {"proposal(=single-knee)": (103, 100), "walk_clean_table": (105, 100), "unsheathe": (105, 100), "toss": (104, 100),
      "squat_pick_ground": (410, 350)}
res["K12_hf_dataset_counts"] = {k: dict(hf_episodes=v[0], paper=v[1], extra=v[0] - v[1]) for k, v in hf.items()}

# ---------------------------------------------------------------- K13 같은 17/20이 세 번
p17 = stats.binom.pmf(17, 20, 0.85)
res["K13_triple_17_of_20"] = dict(p_single_cell_equals_17_given_p085=round(float(p17), 4),
                                  p_three_cells_all_17_given_p085=round(float(p17 ** 3), 5),
                                  note="독립이고 참값 85%라면 가능한 일(낮은 증거). 본문이 같은 17/20을 세 과제에서 쓰므로 표기만 확인.")

OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
print("wrote", OUT)
for r in rows:
    print(f"{r['label']:<28} {r['humi']:>6} vs {r['ablation']:>6} p={r['fisher_p']:.4f} holm={r['holm_p']}")

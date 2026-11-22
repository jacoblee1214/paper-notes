"""H2O (arXiv:2403.04436) 논문 표 수치 재계산과 보상 커널 비교 (CPU, numpy와 scipy, 수 초).

내용
  V1  표 III: 성공 시퀀스 평균과 전체 평균에서 실패 시퀀스 평균을 역산 (반올림 구간 포함)
  V2  상태 차원 138, 90, 778 의 구성 (논문 식과 공식 코드의 관측 구성)
  V3  시퀀스 수 흐름 (13k, 10k, 8.5k)과 정수 가능성
  V4  표 III, IV 성공률의 Wilson 구간과 차이 검정 (N = 10000 가정, 시드 변동은 알 수 없음)
  V5  표 IV 데이터 크기 곡선 (로그 선형 적합, 0.1% 데이터의 시퀀스 수)
  V6  "필터링으로 지운 시퀀스의 성공률"에 따른 정리 집합 성공률 (혼합 항등식)
  V7  표 I 보상 가중치 합과 종료 벌점의 비율
  V8  표 I, II 와 공식 코드 설정 값의 수치 대조
  V9  보상 커널 토이: 같은 오차에서 논문 식과 코드 식의 보상 값 (두 해석)

실행: python3 analysis/recompute.py   → analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"
RES = {}

# ---------------------------------------------------------------- 논문 표 값
# 표 III (논문 쪽 6): All sequences, Successful sequences. 열: Succ, Eg-mpjpe, Empjpe, Eacc, Evel
T3 = {
    "Privileged policy": dict(succ=85.5, all=[50.0, 43.6, 6.9, 7.8], ok=[46.0, 40.9, 5.2, 6.2]),
    "H2O-reduced": dict(succ=53.2, all=[200.2, 115.8, 11.2, 13.8], ok=[182.5, 111.0, 3.0, 8.1]),
    "H2O-w/o-sim2data": dict(succ=67.9, all=[176.6, 95.0, 10.2, 12.2], ok=[163.1, 93.8, 3.0, 7.5]),
    "H2O": dict(succ=72.5, all=[166.7, 91.7, 8.9, 11.0], ok=[151.0, 88.8, 2.9, 7.0]),
}
METRICS = ["Eg-mpjpe", "Empjpe", "Eacc", "Evel"]
# 표 IV (논문 쪽 6): 평가는 uncleaned retargeted AMASS
T4 = [
    ("0.1%", 0.001, 52.0, [198.0, 107.9, 12.4, 13.7]),
    ("1%", 0.01, 58.8, [183.8, 96.4, 10.7, 12.0]),
    ("10%", 0.10, 61.3, [174.3, 92.3, 10.8, 12.1]),
    ("100%", 1.00, 72.5, [166.7, 91.7, 8.9, 11.0]),
]


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def newcombe(p1, n1, p2, n2, z=1.959964):
    """두 비율 차이 p1 - p2 의 Newcombe(Wilson 기반) 95% 구간 (단위 %p)."""
    l1, u1 = (v / 100 for v in wilson(p1 * n1, n1, z))
    l2, u2 = (v / 100 for v in wilson(p2 * n2, n2, z))
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return 100 * lo, 100 * hi


# ---------------------------------------------------------------- V1
v1 = {}
for name, r in T3.items():
    s = r["succ"] / 100
    rows = {}
    for i, m in enumerate(METRICS):
        a, o = r["all"][i], r["ok"][i]
        fail = (a - s * o) / (1 - s)
        # 반올림 범위: 표 값 ±0.05, 성공률 ±0.05pp
        lo, hi = 1e9, -1e9
        for da in (-0.05, 0.05):
            for do in (-0.05, 0.05):
                for ds in (-0.0005, 0.0005):
                    ss = s + ds
                    f = (a + da - ss * (o + do)) / (1 - ss)
                    lo, hi = min(lo, f), max(hi, f)
        rows[m] = dict(fail_mean=round(fail, 1), range=[round(lo, 1), round(hi, 1)],
                       ratio_fail_over_ok=round(fail / o, 2), fail_gt_ok=bool(lo > o))
    v1[name] = rows
RES["V1_implied_failure_means"] = v1

# ---------------------------------------------------------------- V2 차원
prop = 19 + 19 + 3 + 3 + 3 + 19          # q, qdot, v, w, g, a_{t-1}
goal_full = 3 * 8 * 3                     # p_kp, p_kp - p, pdot_kp (8 키포인트 x 3)
goal_reduced = 8 * 3                      # p_kp 만
n_bodies = 20                             # 공식 코드 H1 몸체 수 (extend 제외)
priv_self = 1 + n_bodies * (3 + 6 + 3 + 3) - 3     # phc/humanoid.py: 높이 + 몸체당 15 - 3
priv_task = n_bodies * 1 * 24                      # obs_v 6, numTrajSamples 1
RES["V2_dims"] = dict(proprio=prop, goal_full=goal_full, total_full=prop + goal_full,
                      goal_reduced=goal_reduced, total_reduced=prop + goal_reduced,
                      priv_self=priv_self, priv_task_1sample=priv_task, priv_total_1sample=priv_self + priv_task,
                      priv_total_3samples_as_shipped=priv_self + n_bodies * 3 * 24,
                      paper=dict(full=138, reduced=90, privileged=778),
                      code_readme=dict(H2O=138, H2O_privileged_critic=215))

# ---------------------------------------------------------------- V3 시퀀스 수
N = 10000
priv_ok = round(N * 0.855)
RES["V3_counts"] = dict(retargeted=N, privileged_success=priv_ok, removed=N - priv_ok,
                        paper_text="약 8.5k / 10k", amass_total_paper="13k, 40 h",
                        readme_retarget_log=15886,
                        note="N=10000이면 85.5%는 8,550개로 본문 8.5k와 맞음. 13k 대 15,886은 같은 집합이 아니거나 필터 단계가 다름")

# ---------------------------------------------------------------- V4 구간과 검정
rates = {"Privileged": 85.5, "H2O-reduced": 53.2, "H2O-w/o-sim2data": 67.9, "H2O": 72.5,
         "0.1%": 52.0, "1%": 58.8, "10%": 61.3}
v4 = {"ci": {}, "diff": {}}
for k, v in rates.items():
    lo, hi = wilson(v / 100 * N, N)
    v4["ci"][k] = dict(rate=v, lo=round(lo, 2), hi=round(hi, 2), half_width=round((hi - lo) / 2, 2))
for a, b in [("H2O", "H2O-w/o-sim2data"), ("H2O", "H2O-reduced"), ("H2O", "Privileged"),
             ("H2O", "10%"), ("10%", "1%"), ("1%", "0.1%"), ("H2O-reduced", "0.1%"),
             ("H2O-w/o-sim2data", "10%")]:
    pa, pb = rates[a] / 100, rates[b] / 100
    lo, hi = newcombe(pa, N, pb, N)
    # 독립 두 비율 z 검정 (같은 시퀀스를 공유하므로 쌍체가 더 좁음. 시드 변동은 포함 안 됨)
    pp = (pa + pb) / 2
    z = (pa - pb) / math.sqrt(2 * pp * (1 - pp) / N)
    v4["diff"][f"{a} - {b}"] = dict(diff_pp=round(100 * (pa - pb), 1), lo=round(lo, 2), hi=round(hi, 2),
                                    z=round(z, 1), p=float(2 * stats.norm.sf(abs(z))))
# 민감도: 평가 시퀀스 수가 1000, 5000 이어도 4.6%p 차이는 구분되는가
sens = {}
for n in (500, 1000, 2000, 5000, 10000):
    lo, hi = newcombe(0.725, n, 0.679, n)
    sens[n] = [round(lo, 2), round(hi, 2)]
v4["h2o_vs_wo_sim2data_by_N"] = sens
RES["V4_success_rate_inference"] = v4

# ---------------------------------------------------------------- V5 표 IV 곡선
x = np.log10([f for _, f, _, _ in T4])
y = np.array([s for _, _, s, _ in T4])
slope, icpt, r, p, se = stats.linregress(x, y)
n_seq = {lab: f * priv_ok for lab, f, _, _ in T4}
step = {f"{T4[i][0]} -> {T4[i+1][0]}": round(float(y[i + 1] - y[i]), 1) for i in range(3)}
RES["V5_dataset_size_curve"] = dict(
    seqs_train=n_seq, step_gain_pp=step, log10_slope_pp_per_decade=round(float(slope), 2),
    r2=round(float(r ** 2), 3),
    empjpe=[t[3][1] for t in T4], succ=[t[2] for t in T4],
    note="마지막 구간(10% -> 100%)이 +11.2%p로 가장 크고 중간 두 구간은 +6.8, +2.5. 로그 선형이 아니라 마지막에서 가속. "
         "0.1%는 학습 시퀀스 약 9개인데 H2O-reduced(전체 데이터, 축소 목표 상태)와 성공률이 같은 수준(52.0 대 53.2).",
    empjpe_flat="Empjpe는 10%(92.3)와 100%(91.7)가 사실상 같은데 성공률은 61.3에서 72.5로 +11.2%p. 성공률이 꼬리 시퀀스에 민감함을 시사")

# ---------------------------------------------------------------- V6 혼합 항등식
f_ok, f_rm = 0.855, 0.145
v6 = {}
for name in ("H2O", "H2O-reduced"):
    s_all = rates[name] / 100
    rows = {}
    for r_rm in (0.0, 0.1, 0.25, 0.5):
        s_ok = (s_all - f_rm * r_rm) / f_ok
        rows[f"removed_success={r_rm}"] = round(100 * s_ok, 1)
    v6[name] = rows
v6["note"] = ("정리 집합이 특권 정책이 성공한 시퀀스(85.5%)와 같다고 보면(필터와 표 III의 85.5%가 같은 평가라는 가정), "
              "특권 정책의 정리 집합 성공률은 정의상 100%입니다. H2O가 지운 시퀀스를 하나도 못 맞춘다고 해도 정리 집합 성공률은 "
              "84.8%이므로, 정리 집합에서도 최소 15%는 실패합니다. 지운 시퀀스에서 일부 성공하면 정리 집합 성공률은 더 낮아집니다.")
v6["max_clean_success_pp"] = round(100 * 0.725 / 0.855, 1)
RES["V6_mixture"] = v6

# ---------------------------------------------------------------- V7 보상 가중치
task_w = dict(dof_pos=24, dof_vel=24, body_pos=40, body_rot=16, body_vel=60, body_angvel=60)
RES["V7_reward_weights"] = dict(
    task_sum=sum(task_w.values()), termination=200, feet_air_time=800,
    termination_over_task_sum=round(200 / sum(task_w.values()), 3),
    note="종료 벌점 200은 한 스텝 최대 과제 보상 합(224)보다 작습니다. 공식 코드는 모든 보상 척도에 dt(0.02)를 곱하므로 "
         "(legged_robot.py _prepare_reward_function) 표의 가중치가 dt 곱 전인지 후인지는 논문에 없습니다.")

# ---------------------------------------------------------------- V8 설정 수치 대조
paper = dict(push_v=0.5, friction=(0.2, 1.1), delay_ms=(20, 60), term_dist=0.5, rfi=0.1)
code = dict(push_v=1.0, friction=(-0.6, 1.2), delay_steps=(0, 3), dt_policy_ms=20, term_dist=1.5, rfi=0.1)
RES["V8_config_numbers"] = dict(
    paper=paper, code=code,
    code_delay_ms_support=[0, 20, 40, 60],
    friction_range_len=dict(paper=round(1.1 - 0.2, 2), code=round(1.2 + 0.6, 2)),
    friction_negative_mass=round(0.6 / 1.8, 3),
    term_dist_ratio=code["term_dist"] / paper["term_dist"],
    push_ratio=code["push_v"] / paper["push_v"],
    note="마찰 범위 [-0.6, 1.2]에서 음수 구간은 33%. PhysX가 음수를 어떻게 처리하는지는 확인하지 못함(이슈 #39에도 질문).")

# ---------------------------------------------------------------- V9 보상 커널 토이
rng = np.random.default_rng(0)
# H1 관절 19개의 위치 오차가 모두 같은 크기 e (rad)인 경우와, 상체 4개 관절만 틀린 경우
w_pos = np.array([2.0] * 0 + [2.0, 0.5, 0.5, 0.5, 0.5] + [2.0, 0.5, 0.5, 0.5, 0.5] + [2.0] + [2.0] * 4 + [2.0] * 4)
# 코드 순서: 왼다리 5, 오른다리 5, 몸통 1, 왼팔 4, 오른팔 4 (cfg의 teleop_joint_pos_selection)
assert len(w_pos) == 19


def joint_kernel(e, mode):
    err = np.full(19, e)
    if mode == "paper_sum":        # exp(-0.25 * ||q^ - q||^2) , 19차원 합
        return float(np.exp(-0.25 * np.sum(err ** 2)))
    if mode == "paper_mean":       # 같은 계수를 평균제곱으로 읽은 경우
        return float(np.exp(-0.25 * np.mean(err ** 2)))
    if mode == "code":             # exp(-mean(w * err^2) / 0.5)  (관절별 가중, 코드 sigma = 0.5)
        return float(np.exp(-np.mean(w_pos * err ** 2) / 0.5))


def body_kernel(e, mode, n_bodies=22):
    """몸체 위치 오차: 각 몸체 xyz 한 축에 같은 오차 e(m) (3차원 노름 e)라고 둔다."""
    d2 = e ** 2                                   # 몸체당 ||.||^2
    if mode == "paper_sum":
        return float(np.exp(-0.5 * n_bodies * d2))   # exp(-0.5 ||p - p^||^2), 모든 몸체 합
    if mode == "paper_mean":
        return float(np.exp(-0.5 * d2))              # 몸체 평균
    if mode == "code_upper":                         # 상체 sigma 0.03, 좌표 평균(=||.||^2 / 3)
        return float(np.exp(-(d2 / 3) / 0.03))
    if mode == "code_lower":                         # 하체 sigma 0.5
        return float(np.exp(-(d2 / 3) / 0.5))


def vel_kernel(e, mode, n_bodies=22):
    d2 = e ** 2
    if mode == "paper_sum":
        return float(np.exp(-10.0 * n_bodies * d2))
    if mode == "paper_mean":
        return float(np.exp(-10.0 * d2))
    if mode == "code":                               # sigma 10, 좌표 평균
        return float(np.exp(-(d2 / 3) / 10.0))


toy = {"joint_pos": {}, "body_pos": {}, "body_vel": {}}
for e in (0.05, 0.1, 0.2, 0.4):
    toy["joint_pos"][str(e)] = {m: round(joint_kernel(e, m), 4) for m in ("paper_sum", "paper_mean", "code")}
for e in (0.02, 0.05, 0.1, 0.2, 0.5):
    toy["body_pos"][str(e)] = {m: round(body_kernel(e, m), 4)
                               for m in ("paper_sum", "paper_mean", "code_upper", "code_lower")}
for e in (0.05, 0.1, 0.3, 1.0):
    toy["body_vel"][str(e)] = {m: round(vel_kernel(e, m), 4) for m in ("paper_sum", "paper_mean", "code")}
# 보상이 0.5로 떨어지는 오차 (반감 오차)
half = {}
half["joint_pos_rad"] = dict(
    paper_sum=math.sqrt(math.log(2) / 0.25 / 19), paper_mean=math.sqrt(math.log(2) / 0.25),
    code=math.sqrt(math.log(2) * 0.5 / np.mean(w_pos)))
half["body_pos_m_per_body"] = dict(
    paper_sum=math.sqrt(math.log(2) / 0.5 / 22), paper_mean=math.sqrt(math.log(2) / 0.5),
    code_upper=math.sqrt(math.log(2) * 0.03 * 3), code_lower=math.sqrt(math.log(2) * 0.5 * 3))
half["body_vel_m_s_per_body"] = dict(
    paper_sum=math.sqrt(math.log(2) / 10.0 / 22), paper_mean=math.sqrt(math.log(2) / 10.0),
    code=math.sqrt(math.log(2) * 10.0 * 3))
half = {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in half.items()}
vel_ratio_sum = half["body_vel_m_s_per_body"]["code"] / half["body_vel_m_s_per_body"]["paper_sum"]
vel_ratio_mean = half["body_vel_m_s_per_body"]["code"] / half["body_vel_m_s_per_body"]["paper_mean"]
RES["V9_reward_kernel_toy"] = dict(
    values=toy, half_reward_error=half,
    vel_half_error_ratio_code_over_paper_sum=round(vel_ratio_sum, 1),
    vel_half_error_ratio_code_over_paper_mean=round(vel_ratio_mean, 1),
    note="논문 식의 노름을 몸체 합으로 읽으면 관절 위치(반감 오차 0.38 대 0.50 rad)와 상체 몸체 위치(0.251 대 0.25 m)는 코드와 "
         "같은 크기입니다. 속도 커널은 어느 해석으로도 코드가 17-81배 완만합니다(반감 오차 4.6 m/s). 코드의 속도 보상은 "
         "현실적인 오차 범위에서 거의 1에 붙어 있어 학습 신호가 약합니다.")

# ---------------------------------------------------------------- V10 시드 변동 민감도
# 논문에는 시드 수와 시드 간 변동이 없다. 학습 시드 SD(%p)를 가정하고 H2O 대 w/o-sim2data(+4.6%p)의 Welch 검정이 어떻게 되는지 본다.
v10 = {}
for sd in (1.0, 2.0, 3.0):
    for n in (3, 5):
        se = sd * math.sqrt(2.0 / n)
        t = 4.6 / se
        df = 2 * (n - 1)
        v10[f"sd={sd},n={n}"] = dict(t=round(t, 2), p=round(float(2 * stats.t.sf(t, df)), 4))
RES["V10_seed_sensitivity"] = dict(diff_pp=4.6, table=v10,
                                  note="시드 SD 1%p면 3시드에서도 p = 0.005, 2%p면 3시드 p = 0.048(경계), 3%p면 3시드 p = 0.13, 5시드 p = 0.04. 논문은 시드 수와 변동을 보고하지 않아 판단 불가.")

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------------------------------------------------------------- 출력
print("V1 implied failure means (Eg-mpjpe, Empjpe, Eacc, Evel)")
for name, rows in v1.items():
    print(f"  {name:20s}", [rows[m]["fail_mean"] for m in METRICS], "all fail>ok:", all(rows[m]["fail_gt_ok"] for m in METRICS))
print("V2", RES["V2_dims"])
print("V4 CI", {k: (v["lo"], v["hi"]) for k, v in v4["ci"].items()})
print("V4 diff", {k: (v["diff_pp"], v["lo"], v["hi"], v["p"]) for k, v in v4["diff"].items()})
print("V4 sens", sens)
print("V5", RES["V5_dataset_size_curve"])
print("V6", v6)
print("V10", v10)
print("V9 half", half, "vel ratios", vel_ratio_sum, vel_ratio_mean)

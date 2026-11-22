"""논문 표 수치를 옮겨 적고 (1) 표끼리의 일치, (2) 본문 문장과 표의 일치, (3) 검정과 구간을 다시 계산한다.

입력: 논문 PDF에서 옮긴 값(아래 상수). 원문은 paper/paper.pdf의 Fig. 13(전체 표), Fig. 10(실로봇 표), 본문 V-A, V-C, IX.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 1초 미만)

가정(논문에 명시되지 않음):
  - Fig. 13의 과제별 값은 평가 50회 중 성공 비율이다. 값이 모두 1/50의 배수이고 평균이 본문 평균과 맞으므로 과제당 50회로 본다.
  - 과제 사이 독립과 시드 1개를 가정한 이항 구간은 평가 잡음만 반영한다. 학습 시드 변동은 빠져 있다(하한).
  - Fig. 10의 "평균 ± 값"은 3개 시드의 표본 표준편차로 본다(논문이 정의하지 않음). 시드당 평가 횟수는 본문에 없다.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"
R = {}
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. Fig. 13 전체 표 (24개 원자 과제)
# 열: Human-50, Generated-100, Generated-300, Generated-3000 (성공 비율, 과제당 50회)
TASKS = [
    # (과제, 기술 묶음, H50, G100, G300, G3000)
    ("PickPlaceCabToCounter", "pnp", 0.02, 0.04, 0.10, 0.18),
    ("PickPlaceCounterToCab", "pnp", 0.06, 0.08, 0.16, 0.28),
    ("PickPlaceCounterToMicrowave", "pnp", 0.02, 0.00, 0.00, 0.18),
    ("PickPlaceCounterToSink", "pnp", 0.02, 0.02, 0.16, 0.44),
    ("PickPlaceCounterToStove", "pnp", 0.02, 0.00, 0.00, 0.06),
    ("PickPlaceMicrowaveToCounter", "pnp", 0.02, 0.00, 0.12, 0.08),
    ("PickPlaceSinkToCounter", "pnp", 0.08, 0.02, 0.14, 0.42),
    ("PickPlaceStoveToCounter", "pnp", 0.06, 0.00, 0.04, 0.28),
    ("OpenSingleDoor", "door", 0.46, 0.42, 0.44, 0.50),
    ("OpenDoubleDoor", "door", 0.28, 0.12, 0.22, 0.48),
    ("CloseDoubleDoor", "door", 0.28, 0.18, 0.62, 0.46),
    ("CloseSingleDoor", "door", 0.56, 0.82, 0.86, 0.94),
    ("OpenDrawer", "drawer", 0.42, 0.26, 0.40, 0.74),
    ("CloseDrawer", "drawer", 0.80, 0.92, 0.98, 0.96),
    ("TurnOnStove", "appliance", 0.32, 0.42, 0.44, 0.46),
    ("TurnOffStove", "appliance", 0.04, 0.08, 0.12, 0.24),
    ("TurnOnSinkFaucet", "appliance", 0.38, 0.26, 0.48, 0.34),
    ("TurnOffSinkFaucet", "appliance", 0.50, 0.48, 0.46, 0.72),
    ("TurnSinkSpout", "appliance", 0.54, 0.50, 0.58, 0.96),
    ("TurnOnMicrowave", "appliance", 0.62, 0.36, 0.76, 0.90),
    ("TurnOffMicrowave", "appliance", 0.70, 0.70, 0.62, 0.60),
    ("CoffeeSetupMug", "coffee", 0.00, 0.02, 0.04, 0.12),
    ("CoffeeServeMug", "coffee", 0.22, 0.12, 0.24, 0.34),
    ("CoffeePressButton", "coffee", 0.48, 0.48, 0.42, 0.74),
]
PAPER_AVG = {"H50": 0.288, "G100": 0.263, "G300": 0.350, "G3000": 0.476}  # Fig. 13 Average 행
SETTINGS = ["H50", "G100", "G300", "G3000"]
N_TRIALS = 50
names = [t[0] for t in TASKS]
groups = [t[1] for t in TASKS]
P = np.array([t[2:] for t in TASKS])  # (24, 4)
assert P.shape == (24, 4)
assert np.allclose(P * N_TRIALS, np.round(P * N_TRIALS)), "값이 1/50의 배수가 아님"

means = P.mean(axis=0)
R["fig13_column_means"] = dict(zip(SETTINGS, means.round(5).tolist()))
R["fig13_paper_average"] = PAPER_AVG
R["fig13_mean_matches_paper_rounded"] = {
    s: bool(round(m, 3) == PAPER_AVG[s] or abs(m - PAPER_AVG[s]) < 0.0005) for s, m in zip(SETTINGS, means)
}
R["fig13_total_successes"] = dict(zip(SETTINGS, (P.sum(axis=0) * N_TRIALS).round().astype(int).tolist()))
R["fig13_tasks"] = [dict(task=n, group=g, **{s_: float(v) for s_, v in zip(SETTINGS, row)})
                    for n, g, row in zip(names, groups, P)]
R["n_tasks"] = int(P.shape[0])
R["n_trials_total_per_setting"] = int(P.shape[0] * N_TRIALS)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# 과제별 Wilson 구간(성공 k = 50p) 중 평균 과제의 구간 근사: 과제 독립 가정, 분산 합 / 24^2
R["avg_se_binomial"] = {}
R["avg_ci_binomial"] = {}
for j, s in enumerate(SETTINGS):
    p = P[:, j]
    var = (p * (1 - p) / N_TRIALS).sum() / 24**2
    se = float(np.sqrt(var))
    R["avg_se_binomial"][s] = round(se, 4)
    R["avg_ci_binomial"][s] = [round(means[j] - 1.96 * se, 4), round(means[j] + 1.96 * se, 4)]

# 차이 검정 1: H50 대 G3000. 과제 독립 가정 두 평균 차이 (평가 잡음만)
d = means[3] - means[0]
se_d = np.sqrt(R["avg_se_binomial"]["H50"] ** 2 + R["avg_se_binomial"]["G3000"] ** 2)
R["diff_G3000_minus_H50"] = dict(diff=round(d, 4), se_eval_only=round(se_d, 4), z=round(d / se_d, 2))

# 차이 검정 2: 과제를 짝으로 본 대응 t 검정 (과제 간 변동 포함, 실제로 과제를 표본으로 본다)
def paired(a, b):
    diff = b - a
    t, pval = stats.ttest_rel(b, a)
    return dict(mean_diff=round(float(diff.mean()), 4), t=round(float(t), 2), p=float(pval),
                n_positive=int((diff > 0).sum()), n_negative=int((diff < 0).sum()), n_zero=int((diff == 0).sum()))

R["paired_task_test"] = {
    "G3000_vs_H50": paired(P[:, 0], P[:, 3]),
    "G300_vs_H50": paired(P[:, 0], P[:, 2]),
    "G100_vs_H50": paired(P[:, 0], P[:, 1]),
    "G3000_vs_G300": paired(P[:, 2], P[:, 3]),
    "G300_vs_G100": paired(P[:, 1], P[:, 2]),
}

# 규모 추세: 과제당 생성 수 N(100, 300, 3000)의 로그10 대 평균 성공률 기울기 (pp/십진 단위), 과제 부트스트랩 구간
logN = np.log10([100, 300, 3000])
mean_gen = means[1:]
slope = np.polyfit(logN, mean_gen, 1)[0]
boots = []
for _ in range(10000):
    idx = rng.integers(0, 24, 24)
    Pb = P[idx][:, 1:]
    mb = Pb.mean(axis=0)
    boots.append(np.polyfit(logN, mb, 1)[0])
boots = np.array(boots)
R["scaling_slope_pp_per_decade"] = dict(
    slope=round(float(slope) * 100, 2),
    task_bootstrap_95=[round(float(np.percentile(boots, 2.5)) * 100, 2), round(float(np.percentile(boots, 97.5)) * 100, 2)],
    share_of_bootstrap_slopes_positive=round(float((boots > 0).mean()), 4),
)
# 단조성: 과제 중 G100 < G300 < G3000 순서인 과제 수
mono = int(((P[:, 1] <= P[:, 2]) & (P[:, 2] <= P[:, 3])).sum())
R["tasks_monotone_G100_le_G300_le_G3000"] = mono

# 기술 묶음별 평균 (논문 그림 7의 묶음과 같은 방식이지만 묶음 정의는 우리가 한 것)
R["group_means"] = {}
for g in ["pnp", "door", "drawer", "appliance", "coffee"]:
    idx = [i for i, x in enumerate(groups) if x == g]
    R["group_means"][g] = dict(n=len(idx), **{s: round(float(P[idx, j].mean()), 4) for j, s in enumerate(SETTINGS)})

# ---------------------------------------------------------------- 2. 본문 수치와 표의 일치 (본문 V-A, V-C, I)
R["text_checks"] = {
    "human_1250_is_50x25": 50 * 25 == 1250,
    "gen3000_72000_is_3000x24": 3000 * 24 == 72000,
    "gen300_7200_is_300x24": 300 * 24 == 7200,
    "gen100_2400_is_100x24": 100 * 24 == 2400,
    "gen_72000_plus_28000_is_100k": 72000 + 28000 == 100000,
    "scenes_120_is_10_floorplans_x_12_styles": 10 * 12 == 120,
    "objects_1592_plus_objaverse_917_is_2509": 2509 - 1592 == 917,
    "atomic_tasks_24_plus_navigate_is_25": len(TASKS) + 1 == 25,
}

# ---------------------------------------------------------------- 3. Fig. 10 실로봇 (3시드, 평균 ± 표본 SD)
# (과제, 평균, SD) : seen real, seen sim, unseen real, unseen sim
REAL = {
    "seen": {
        "real": [(12.7, 2.5), (20.0, 5.9), (8.0, 1.6)],
        "sim": [(22.0, 2.8), (29.3, 4.1), (22.0, 5.8)],
        "paper_avg": {"real": 13.6, "sim": 24.4},
    },
    "unseen": {
        "real": [(3.3, 4.7), (1.1, 1.6), (3.3, 4.7)],
        "sim": [(8.9, 7.9), (7.8, 4.2), (11.1, 11.0)],
        "paper_avg": {"real": 2.6, "sim": 9.3},
    },
}
R["real_world"] = {}
for cat, d_ in REAL.items():
    out = {}
    for arm in ["real", "sim"]:
        m = np.array([x[0] for x in d_[arm]])
        sd = np.array([x[1] for x in d_[arm]])
        avg = m.mean()
        # 세 과제 평균의 SD를 과제 독립(공분산 0) 가정으로 근사: sqrt(sum sd^2)/3, 시드 3개로 나눔
        sd_avg_approx = np.sqrt((sd**2).sum()) / 3
        out[arm] = dict(avg_from_tasks=round(float(avg), 2), paper_avg=d_["paper_avg"][arm],
                        sd_of_avg_approx=round(float(sd_avg_approx), 2))
    # Welch t (시드 3개, 평균 두 개로 비교, 과제 평균 기준)
    a, b = out["real"], out["sim"]
    se = np.sqrt(a["sd_of_avg_approx"] ** 2 / 3 + b["sd_of_avg_approx"] ** 2 / 3)
    t = (b["avg_from_tasks"] - a["avg_from_tasks"]) / se
    # Welch df
    v1, v2 = a["sd_of_avg_approx"] ** 2 / 3, b["sd_of_avg_approx"] ** 2 / 3
    df = (v1 + v2) ** 2 / (v1**2 / 2 + v2**2 / 2)
    pval = 2 * stats.t.sf(abs(t), df)
    rel = b["avg_from_tasks"] / a["avg_from_tasks"] - 1
    out["sim_minus_real_pp"] = round(b["avg_from_tasks"] - a["avg_from_tasks"], 2)
    out["relative_improvement"] = round(rel, 3)
    out["welch_t"] = round(float(t), 2)
    out["welch_df"] = round(float(df), 2)
    out["welch_p_approx"] = round(float(pval), 4)
    R["real_world"][cat] = out

# 본문 79% 검산
R["real_world_seen_relative_79pct_check"] = round(24.4 / 13.6 - 1, 3)

# ---------------------------------------------------------------- 4. 표 2 (Diffusion Policy 문장과 표 불일치 점검)
R["diffusion_claim"] = dict(
    text_bc_transformer_PickPlaceCounterToSink=0.56,
    text_diffusion_PickPlaceCounterToSink=0.12,
    fig13_H50=float(P[names.index("PickPlaceCounterToSink"), 0]),
    fig13_G3000=float(P[names.index("PickPlaceCounterToSink"), 3]),
    note="IX 본문의 56%는 Fig. 13의 어느 설정(H50 0.02, G100 0.02, G300 0.16, G3000 0.44)에도 없다. 학습 데이터 설정이 명시되지 않음",
)

OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding="utf-8")

# ---------------------------------------------------------------- 요약 출력
print("Fig 13 means:", {k: round(v, 4) for k, v in zip(SETTINGS, means)}, "paper:", PAPER_AVG)
print("total successes per setting:", R["fig13_total_successes"])
print("avg SE (binomial, eval only):", R["avg_se_binomial"])
print("G3000 - H50:", R["diff_G3000_minus_H50"])
print("paired G3000 vs H50:", R["paired_task_test"]["G3000_vs_H50"])
print("paired G100 vs H50:", R["paired_task_test"]["G100_vs_H50"])
print("paired G300 vs G100:", R["paired_task_test"]["G300_vs_G100"])
print("paired G3000 vs G300:", R["paired_task_test"]["G3000_vs_G300"])
print("scaling slope (pp/decade):", R["scaling_slope_pp_per_decade"])
print("tasks monotone:", mono, "of 24")
print("group means:", R["group_means"])
print("real world:", json.dumps(R["real_world"], indent=1))
print("wrote", OUT)

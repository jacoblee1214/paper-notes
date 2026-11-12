"""GR00T 2 백서의 표·본문 수치를 재계산하고 통계 주장을 확인한다 (CPU, 수 초).

실행: python3 analysis/paper_checks.py  →  analysis/paper_checks_results.json
입력: 백서 본문과 표에 적힌 수치만 사용한다(원자료는 공개되지 않았다).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "paper_checks_results.json"
R = {}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# 1) 표 1a·1b 평균 재계산 (각 행: (Prog, SR))
T1A = {  # YAM 장기 과제 6개, 행 순서 = Hot Wheels, Tractor, Plane, iPhone, Fixed-Pipe, Wire
    "GR00T 2": [(95.4, 85.0), (63.8, 45.0), (71.0, 70.0), (89.0, 65.0), (70.0, 70.0), (98.5, 95.0)],
    "GR00T 2 Turbo": [(68.6, 50.0), (54.5, 40.0), (63.5, 55.0), (91.0, 55.0), (90.0, 90.0), (100.0, 100.0)],
    "pi0.5": [(13.1, 0.0), (8.4, 0.0), (24.0, 10.0), (67.5, 5.0), (30.0, 30.0), (30.0, 30.0)],
    "GR00T 1.7": [(19.5, 0.0), (3.6, 0.0), (11.0, 0.0), (49.5, 15.0), (35.0, 35.0), (17.0, 0.0)],
    "FastWAM": [(0.8, 0.0), (1.9, 0.0), (10.0, 0.0), (17.0, 0.0), (45.0, 45.0), (18.5, 5.0)],
    "MolmoAct2": [(1.5, 0.0), (1.1, 0.0), (13.5, 0.0), (38.0, 0.0), (35.0, 35.0), (10.0, 10.0)],
    "FLUX 3 Action": [(0.5, 0.0), (2.4, 0.0), (10.0, 0.0), (19.5, 0.0), (0.0, 0.0), (8.0, 0.0)],
}
REPORTED_1A = {"GR00T 2": (81.3, 71.7), "GR00T 2 Turbo": (77.9, 65.0), "pi0.5": (28.8, 12.5),
               "GR00T 1.7": (22.6, 8.3), "FastWAM": (15.5, 8.3), "MolmoAct2": (16.5, 7.5),
               "FLUX 3 Action": (6.7, 0.0)}
R["table1a_avg"] = {}
for k, rows in T1A.items():
    prog = np.mean([r[0] for r in rows])
    sr = np.mean([r[1] for r in rows])
    R["table1a_avg"][k] = {"prog_calc": round(prog, 2), "sr_calc": round(sr, 2),
                           "prog_reported": REPORTED_1A[k][0], "sr_reported": REPORTED_1A[k][1]}

T1B = {  # Dexter 7개 과제, (G2 Prog,SR), (Turbo), (1.7), (pi0.5)
    "Make Shipping Label": [(78, 60), (73, 60), (41, 5), (16, 0)],
    "Flashlight Battery Replacement": [(86, 55), (49, 15), (34, 0), (41, 0)],
    "GB300 Memory": [(96, 85), (81, 50), (43, 0), (55, 0)],
    "Truck Assembly": [(98, 65), (43, 5), (0, 0), (0, 0)],
    "Crude Salt Filtering": [(98, 85), (88, 65), (17, 0), (14, 0)],
    "Vitamin C Test": [(86, 75), (67, 25), (9, 0), (14, 0)],
    "Microscope": [(64, 55), (33, 20), (10, 0), (9, 0)],
}
names = ["GR00T 2", "GR00T 2 Turbo", "GR00T 1.7", "pi0.5"]
R["table1b_avg"] = {}
for i, nm in enumerate(names):
    prog = np.mean([v[i][0] for v in T1B.values()])
    sr = np.mean([v[i][1] for v in T1B.values()])
    R["table1b_avg"][nm] = {"prog_calc": round(prog, 2), "sr_calc": round(sr, 2)}
# 트럭 과제: 원래 장면에서 GR00T 2 성공 7/20 (표는 쉬운 시작 장면 값 65%)
g2_sr = [v[0][1] for v in T1B.values()]
g2_sr_orig = [35.0 if k == "Truck Assembly" else v for k, v in zip(T1B.keys(), g2_sr)]
R["table1b_truck_sensitivity"] = {
    "sr_table": round(np.mean(g2_sr), 2),
    "sr_if_truck_original_scene_7_of_20": round(np.mean(g2_sr_orig), 2),
}

# 1a 평균 차이(Turbo - GR00T 2)의 표준오차: 과제당 n=20 가정, 과제 평균으로 근사
def mean_diff_se(a_sr, b_sr, n=20):
    a = np.array(a_sr) / 100
    b = np.array(b_sr) / 100
    var = (a * (1 - a) / n + b * (1 - b) / n).sum() / len(a) ** 2
    return math.sqrt(var) * 100


yam_g2 = [r[1] for r in T1A["GR00T 2"]]
yam_t = [r[1] for r in T1A["GR00T 2 Turbo"]]
R["yam_turbo_minus_g2"] = {
    "diff_pts": round(np.mean(yam_t) - np.mean(yam_g2), 2),
    "se_pts_n20_per_task": round(mean_diff_se(yam_t, yam_g2), 2),
}
dex_g2 = [v[0][1] for v in T1B.values()]
dex_t = [v[1][1] for v in T1B.values()]
R["dex_turbo_minus_g2"] = {
    "diff_pts": round(np.mean(dex_t) - np.mean(dex_g2), 2),
    "se_pts_n20_per_task_approx": round(mean_diff_se(dex_t, dex_g2), 2),
}
# 단일 과제 SR 95% Wilson (n=20) 예시
R["wilson_n20"] = {f"{k}/20": [round(100 * x, 1) for x in wilson(k, 20)] for k in (17, 10, 14)}

# 2) 표 3: Dexter 제로샷, 71 과제 × 3 시도 = 213 (정수 확인)
n3 = 213
for nm, sr in [("GR00T 2", 49.8), ("GR00T 2 Turbo", 36.6), ("GR00T 1.7", 18.3), ("pi0.5", 26.3)]:
    k = round(sr / 100 * n3)
    R.setdefault("table3_counts", {})[nm] = {"k_nearest": k, "sr_from_k": round(100 * k / n3, 2),
                                            "wilson95": [round(100 * x, 1) for x in wilson(k, n3)]}


def two_prop(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    z = (p1 - p2) / math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


k_g2, k_17 = 106, 39
z, p = two_prop(k_g2, n3, k_17, n3)
R["table3_g2_vs_gr001_7"] = {"z": round(z, 2), "p_two_sided_unpaired": float(f"{p:.2e}"),
                             "note": "작업이 같은 과제에 묶여 있어 쌍 검정이 맞지만 쌍 데이터가 없어 비쌍 근사"}

# 3) 본문의 Fisher 정확 검정: 19/30 대 12/30 (H.3)
tab = [[19, 11], [12, 18]]
fe = stats.fisher_exact(tab, alternative="two-sided")
R["fisher_19_30_vs_12_30"] = {"p_two_sided": round(float(fe[1]), 3), "paper_says": 0.12}

# 4) 표본 크기 (두 비율, alpha=0.05 양측, power=0.8)
def n_per_arm(p1, p2, za=1.959964, zb=0.841621):
    pb = (p1 + p2) / 2
    num = za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return num ** 2 / (p1 - p2) ** 2


R["sample_size"] = {
    "95_vs_99_per_arm": round(n_per_arm(0.95, 0.99), 1),
    "99_vs_999_per_arm": round(n_per_arm(0.99, 0.999), 1),
    "paper_says": ["about 285", "more than 1,000"],
}

# 5) Elo 차이 → 기대 승률. Elo 식의 승률은 동점을 뺀 승부에 대한 값이다.
def elo_win(d):
    return 1 / (1 + 10 ** (-d / 400))


R["elo_expected_win_decisive"] = {f"{d}": round(100 * elo_win(d), 1) for d in (236, 95, 109, 141, 12)}
R["elo_ties"] = {
    "yam_v1_matches": 5732, "yam_v1_tie_share_reported": 0.30,
    "yam_v2_matches": 6035, "yam_v2_tie_share_reported": 0.35,
    "yam_v1_ties_calc": round(5732 * 0.30), "yam_v2_ties_calc": round(6035 * 0.35),
}
# GR00T 2 대 pi0.5 (+236), 동점 35% 가정: 전체 매치 중 A가 이기는 비율과 B가 이기는 비율
t = 0.35
q = elo_win(236)
R["g2_vs_pi05_share_of_all_matches"] = {
    "A_strict_win_pct": round(100 * (1 - t) * q, 1),
    "B_strict_win_pct": round(100 * (1 - t) * (1 - q), 1),
    "tie_pct": round(100 * t, 1),
    "note": "80%는 동점을 뺀 승부의 기대치다. 동점 35%를 포함하면 A의 엄격한 승리는 약 52%다",
}

# 6) 표 12 집계 일관성
R["table12_consistency"] = {
    "yam_campaigns_v1_plus_v2_minus_overlap": 78 + 122 - 5,  # 본문 195
    "yam_rollouts_v1_plus_v2": 6373 + 9671,  # 본문 16,044
    "yam_trials_v1_plus_v2": 2167 + 3403,  # 본문 5,570
    "yam_matches_v1_plus_v2": 5732 + 6035,  # 본문 11,767
    "yam_checkpoints_v1_plus_v2": 45 + 26,  # 본문 71
    "reported": {"campaigns": 195, "rollouts": 16044, "trials": 5570, "matches": 11767, "checkpoints": 71},
}

# 7) 표 11 학습 혼합
ep = [610.7, 130.7, 202.6, 80.9, 57.7, 76.0]
hr = [15146, 3591, 2890, 1256, 275, 350]
sh = [51.2, 10.4, 20.6, 14.0, 1.3, 2.5]
R["table11"] = {
    "episodes_k_sum": round(sum(ep), 1), "reported_episodes_M": 1.16,
    "hours_sum": sum(hr), "reported_hours": 23500,
    "share_sum": round(sum(sh), 1),
    "hour_share_pct": [round(100 * h / sum(hr), 1) for h in hr],
}

# 8) 전체 시간 수와 프레임 수 (30 Hz 가정)
hours_total = 402730 + 10184 + 23500
steps_30hz = hours_total * 3600 * 30
R["data_hours"] = {
    "human_plus_umi_plus_teleop_hours": hours_total,
    "frames_at_30hz_billion": round(steps_30hz / 1e9, 2),
    "reported_time_steps_billion": 46.2,
    "ratio_reported_over_calc": round(46.2e9 / steps_30hz, 3),
    "note": "백서 4.7.2의 46.2십억 시간 단계는 필터 전 기록 기준이라 30 Hz 계산과 2% 차이 정도로 맞는다",
    "human_hours_per_episode_min": round(402730 * 60 / 13.75e6, 2),
    "hf_open_subset_hours_per_episode_min": round(100000 * 60 / 3405573, 2),
}

# 9) 학습 처리량에서 본 step 시간 (가정: 데이터 대기 시간이 거의 0이면 소비율 = 전달율)
gpus, per_dev = 1024, 16
global_batch = gpus * per_dev
delivered = 12000.0
step_s = global_batch / delivered
R["step_time_envelope"] = {
    "global_batch_windows": global_batch,
    "delivered_windows_per_s": delivered,
    "implied_step_time_s": round(step_s, 3),
    "assumption": "per-device batch 16이 Tab. 35 측정과 같다고 가정, 노출 대기 0.049%",
}

# 10) 계산량 envelope: 2B 파라미터, 학습 FLOP ≈ 6 N T (토큰 수 T는 백서에 없음)
N = 2e9
peak = 2.5e15  # GB300 dense BF16
for tokens_per_window in (1000, 2000, 4000):
    flops_step = 6 * N * global_batch * tokens_per_window
    per_gpu = flops_step / gpus
    mfu = per_gpu / step_s / peak
    R.setdefault("compute_envelope", []).append({
        "tokens_per_window": tokens_per_window,
        "flop_per_step_e17": round(flops_step / 1e17, 2),
        "implied_mfu_at_step_time": round(mfu, 3),
    })
R["compute_envelope"].append({"note": "백서는 토큰 수와 MFU를 보고하지 않는다. 위 값은 가정별 범위일 뿐이며 측정값이 아니다. 동결된 VAE·VLM의 연산은 빠져 있다."})

# 11) 지연 시간 배율 (Tab. 27)
rows27 = [("Dexter GR00T 2 RTX5090", 449.20, 152.78, 2.94), ("Dexter GR00T 2 GB300", 219.02, 48.47, 4.52),
          ("G1 Turbo RTX5090", 224.73, 38.40, 5.85), ("G1 Turbo Thor", 720.41, 123.97, 5.81),
          ("YAM GR00T 2 RTX5090", 550.96, 179.79, 3.06), ("YAM GR00T 2 GB300", 386.06, 54.00, 7.15)]
R["table27_speedup"] = [{"row": r[0], "calc": round(r[1] / r[2], 2), "reported": r[3]} for r in rows27]

# 12) 시뮬레이션-실물 상관 (n=14 체크포인트, 본문 r = 0.93)
r_obs, n_c = 0.93, 14
zc = np.arctanh(r_obs)
se = 1 / math.sqrt(n_c - 3)
R["sim_real_pearson_ci"] = {"r": r_obs, "n": n_c,
                            "ci95": [round(float(np.tanh(zc - 1.96 * se)), 3), round(float(np.tanh(zc + 1.96 * se)), 3)]}

# 13) 인간 데이터 스케일링 (Fig. 10): 실물 3 에피소드 × 72과제 = 216 per arm
for lo, hi, nm in [(0.18, 0.41, "real robot 0% vs 100%")]:
    n_arm = 216
    k1, k2 = round(lo * n_arm), round(hi * n_arm)
    z, p = two_prop(k1, n_arm, k2, n_arm)
    R["human_scaling_naive"] = {"label": nm, "k_0pct": k1, "k_100pct": k2,
                                "z_unpaired_no_task_cluster": round(z, 2), "p": float(f"{p:.1e}"),
                                "note": "72과제 군집을 무시한 naive 검정. 시드 반복이 없다"}

# 14) RoboCasa365 평균 재계산 (표 30): 단순 평균과 보고값이 다르다 → 과제 가중치 필요
rc = {"GR00T 2": (87.4, 73.6, 38.2, 67.2), "pi0.5": (39.6, 7.1, 1.2, 16.9), "GR00T 1.7": (41.7, 17.7, 4.4, 22.1)}
R["robocasa_simple_mean_check"] = {k: {"simple_mean": round(np.mean(v[:3]), 2), "reported": v[3]} for k, v in rc.items()}

# 15) RoboDojo 평균은 다섯 차원의 단순 평균인지 (표 19, SFT 행)
dj = [40.0, 40.5, 53.8, 36.0, 5.8]
R["robodojo_mean_check_sft"] = {"simple_mean_5dims": round(np.mean(dj), 2), "reported": 35.2}

# 16) 외부 Elo 기준 (표 13) 신뢰구간이 0을 포함하는지
R["elo_ci_contains_zero"] = {"g2_vs_turbo_early": [-20, 48], "g2_vs_turbo_star": [-58, 28]}

OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False))
print(json.dumps(R, indent=1, ensure_ascii=False)[:6000])

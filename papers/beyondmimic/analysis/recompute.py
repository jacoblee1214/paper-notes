"""BeyondMimic (arXiv:2508.08241 v4) 보고 수치 재계산. CPU 전용, 수 초 안에 끝남.

실행: python3 analysis/recompute.py  -> analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
out = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cohen_h(p1, p2):
    return abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))


# 1. 사용자 연구 (본문 Results, Fig. 4E). N=77, 쌍 20개 (걷기 10, 달리기 10)
N_PART = 77
PAIRS = {"overall": 20, "walking": 10, "running": 10}
reported = {  # 보고된 선호율(%) 과 p, h
    "overall": {"ours": 70.8, "h": 0.859},
    "walking": {"ours": 57.0, "h": 0.281},
    "running": {"ours": 84.7, "h": 1.532},
}
user = {}
for gait, npair in PAIRS.items():
    n = N_PART * npair
    pct = reported[gait]["ours"]
    # 보고 백분율을 정수 표로 되돌리기: n에서 가장 가까운 정수 k
    k = min(range(n + 1), key=lambda j: abs(100 * j / n - pct))
    k_round_ok = round(100 * k / n, 1) == pct
    res = binomtest(k, n, 0.5, alternative="two-sided")
    lo, hi = wilson(k, n)
    user[gait] = {
        "n_judgments": n,
        "ours_count": k,
        "ours_pct_recomputed": round(100 * k / n, 2),
        "reported_pct_matches": k_round_ok,
        "wilson95": [round(100 * lo, 1), round(100 * hi, 1)],
        "binom_p_two_sided": res.pvalue,
        "cohen_h_vs_50": round(cohen_h(k / n, 0.5), 3),
        "cohen_h_ours_vs_unitree": round(cohen_h(k / n, 1 - k / n), 3),
        "cohen_h_reported": reported[gait]["h"],
    }
# 전체 판단 수 1540 = 77 x 20. 걷기 + 달리기 합
k_all = user["walking"]["ours_count"] + user["running"]["ours_count"]
n_all = user["walking"]["n_judgments"] + user["running"]["n_judgments"]
lo, hi = wilson(k_all, n_all)
user["overall"] = {
    "n_judgments": n_all,
    "ours_count": k_all,
    "ours_pct_recomputed": round(100 * k_all / n_all, 2),
    "reported_pct_matches": round(100 * k_all / n_all, 1) == 70.8,
    "wilson95": [round(100 * lo, 1), round(100 * hi, 1)],
    "binom_p_two_sided": binomtest(k_all, n_all, 0.5).pvalue,
    "cohen_h_vs_50": round(cohen_h(k_all / n_all, 0.5), 3),
    "cohen_h_ours_vs_unitree": round(cohen_h(k_all / n_all, 1 - k_all / n_all), 3),
    "cohen_h_reported": 0.859,
}
# 민감도: 참가자 단위 군집을 무시하지 않으려면 설계효과(DEFF) 필요. 보고된 자료에는 없음.
# 참가자 수 77명을 유효 독립 단위로 본 최악의 경우 검정 (판단 1개 = 참가자 1명, 쌍 수준 평균)
# 각 쌍의 ours 비율을 알 수 없으므로 하한만 계산: 참가자당 쌍 20개 평균 비율이 0.5를 넘는지 t검정 근사 불가.
out["user_study"] = user

# 2. 보행 속도 추적 오차 12.14 % (걷기), 13.65 % (달리기) — 시뮬, 시도 수·분산 미보고 -> 재계산 불가 표기
out["velocity_error_pct"] = {"walking": 12.14, "running": 13.65, "note": "n·분산 미보고, 재계산 불가"}

# 3. 잠재 확산 절제: 시뮬 성공률 5% (기준) vs 95% (제안), 시도 수 미보고
#    시도 수를 n이라 가정했을 때 Wilson 95% 구간 예시 (n=20 가정은 보고되지 않음; 참고용)
ablation = {}
for n_try in (20, 50, 100):
    ablation[f"n={n_try}"] = {
        "baseline_5pct_k=round(0.05n)": {"k": round(0.05 * n_try), "wilson95": [round(100 * x, 1) for x in wilson(round(0.05 * n_try), n_try)]},
        "proposed_95pct_k=round(0.95n)": {"k": round(0.95 * n_try), "wilson95": [round(100 * x, 1) for x in wilson(round(0.95 * n_try), n_try)]},
    }
out["latent_ablation_if_n"] = ablation

# 4. 보행 -> 공중 회전 동역학 수치 (Results): 평균 7.01 rad/s vs 인간 7.75 rad/s (ref 50)
out["pelvic_angvel"] = {"robot_mean": 7.01, "robot_peak": 20, "human_mean_ref50": 7.75,
                        "ratio_mean": round(7.01 / 7.75, 3), "note": "측정 조건·정의가 인간 값과 같은지 보고되지 않음"}

# 5. 표 S3: 구동기 반사 관성 J_arm = Jr (g1 g2)^2 + J1 g2^2 + J2
t_s3 = [
    ("5020-16", 3.56, 4.5, 1.390e-05, 1.700e-06, 1.690e-05, 3.610e-03),
    ("7520-14.3", 4.5, 3.18, 4.890e-05, 9.800e-06, 5.330e-05, 1.018e-02),
    ("7520-22.5", 4.5, 5.0, 4.890e-05, 1.090e-05, 7.380e-05, 2.510e-02),
    ("4010-25", 5.0, 5.0, 6.800e-06, 0.0, 0.0, 4.250e-03),
]
s3 = []
for name, g1, g2, jr, j1, j2, total_rep in t_s3:
    calc = jr * (g1 * g2) ** 2 + j1 * g2 ** 2 + j2
    s3.append({"actuator": name, "total_recomputed": calc, "total_reported": total_rep,
               "rel_diff_pct": round(100 * (calc - total_rep) / total_rep, 2)})
out["tableS3"] = s3

# 6. 표 S4 armature % = I_arm / (I_sub + I_arm) 검산
t_s4 = [
    ("left shoulder pitch", 1.748e-01, 3.610e-03, 2.0),
    ("left shoulder yaw", 2.742e-02, 3.610e-03, 11.6),
    ("left wrist roll", 3.664e-04, 3.610e-03, 90.8),
    ("left wrist pitch", 4.804e-03, 4.250e-03, 46.9),
    ("left hip pitch", 8.543e-01, 1.018e-02, 1.2),
    ("left knee", 1.115e-01, 2.510e-02, 18.4),
    ("left ankle pitch", 2.777e-03, 7.219e-03, 72.2),
    ("left ankle roll", 3.871e-04, 7.219e-03, 94.9),
]
s4 = []
for name, isub, iarm, pct_rep in t_s4:
    pct = 100 * iarm / (isub + iarm)
    s4.append({"joint": name, "armature_pct_recomputed": round(pct, 1), "armature_pct_reported": pct_rep,
               "match_0.2pct": abs(pct - pct_rep) <= 0.2})
out["tableS4"] = s4

# 7. 예측 지평: 0.64 s = H / f (표 S7: H = 16, 정책 25 Hz)
out["horizon_s"] = {"H": 16, "f_Hz": 25, "H_over_f": round(16 / 25, 3), "reported_s": 0.64}

# 8. 표 S8 집계: 행 수와 구간 수 (실로봇 열이 '-'가 아닌 항목)
out["tableS8_note"] = "수작업 집계: 실로봇 행 21개(짧은 시퀀스 7 + LAFAN1 14), 구간 기준 약 32개; 본문 '30 clips'와 정의가 불명확"

(HERE / "recompute_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: user[k] for k in user}, ensure_ascii=False, indent=1))
print("S3", [(r["actuator"], r["rel_diff_pct"]) for r in s3])
print("S4", [(r["joint"], r["armature_pct_recomputed"], r["armature_pct_reported"], r["match_0.2pct"]) for r in s4])
print("horizon", out["horizon_s"])

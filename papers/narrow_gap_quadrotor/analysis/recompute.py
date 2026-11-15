"""2610.37557 (Narrow-gap quadrotor, CoRL 2026 poster) 논문 수치 재계산과 일관성 검사 (scipy, numpy).

입력 1: 논문 본문·표에서 옮긴 숫자 (각 상수 옆에 출처 절·표 번호).
입력 2: 공식 저장소 정적 읽기 클론의 공개 결과 파일 repo/runs/*/paper_data/*.csv (있을 때만).
        evaluate_all.out 의 최종 평가 비율(1000 롤아웃)도 같은 방식으로 읽는다.
출력: analysis/recompute_results.json   (실행: python3 analysis/recompute.py)

검사 항목
  1. 시행 수와 구간: 1000 롤아웃 성공률의 Wilson 95% 구간, 실물 10회 Wilson 구간과 12회 민감도
  2. 독립 표본 검정: Fisher 정확 검정 (쌍체 정보가 없으므로 McNemar 불가)
  3. Table 4 (K 변화): 차이 0.7 pt의 검정력
  4. Table 3 산술: 전이·업데이트·시간 비율, 에폭당 전이 수, 업데이트 정의
  5. Table 3 절단(censoring) 민감도: PPO 시드 2 (30k 에폭 절단)를 하한으로 넣었을 때 비율
  6. 공개 학습 로그: 수렴 에폭(Table 1 정의) 재계산, 벽시계 시간 정의(exec 대 전체) 대조
  7. 기하: 틈 4.0 대 2.5 cm, 드론 반지름 대비 여유
  8. 부록 A.3: 분산 비율 (27.93 / 2.38)^2
"""
import csv
import glob
import json
import math
import os
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "repo" / "runs"
OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def cp(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def pct(x):
    return round(100 * x, 2)


# ---- 1. 최종 시행 (Table 1, Table 2, evaluate_all.out) ----------------------
# evaluate_all.out 의 [SUCCESS RATE] 와 "Gap crossing rate", 1000 롤아웃, seed 0
final_sim = {
    "state_static": (1000, 1000),   # 100.00%
    "vision_static": (989, 1000),   # 98.90%
    "state_moving": (1000, 1000),   # 100.00%
    "vision_moving": (993, 1000),   # 99.30%  (논문 Table 1, 초록 99.3%)
}
res["final_sim_trials"] = {}
for k, (s, n) in final_sim.items():
    w = wilson(s, n)
    res["final_sim_trials"][k] = {
        "success": s, "n": n, "rate_pct": pct(s / n),
        "wilson95_pct": [pct(w[0]), pct(w[1])],
    }
# 하이-피델리티 시뮬레이션 (Table 2): 81.0%, 100 시행 (가정: n=100, k=81)
hifi_k, hifi_n = 81, 100
res["hifi_sim"] = {"success": hifi_k, "n": hifi_n, "wilson95_pct": [pct(wilson(hifi_k, hifi_n)[0]), pct(wilson(hifi_k, hifi_n)[1])]}

# 실물 (Table 2, 본문 §4 Transfer): 기록 12회, 로그만 있던 2회 제외, 채점 10회 중 8회 성공
real_k, real_n = 8, 10
w = wilson(real_k, real_n)
res["real_world"] = {
    "scored": {"success": real_k, "n": real_n, "rate_pct": pct(real_k / real_n),
               "wilson95_pct": [pct(w[0]), pct(w[1])], "cp95_pct": [pct(cp(real_k, real_n)[0]), pct(cp(real_k, real_n)[1])]},
    # 민감도: 제외한 2회가 모두 성공이었거나 모두 실패였다면
    "if_both_excluded_success": {"success": 10, "n": 12, "rate_pct": pct(10 / 12), "wilson95_pct": [pct(wilson(10, 12)[0]), pct(wilson(10, 12)[1])]},
    "if_both_excluded_fail": {"success": 8, "n": 12, "rate_pct": pct(8 / 12), "wilson95_pct": [pct(wilson(8, 12)[0]), pct(wilson(8, 12)[1])]},
}
# 실물 8/10 과 시뮬 993/1000 의 Fisher 정확 검정, 실물 12회 가정 두 경우 포함
res["fisher"] = {}
def fisher(name, a, b):
    odds, p = stats.fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])
    res["fisher"][name] = {"p_two_sided": float(p), "a": list(a), "b": list(b)}

fisher("sim_vision_moving_vs_hifi", (993, 1000), (81, 100))
fisher("sim_vision_moving_vs_real_scored", (993, 1000), (8, 10))
fisher("sim_vision_moving_vs_real_if_10of12", (993, 1000), (10, 12))
fisher("sim_vision_moving_vs_real_if_8of12", (993, 1000), (8, 12))
fisher("hifi_vs_real_scored", (81, 100), (8, 10))
fisher("state_static_vs_vision_static", (1000, 1000), (989, 1000))
fisher("state_moving_trained_vs_state_static_transfer", (1000, 1000), (315, 1000))
fisher("vision_moving_trained_vs_vision_static_transfer", (993, 1000), (66, 1000))

# 정적 전이 실험 (본문 §4 Static-to-Moving Transfer, 1000 롤아웃): 31.5%, 6.6%
res["transfer_static_to_moving"] = {
    "state_static_ckpt": {"success": 315, "n": 1000, "wilson95_pct": [pct(wilson(315, 1000)[0]), pct(wilson(315, 1000)[1])]},
    "vision_static_ckpt": {"success": 66, "n": 1000, "wilson95_pct": [pct(wilson(66, 1000)[0]), pct(wilson(66, 1000)[1])]},
}

# ---- 2. Table 4: K 변화 (가정: 각 1000 롤아웃, 표준 Table 1 프로토콜) ---------
K_rows = {1: 995, 2: 988, 4: 993, 8: 989}   # 99.5, 98.8, 99.3, 98.9 (n=1000 가정)
res["table4_K"] = {}
for K, s in K_rows.items():
    res["table4_K"][str(K)] = {"success_pct": s / 10, "wilson95_pct": [pct(wilson(s, 1000)[0]), pct(wilson(s, 1000)[1])]}
fisher("K1_vs_K8", (995, 1000), (989, 1000))
fisher("K4_vs_K8", (993, 1000), (989, 1000))
res["table4_K"]["max_minus_min_pts"] = round(99.5 - 98.8, 2)   # 본문 "0.7 points" 와 일치
res["table4_K"]["note"] = "n=1000 가정. 논문은 Table 4의 시행 수를 명시하지 않음. 차이가 0.7pt이면 two-sample SE 약 0.45pt"

# ---- 3. Table 3 산술 (1000 환경, 상태 과제, 5 seeds) ------------------------
T_TRANS_PER_EPOCH = 1000 * 150          # 1000 env × 150 step = 150,000 (Table 3 caption, §4)
MB = 750                                # PPO 미니배치 크기 (§4, A.10)
b_static = dict(trans=75e6, upd=500, time=38.3)
p_static = dict(trans=210e6, upd=1.12e6, time=246.0)
b_mov = dict(trans=135e6, upd=900, time=67.5)
p_mov = dict(trans=2.57e9, upd=13.7e6, time=3073.0)
t3 = {}
for name, b, p in [("static", b_static, p_static), ("moving", b_mov, p_mov)]:
    t3[name] = {
        "trans_ratio": round(p["trans"] / b["trans"], 2),
        "updates_ratio": round(p["upd"] / b["upd"], 1),
        "time_ratio": round(p["time"] / b["time"], 2),
        "bptt_epochs_from_trans": round(b["trans"] / T_TRANS_PER_EPOCH, 1),
        "ppo_epochs_from_trans": round(p["trans"] / T_TRANS_PER_EPOCH, 1),
        "ppo_updates_per_epoch": round(p["upd"] / (p["trans"] / T_TRANS_PER_EPOCH), 1),
    }
res["table3_arithmetic"] = t3
res["table3_arithmetic"]["minibatches_per_epoch_check"] = {
    "expected_from_750": T_TRANS_PER_EPOCH / MB,           # 200 → 200 × 4 update epochs = 800 (§4)
    "ppo_updates_per_epoch_reported": 800,
}
res["table3_arithmetic"]["note"] = "updates 비율은 BPTT 1 업데이트/에폭, PPO 800 미니배치/에폭의 정의 차이이므로 속도 비율이 아님"

# ---- 4. Table 3 절단 민감도: PPO 시드 2 (30k 에폭 절단, 4.5 B 전이, 24 M 업데이트, 5,377 s) -----
# 4개 수렴 시드 평균을 합계로 되돌린 뒤, 절단된 시드 값을 하한으로 넣어 5 시드 평균을 다시 계산
sum4_trans = 4 * 2.57e9
sum4_upd = 4 * 13.7e6
sum4_time = 4 * 3073.0
cens = dict(trans=4.5e9, upd=24e6, time=5377.0)
mean5 = dict(trans=(sum4_trans + cens["trans"]) / 5, upd=(sum4_upd + cens["upd"]) / 5, time=(sum4_time + cens["time"]) / 5)
res["table3_censoring_bound"] = {
    "ppo_5seed_mean_lower_bound": {k: v for k, v in mean5.items()},
    "trans_ratio_lower_bound": round(mean5["trans"] / b_mov["trans"], 1),
    "time_ratio_lower_bound": round(mean5["time"] / b_mov["time"], 1),
    "updates_ratio_lower_bound_k": round(mean5["upd"] / b_mov["upd"] / 1e3, 2),
    "note": "논문 19×, 45.5×는 수렴한 4개 시드 평균 기준. 절단 시드를 넣으면 하한이 21.9×, 52× 이상이 됨. 절단 시드의 실제 값은 이보다 클 수 있음",
}

# ---- 5. 공개 학습 로그 (repo/runs/*/paper_data) -----------------------------
def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_evals(name):
    p = RUNS / name / "paper_data" / f"{name}_eval_metrics.csv"
    if not p.exists():
        return None
    rows = [r for r in csv.DictReader(open(p, encoding="utf-8")) if r.get("checkpoint_epoch")]
    return [(int(float(r["checkpoint_epoch"])), f(r["success_rate_no_collision_pct"]), f(r["global_elapsed_seconds"]),
             f(r["avg_crossing_dist_from_gap_center_m"])) for r in rows]


def load_timing(name):
    p = RUNS / name / "paper_data" / f"{name}_timing_blocks.csv"
    if not p.exists():
        return None
    return list(csv.DictReader(open(p, encoding="utf-8")))


def first_streak(evals, thr=95.0, n=3):
    """Table 1 정의: 3회 연속 평가가 95% 이상인 첫 체크포인트(에폭)."""
    for i in range(len(evals) - n + 1):
        if all(evals[j][1] is not None and evals[j][1] >= thr for j in range(i, i + n)):
            return evals[i]
    return None


runs_all = sorted(os.path.basename(d) for d in glob.glob(str(RUNS / "*")) if os.path.isdir(d))
res["released_logs"] = {"present": RUNS.exists(), "n_runs": len(runs_all), "runs": {}}
if RUNS.exists():
    for name in runs_all:
        ev = load_evals(name)
        if ev is None:
            continue
        conv = first_streak(ev)
        tb = load_timing(name)
        exec_cum = co_cum = train_cum = None
        if tb:
            conv_ep = conv[0] if conv else None
            if conv_ep is not None:
                upto = [r for r in tb if int(float(r["end_epoch"])) <= conv_ep]
                exec_cum = sum(f(r.get("exec_seconds")) or 0.0 for r in upto)
                co_cum = sum(f(r.get("compile_seconds")) or 0.0 for r in upto)
                train_cum = sum(f(r.get("train_seconds")) or 0.0 for r in upto)
        res["released_logs"]["runs"][name] = {
            "convergence_epoch_table1_rule": conv[0] if conv else None,
            "global_wallclock_s_at_conv": round(conv[2], 1) if conv else None,
            "exec_cum_s_at_conv": round(exec_cum, 1) if exec_cum else exec_cum,
            "compile_cum_s_at_conv": round(co_cum, 1) if co_cum else co_cum,
            "train_cum_s_at_conv": round(train_cum, 1) if train_cum else train_cum,
            "max_success_pct_100rollouts": max(x[1] for x in ev if x[1] is not None),
            "final_success_pct_100rollouts": ev[-1][1],
        }
    # 평가 로그에 100 롤아웃 체크포인트만 있음: 논문 Table 1 프로토콜(1000 롤아웃)과 다름

    # 시간 정의 대조: 1000 환경 상태 과제, Table 3의 BPTT 38.3 s, 67.5 s / PPO 246 s, 3073 s (5시드 평균)
    def ratio_pair(bp, pp):
        b, p = res["released_logs"]["runs"].get(bp), res["released_logs"]["runs"].get(pp)
        if not b or not p:
            return None
        return {
            "bptt": bp, "ppo": pp,
            "bptt_exec_s": b["exec_cum_s_at_conv"], "ppo_exec_s": p["exec_cum_s_at_conv"],
            "bptt_global_s": b["global_wallclock_s_at_conv"], "ppo_global_s": p["global_wallclock_s_at_conv"],
            "ratio_exec_basis": round(p["exec_cum_s_at_conv"] / b["exec_cum_s_at_conv"], 2) if b["exec_cum_s_at_conv"] and p["exec_cum_s_at_conv"] else None,
            "ratio_global_basis": round(p["global_wallclock_s_at_conv"] / b["global_wallclock_s_at_conv"], 2) if b["global_wallclock_s_at_conv"] and p["global_wallclock_s_at_conv"] else None,
        }
    res["timing_basis_check_1000env_single_seed"] = {
        "static": ratio_pair("bptt_static_state_envs1000", "ppo_static_state_envs1000"),
        "moving": ratio_pair("bptt_moving_state_envs1000", "ppo_moving_state_envs1000"),
        "note": "단일 시드 로그. 논문 Table 3 값(38.3 s, 246 s 등)은 5시드 평균이라 직접 비교는 근사. 방법별 시간 정의가 다른지 확인하는 용도",
    }
    # 논문 "0.4–5.6 minutes" 주장의 대조: 수렴한 실행들의 벽시계와 exec 시간
    vis = [k for k in res["released_logs"]["runs"] if "vision" in k and res["released_logs"]["runs"][k]["convergence_epoch_table1_rule"]]
    res["minutes_claim_check"] = {
        "claim_text": "정책은 단일 GPU에서 0.4–5.6분에 학습된다 (§1, 초록은 '수 분')",
        "converged_runs_global_min": {k: round(res["released_logs"]["runs"][k]["global_wallclock_s_at_conv"] / 60, 2)
                                      for k in res["released_logs"]["runs"] if res["released_logs"]["runs"][k]["convergence_epoch_table1_rule"] and res["released_logs"]["runs"][k]["global_wallclock_s_at_conv"]},
        "vision_runs_global_min_range": [round(min(res["released_logs"]["runs"][k]["global_wallclock_s_at_conv"] for k in vis) / 60, 2),
                                         round(max(res["released_logs"]["runs"][k]["global_wallclock_s_at_conv"] for k in vis) / 60, 2)] if vis else None,
        "note": "전체 벽시계는 JIT 컴파일과 평가 시간을 포함. exec 시간만 쓰면 더 짧음. 어느 시간인지 논문에 정의 없음",
    }

# ---- 6. Table 1 과 Table 3 의 환경 수 (100 대 1000) 확인 --------------------
res["table1_epochs_check"] = {
    "paper": {"state_static": 500, "vision_static": 3500, "state_moving": 1500, "vision_moving": 4000},
    "released_100env_run": {
        "state_static": res.get("released_logs", {}).get("runs", {}).get("bptt_static_state_envs100", {}).get("convergence_epoch_table1_rule"),
        "vision_static": res.get("released_logs", {}).get("runs", {}).get("bptt_static_vision_envs100", {}).get("convergence_epoch_table1_rule"),
        "state_moving": res.get("released_logs", {}).get("runs", {}).get("bptt_moving_state_envs100", {}).get("convergence_epoch_table1_rule"),
        "vision_moving": res.get("released_logs", {}).get("runs", {}).get("bptt_moving_vision_envs100", {}).get("convergence_epoch_table1_rule"),
    },
}

# ---- 7. 기하: 드론 반축 (a = 0.19 m, c = 0.05 m), 틈 80 × 25 cm -------------------
W, H = 0.80, 0.25
a_drone, c_drone = 0.19, 0.05
res["geometry"] = {
    "half_clearance_height_cm": round((H / 2 - c_drone) * 100, 2),   # 7.5 cm
    "half_clearance_width_cm": round((W / 2 - a_drone) * 100, 2),   # 21 cm
    "note": "드론 축 길이는 quadrotor_files/example_quad.yaml 과 static_gap_state_env.py 기본값 기준(코드 독해). 논문 본문은 10×38×38 cm (§3.1)",
    "reported_penetrations_cm": [4.6, 9.3],
    "penetration_definition_in_paper": None,
}

# ---- 8. 부록 A.3: 분산 비율 -------------------------------------------------
res["A3_variance_ratio"] = {"std_sf": 27.93, "std_pw": 2.38, "ratio_of_variances": round((27.93 / 2.38) ** 2, 1),
                            "paper_claim": "약 140×"}

# ---- 9. 실물 기록 ------------------------------------------------------------
res["real_flight_recording_check"] = {"recorded": 12, "excluded": 2, "scored": 10, "success": 8, "failures_listed": ["03", "08"]}

OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


def show():
    print("Final sim trials:")
    for k, v in res["final_sim_trials"].items():
        print(f"  {k:14s} {v['success']}/{v['n']} = {v['rate_pct']}%  Wilson {v['wilson95_pct']}")
    print("Hi-fi:", res["hifi_sim"]["success"], "/", res["hifi_sim"]["n"], res["hifi_sim"]["wilson95_pct"])
    r = res["real_world"]
    print("Real scored 8/10:", r["scored"]["rate_pct"], "Wilson", r["scored"]["wilson95_pct"], "CP", r["scored"]["cp95_pct"])
    print("Real if 10/12:", r["if_both_excluded_success"]["wilson95_pct"], " if 8/12:", r["if_both_excluded_fail"]["wilson95_pct"])
    for k, v in res["fisher"].items():
        print(f"  Fisher {k}: p={v['p_two_sided']:.3g}")
    print("Table3 arithmetic:", json.dumps(res["table3_arithmetic"], ensure_ascii=False)[:400])
    print("Censoring bound:", res["table3_censoring_bound"]["trans_ratio_lower_bound"], res["table3_censoring_bound"]["time_ratio_lower_bound"])
    if res["released_logs"]["present"]:
        print("Released convergence (Table 1 rule):", res["table1_epochs_check"]["released_100env_run"])
        print("Timing basis check:", json.dumps(res.get("timing_basis_check_1000env_single_seed"), ensure_ascii=False))
        print("Minutes claim:", json.dumps(res.get("minutes_claim_check"), ensure_ascii=False))
    print("Geometry:", res["geometry"]["half_clearance_height_cm"], res["geometry"]["half_clearance_width_cm"])
    print("A.3 ratio:", res["A3_variance_ratio"]["ratio_of_variances"])


if __name__ == "__main__":
    show()
    print("wrote", OUT)

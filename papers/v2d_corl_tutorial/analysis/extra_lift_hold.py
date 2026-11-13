"""lift-hold 성공 판정의 numpy 포트와 합성 높이 트레이스 검증 (numpy, CPU).

저장소의 판정 식(closed_loop/evaluation.py의 LiftHoldTracker.update, complete와 같은 순서)과 임계값
(task_profiles/tissue_box_lift_hold.json)을 읽고 손으로 옮겨 쓴 독자 구현이다. 저장소 파일은 import하거나 실행하지 않았다.

검증 구성
  1. 알려진 답 테스트: 손으로 정한 정답을 가진 시나리오(깨끗한 들기-유지-내려놓기, 조기 낙하, 유지 중간 끊김,
     끊김 뒤 긴 유지, 느린 들기, 낮은 들기, 들고 있는 채로 끝남, 가만히 있음)를 판정기에 넣는다.
  2. 프레임 속도 변화: 같은 물리 궤적을 10·20·30 Hz로 샘플링한다. 임계는 샘플 수 단위라서 판정이 바뀌는지 본다.
  3. 높이 측정 잡음 σ에 따른 거짓 채택률(FAR)과 거짓 기각률(FRR).
  4. 임계 선택의 교환: 유지 샘플 수, 안정 배치의 샘플 간 변화량, 하강 한계를 바꾸며 FAR와 FRR을 함께 본다.

잡음은 모든 샘플(시작 샘플 포함)에 독립 가우시안으로 더한다. 실제 값이 시뮬레이션 참값인지 추정값인지는
이 검증으로 알 수 없다. 그래서 σ 축은 "추정 잡음이 있다면"이라는 조건으로 읽는다.

실행: python3 -I analysis/extra_lift_hold.py
  → analysis/extra_lift_hold_results.json
"""
import json
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "analysis" / "extra_lift_hold_results.json"
SEED = 20261113
TRIALS = 200

# 저장소 설정 (task_profiles/tissue_box_lift_hold.json의 evaluator 블록을 옮김)
BASE_CFG = dict(
    final_min_lift_m=-0.01,
    hold_threshold_m=0.05,
    lift_threshold_m=0.10,
    max_descent_step_m=0.01,
    max_terminal_range_m=0.01,
    max_terminal_step_m=0.005,
    min_hold_steps=20,
    placement_threshold_m=0.03,
    placement_window_steps=20,
)
NOMINAL_FPS = 20.0


def judge(z, cfg):
    """판정 포트. 첫 샘플을 기준 높이로 두고, 나머지 샘플을 순서대로 누적한다 (evaluate_lift_hold와 같은 순서).

    반환: dict(success, lift_hold_success, max_lift_m, max_hold_steps, final_lift_m, sample_count).
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 1 or z.size == 0 or not np.isfinite(z).all():
        raise ValueError("z must be a finite non-empty 1-D trace")
    z0 = float(z[0])
    max_lift = 0.0
    max_descent = 0.0
    consec = 0
    max_hold = 0
    latest = 0.0
    lifts = []
    for v in z[1:]:
        lift = float(v) - z0
        max_lift = max(max_lift, lift)
        # 하강: 최대 들기가 유지 임계에 도달한 뒤에만 센다 (직전 샘플 대비 감소)
        if max_lift >= cfg["hold_threshold_m"]:
            max_descent = max(max_descent, latest - lift)
        lifts.append(lift)
        consec = consec + 1 if lift >= cfg["hold_threshold_m"] else 0
        max_hold = max(max_hold, consec)
        latest = lift
    n = len(lifts)
    lift_hold = max_lift >= cfg["lift_threshold_m"] and max_hold >= cfg["min_hold_steps"]
    W = cfg["placement_window_steps"]
    stable = False
    if n >= W:
        term = np.asarray(lifts[-W:])
        stable = (np.max(term) <= cfg["placement_threshold_m"]
                  and np.max(np.abs(np.diff(term))) <= cfg["max_terminal_step_m"]
                  and np.ptp(term) <= cfg["max_terminal_range_m"])
    success = bool(lift_hold and latest >= cfg["final_min_lift_m"] and stable
                   and max_descent <= cfg["max_descent_step_m"])
    return {"success": success, "lift_hold_success": bool(lift_hold), "max_lift_m": max_lift,
            "max_hold_steps": int(max_hold), "final_lift_m": latest, "sample_count": n,
            "clauses": {"lift": max_lift >= cfg["lift_threshold_m"], "hold": max_hold >= cfg["min_hold_steps"],
                        "final": latest >= cfg["final_min_lift_m"], "placement": bool(stable),
                        "descent": max_descent <= cfg["max_descent_step_m"]}}


# ---- 합성 트레이스: 시간-높이 꼭짓점을 주고 선형 보간으로 샘플링 ----
SCENARIOS = {
    # 들기 1.0 s, 2.0 s 유지, 1.0 s에 걸쳐 내려놓음, 이후 정지. 정답: 성공
    "clean_lift_hold": {"knots": [(0, 0), (0.5, 0), (1.5, 0.15), (3.5, 0.15), (4.5, 0.0), (6.0, 0.0)], "label": 1},
    # 0.5 s 유지 후 0.1 s 안에 떨어뜨림. 정답: 실패 (유지, 하강)
    "early_drop": {"knots": [(0, 0), (0.5, 0), (1.0, 0.15), (1.5, 0.15), (1.6, 0.0), (3.0, 0.0)], "label": 0},
    # 들기(0.1 s 상승) 뒤 0.1 s 유지, 0.9 s 하강 후 0.3 s 바닥, 같은 방식으로 다시 들고 0.1 s 유지 후 하강.
    # 두 연속 유지는 각각 약 0.77 s(15–16 샘플 @20 Hz)라 20 샘플 미만. 하강 속도 0.1667 m/s = 샘플당 0.0083 (20 Hz)라 하강 조건은 통과.
    # 정답: 실패 (유지 조건만)
    "dip_splits_hold": {"knots": [(0, 0), (0.5, 0), (0.6, 0.15), (0.7, 0.15), (1.6, 0.0), (1.9, 0.0), (2.0, 0.15),
                                  (2.1, 0.15), (3.0, 0.0), (5.0, 0.0), (6.0, 0.0)], "label": 0},
    # 앞의 끊김 뒤 두 번째 유지를 1.2 s로 늘림: 연속 유지 약 1.9 s. 정답: 성공
    "dip_then_long_hold": {"knots": [(0, 0), (0.5, 0), (0.6, 0.15), (0.7, 0.15), (1.6, 0.0), (1.9, 0.0), (2.0, 0.15),
                                     (3.2, 0.15), (4.1, 0.0), (6.0, 0.0)], "label": 1},
    # 4.0 s에 걸쳐 0.15 m까지 천천히 들고 1.5 s 유지, 1.0 s에 내려놓음. 정답: 성공
    "slow_lift": {"knots": [(0, 0), (0.5, 0), (4.5, 0.15), (6.0, 0.15), (7.0, 0.0), (8.5, 0.0)], "label": 1},
    # 유지 구간이 기준(20 샘플) 바로 아래: 연속 유지 약 0.95 s, 하강 0.9 s(0.0083/샘플). 정답: 실패 (유지 조건, 경계 근처)
    "near_miss_hold": {"knots": [(0, 0), (0.5, 0), (0.6, 0.15), (0.88, 0.15), (1.78, 0.0), (3.0, 0.0), (5.0, 0.0)], "label": 0},
    # 0.08 m까지만 들고 2 s 유지 후 내려놓음. 정답: 실패 (들기 조건)
    "low_lift": {"knots": [(0, 0), (0.5, 0), (1.5, 0.08), (3.5, 0.08), (4.5, 0.0), (6.0, 0.0)], "label": 0},
    # 0.15 m까지 들고 끝까지 든 채로 있음. 정답: 실패 (안정 배치 조건)
    "held_at_end": {"knots": [(0, 0), (0.5, 0), (1.5, 0.15), (5.0, 0.15)], "label": 0},
    # 가만히 있음. 정답: 실패
    "static": {"knots": [(0, 0), (6.0, 0.0)], "label": 0},
}


def trace(name, fps, rng=None, sigma=0.0):
    """시나리오를 fps로 샘플링하고 잡음을 더한다."""
    knots = np.asarray(SCENARIOS[name]["knots"], dtype=float)
    t = np.arange(0.0, knots[-1, 0] + 1e-9, 1.0 / fps)
    z = np.interp(t, knots[:, 0], knots[:, 1])
    if rng is not None and sigma > 0:
        z = z + rng.normal(0.0, sigma, size=z.shape)
    return z


def check_known_answers():
    """정답을 손으로 정한 테스트. 하나라도 어긋나면 AssertionError."""
    out = {}
    for name, spec in SCENARIOS.items():
        r = judge(trace(name, NOMINAL_FPS), BASE_CFG)
        out[name] = {"expected": bool(spec["label"]), "judge": r["success"], "clauses": r["clauses"],
                     "max_hold_steps": r["max_hold_steps"], "max_lift_m": round(r["max_lift_m"], 4),
                     "ok": r["success"] == bool(spec["label"])}
    bad = [k for k, v in out.items() if not v["ok"]]
    assert not bad, f"known-answer mismatch: {bad}"
    # 추가 검증: 전체 높이에 0.3 m 상수 오프셋을 더해도 판정이 같아야 한다 (기준 높이를 빼므로)
    z = trace("clean_lift_hold", NOMINAL_FPS)
    assert judge(z + 0.3, BASE_CFG)["success"] is True
    # 추가 검증: 1 샘플 길이 및 빈 입력 방어
    assert judge(np.array([0.0]), BASE_CFG)["success"] is False
    return out


def fps_sweep():
    res = {}
    for fps in [10.0, 20.0, 30.0]:
        res[str(fps)] = {}
        for name in ["clean_lift_hold", "early_drop", "dip_splits_hold", "dip_then_long_hold", "slow_lift"]:
            r = judge(trace(name, fps), BASE_CFG)
            res[str(fps)][name] = {"label": SCENARIOS[name]["label"], "judge": r["success"],
                                   "clauses": r["clauses"], "max_hold_steps": r["max_hold_steps"]}
    return res


def mc(name, sigma, fps, cfg, rng, trials=TRIALS):
    acc = 0
    for _ in range(trials):
        z = trace(name, fps, rng=rng, sigma=sigma)
        acc += judge(z, cfg)["success"]
    return acc / trials


def noise_sweep():
    sigmas = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
    rows = {}
    for s in sigmas:
        rng = np.random.default_rng(SEED + int(round(s * 1e5)))
        pos = {n: mc(n, s, NOMINAL_FPS, BASE_CFG, rng) for n in ["clean_lift_hold", "dip_then_long_hold", "slow_lift"]}
        neg = {n: mc(n, s, NOMINAL_FPS, BASE_CFG, rng) for n in ["early_drop", "dip_splits_hold", "near_miss_hold", "low_lift", "held_at_end", "static"]}
        frr = {n: 1 - v for n, v in pos.items()}
        rows[str(s)] = {"sigma_m": s, "FRR_positive_rejected": frr, "FAR_negative_accepted": neg,
                        "FRR_mean": float(np.mean(list(frr.values()))), "FAR_mean": float(np.mean(list(neg.values())))}
    return rows


def clause_breakdown(sigma, rng_seed):
    """clean_lift_hold가 어느 조건에서 떨어지는지, 잡음 σ별로 본다."""
    rng = np.random.default_rng(rng_seed)
    fails = {"descent": 0, "placement": 0, "hold": 0, "lift": 0, "final": 0}
    for _ in range(TRIALS):
        r = judge(trace("clean_lift_hold", NOMINAL_FPS, rng=rng, sigma=sigma), BASE_CFG)
        for k, ok in r["clauses"].items():
            if k in fails and not ok:
                fails[k] += 1
    return {k: v / TRIALS for k, v in fails.items()}


def tradeoff():
    """임계 하나씩 바꿔 FRR(깨끗한 성공 + 잡음)과 FAR(끊김 실패, 낮은 들기)을 함께 본다."""
    out = {}
    rng = np.random.default_rng(SEED + 7)
    out["min_hold_steps"] = {}
    for m in [10, 14, 16, 20, 26]:
        cfg = dict(BASE_CFG, min_hold_steps=m)
        out["min_hold_steps"][str(m)] = {
            "FAR_dip_splits_hold": mc("dip_splits_hold", 0.0, NOMINAL_FPS, cfg, rng),
            "FAR_near_miss_hold_sigma0": mc("near_miss_hold", 0.0, NOMINAL_FPS, cfg, rng),
            "FAR_near_miss_hold_sigma0.002": mc("near_miss_hold", 0.002, NOMINAL_FPS, cfg, rng),
            "FAR_early_drop": mc("early_drop", 0.0, NOMINAL_FPS, cfg, rng),
            "FRR_clean_sigma0": 1 - mc("clean_lift_hold", 0.0, NOMINAL_FPS, cfg, rng),
            "FRR_dip_then_long_hold_sigma0": 1 - mc("dip_then_long_hold", 0.0, NOMINAL_FPS, cfg, rng)
        }
    out["max_terminal_step_m_vs_noise"] = {}
    for step in [0.003, 0.005, 0.008, 0.012]:
        cfg = dict(BASE_CFG, max_terminal_step_m=step)
        out["max_terminal_step_m_vs_noise"][str(step)] = {
            f"FRR_clean_sigma{s}": 1 - mc("clean_lift_hold", s, NOMINAL_FPS, cfg, rng) for s in [0.0005, 0.001, 0.002]}
    out["max_descent_step_m_vs_noise"] = {}
    for d in [0.005, 0.01, 0.02]:
        cfg = dict(BASE_CFG, max_descent_step_m=d)
        out["max_descent_step_m_vs_noise"][str(d)] = {
            f"FRR_clean_sigma{s}": 1 - mc("clean_lift_hold", s, NOMINAL_FPS, cfg, rng) for s in [0.0, 0.001, 0.002]}
    out["lift_threshold_m_vs_low_lift"] = {}
    for lt in [0.07, 0.10, 0.12]:
        cfg = dict(BASE_CFG, lift_threshold_m=lt)
        out["lift_threshold_m_vs_low_lift"][str(lt)] = {
            "FAR_low_lift_0.08m": mc("low_lift", 0.0, NOMINAL_FPS, cfg, rng),
            "FRR_clean_sigma0": 1 - mc("clean_lift_hold", 0.0, NOMINAL_FPS, cfg, rng)}
    return out


def main():
    t0 = time.time()
    known = check_known_answers()
    fps = fps_sweep()
    noise = noise_sweep()
    breakdown = {str(s): clause_breakdown(s, SEED + 99 + int(round(s * 1e5))) for s in [0.0, 0.001, 0.002, 0.005]}
    to = tradeoff()
    results = {
        "script": "analysis/extra_lift_hold.py",
        "source": {"judge": "repo/robotic_grounding/groot_finetune/closed_loop/evaluation.py (LiftHoldTracker.update, complete; evaluate_lift_hold)",
                   "config": "repo/robotic_grounding/groot_finetune/task_profiles/tissue_box_lift_hold.json (evaluator 블록, 값은 손으로 옮김)",
                   "commit": "sparse clone 5226e34", "method": "식과 임계를 읽고 옮긴 독자 구현. 저장소 코드는 실행하지 않음"},
        "config": BASE_CFG,
        "port_notes": {"baseline": "첫 샘플이 기준 높이. evaluate_lift_hold에서 첫 샘플은 샘플 수에 들지 않음 (sample_count = N-1)",
                       "descent": "최대 들기가 hold_threshold 이상이 된 뒤에만 직전 샘플 대비 감소를 셈",
                       "placement": "마지막 placement_window 샘플(들기 기준)의 최댓값, 연속 차, 범위를 본다"},
        "nominal_fps": NOMINAL_FPS,
        "scenarios": {k: {"label": v["label"], "knots": v["knots"]} for k, v in SCENARIOS.items()},
        "known_answers": known,
        "fps_sweep": fps,
        "noise_sweep": {"trials_per_cell": TRIALS, "rows": noise},
        "clause_failure_rate_clean_lift_hold": breakdown,
        "tradeoff": to,
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "wall_time_s": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(json.dumps({"known": {k: v["ok"] for k, v in known.items()}, "fps": fps,
                      "noise": {k: (v["FRR_mean"], v["FAR_mean"]) for k, v in noise.items()}}, indent=1))


if __name__ == "__main__":
    main()

"""기하 타당성 게이트 toy (numpy만, CPU).

합성 '잡기-들기-내려놓기' 궤적(상자 물체, 손가락 끝 구 4개)을 만들고 네 가지 지표를 계산한다.
  (1) 손-물체 관통 깊이 (구 표면이 상자 안으로 들어간 깊이, mm): max 또는 p99
  (2) 물체 위치 가속도 p99 (m/s^2): 떨림
  (3) 프레임 간 물체 최대 이동량 (mm/frame): 튐·끊김
  (4) 물체가 움직이는 동안 접촉이 있는 비율: 떠 있음
정상 궤적(관통 0-2 mm, 측정 잡음 0.05-0.3 mm)과 결함 4종(깊은 관통, 떨림, 튐, 떠 있음)을 섞고,
임계값을 바꿀 때 거짓 기각(FRR, 정상을 버림)과 거짓 수용(FAR, 결함을 통과시킴)이 어떻게 맞바뀌는지 본다.

접촉 규칙은 robotic_grounding/scripts/retarget/contact_labels.py 와 같은 꼴이다:
|거리| < 18 mm 이고 2프레임 이상 연속일 때만 접촉. 이 규칙은 관통 방향을 보지 않는다.

실행: python3 -I analysis/gate_toy.py   → analysis/gate_toy_results.json
"""
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "gate_toy_results.json"

FPS = 30.0
DT = 1.0 / FPS
T = 300                                   # 10 s
HALF = np.array([0.03, 0.03, 0.05])       # 6 x 6 x 10 cm 상자 반치수 (m)
R_FINGER = 0.008                          # 손가락 끝 구 반지름 (m)
CONTACT_THR = 0.018                       # repo 기본 접촉 거리 (m)
DEBOUNCE = 2                              # repo 기본 연속 프레임 수
MIN_CONTACT = 0.95                        # 물체가 움직일 때 접촉 비율 하한
TAU_JUMP_MM = 10.0                        # 프레임 간 최대 이동 상한 (고정)
LIFT_T = (1.5, 6.5)                       # 들기 구간 (s)

# 상자 면의 점 (BASE)과 바깥 법선 (N). 구 중심은 BASE + N * (r - 관통깊이)
BASE = np.array([[HALF[0], 0, 0.02], [-HALF[0], 0, 0.02],
                 [0, HALF[1], -0.02], [0, -HALF[1], -0.02]])
NORMAL = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=float)
N_NORMAL = 400
N_PER_DEFECT = 100
SEEDS = [0, 1, 2]
KINDS = ["normal", "deep_pen", "jitter", "jump", "float"]


def min_jerk(tau):
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def box_sdf(p):
    """상자 표면까지 부호 거리 (바깥 +, 안 -). p: (..., 3) 물체 좌표계."""
    q = np.abs(p) - HALF
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)


def scene(rng):
    """참값 궤적. 물체 위치 (T,3), 손가락 구 중심 (T,4,3)."""
    t = np.arange(T) * DT
    lift = 0.10 * (min_jerk((t - LIFT_T[0]) / 3.0) - min_jerk((t - 6.5) / 2.0))
    obj = np.zeros((T, 3))
    obj[:, 2] = 0.05 + lift
    approach = 0.05 * (1.0 - min_jerk((t - 0.5) / 1.0))           # 1.5 s 전까지 바깥에서 접근
    e = rng.uniform(0.0, 0.002)                                     # 정상 관통 0-2 mm
    fingers = obj[:, None, :] + BASE[None] + NORMAL[None] * (R_FINGER - e + approach[:, None, None])
    return obj, fingers


def corrupt(obj, fingers, rng, kind):
    sigma = rng.uniform(0.00005, 0.0003)                            # 정상 측정 잡음 0.05-0.3 mm
    if kind == "deep_pen":                                          # 관통을 추가로 0.5-12 mm (정상 범위와 겹치는 약한 결함 포함)
        fingers = fingers - NORMAL[None] * rng.uniform(0.0005, 0.012)
    elif kind == "jitter":                                          # 물체 위치 떨림 0.1-4 mm (정상 잡음과 겹치는 약한 결함 포함)
        sigma = rng.uniform(0.0001, 0.004)
    elif kind == "jump":                                            # 1-3 프레임 15-40 mm 튐
        obj = obj.copy()
        start = rng.integers(60, T - 20)
        jump = rng.normal(size=3)
        obj[start:start + rng.integers(1, 4)] += jump / np.linalg.norm(jump) * rng.uniform(0.015, 0.04)
    elif kind == "float":                                           # 들 때만 손을 22-40 mm 바깥으로
        lifted = obj[:, 2] > obj[0, 2] + 0.002
        fingers = fingers + NORMAL[None] * (rng.uniform(0.022, 0.040) * lifted[:, None, None])
    obj = obj + rng.normal(0, sigma, obj.shape)
    fingers = fingers + rng.normal(0, sigma, fingers.shape)
    return obj, fingers


def contact_flags(g):
    """|g| < 18 mm 이고 DEBOUNCE 프레임 이상 연속인 구간만 접촉 (repo 규칙)."""
    active = np.abs(g) < CONTACT_THR
    out = np.zeros_like(active)
    for k in range(active.shape[1]):
        run = 0
        for i in range(active.shape[0]):
            run = run + 1 if active[i, k] else 0
            if run >= DEBOUNCE:
                out[i - DEBOUNCE + 1:i + 1, k] = True
    return out


def features(obj, fingers):
    rel = fingers - obj[:, None, :]
    g = box_sdf(rel) - R_FINGER                                    # 구 표면과 상자 표면 사이 간격
    depth_mm = np.maximum(-g, 0.0) * 1e3
    acc = np.linalg.norm(np.diff(obj, n=2, axis=0), axis=1) * FPS**2
    step_mm = np.linalg.norm(np.diff(obj, axis=0), axis=1) * 1e3
    moving = np.r_[False, step_mm > 1.0]                           # 1 mm/frame 초과를 움직임으로 봄
    contact_any = contact_flags(g).any(axis=1)
    return {
        "max_depth_mm": float(depth_mm.max()),
        "p99_depth_mm": float(np.percentile(depth_mm, 99)),
        "p99_acc": float(np.percentile(acc, 99)),
        "max_step_mm": float(step_mm.max()),
        "contact_frac_moving": float(contact_any[moving].mean()) if moving.any() else 1.0,
        "contact_exists": bool(contact_any.any()),
    }


def gate(f, tau_pen, tau_acc, pen_stat, min_contact=MIN_CONTACT):
    d = f["max_depth_mm"] if pen_stat == "max" else f["p99_depth_mm"]
    return (d <= tau_pen and f["p99_acc"] <= tau_acc
            and f["max_step_mm"] <= TAU_JUMP_MM and f["contact_frac_moving"] >= min_contact)


def run_seed(seed):
    rng = np.random.default_rng(seed)
    feats = {k: [] for k in KINDS}
    for k in KINDS:
        for _ in range(N_NORMAL if k == "normal" else N_PER_DEFECT):
            obj, fingers = scene(rng)
            feats[k].append(features(*corrupt(obj, fingers, rng, k)))
    return feats


def rates(feats, tau_pen, tau_acc, pen_stat, min_contact=MIN_CONTACT):
    frr = float(np.mean([not gate(f, tau_pen, tau_acc, pen_stat, min_contact) for f in feats["normal"]]))
    by = {k: float(np.mean([gate(f, tau_pen, tau_acc, pen_stat, min_contact) for f in feats[k]]))
          for k in KINDS if k != "normal"}
    defects = [f for k in KINDS if k != "normal" for f in feats[k]]
    far = float(np.mean([gate(f, tau_pen, tau_acc, pen_stat, min_contact) for f in defects]))
    return frr, far, by


def summarize(arr):
    out = {}
    for key in ["max_depth_mm", "p99_depth_mm", "p99_acc", "max_step_mm", "contact_frac_moving"]:
        v = np.array([f[key] for f in arr])
        out[key] = {"p05": float(np.percentile(v, 5)), "p50": float(np.median(v)), "p95": float(np.percentile(v, 95))}
    out["contact_exists_rate"] = float(np.mean([f["contact_exists"] for f in arr]))
    return out


def main():
    t0 = time.time()
    pen_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
    acc_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
    per_seed = []
    for seed in SEEDS:
        feats = run_seed(seed)
        grid = {}
        for stat in ["max", "p99"]:
            for tp in pen_grid:
                for ta in acc_grid:
                    frr, far, by = rates(feats, tp, ta, stat)
                    grid[f"{stat}|{tp}|{ta}"] = {"frr": frr, "far": far, "far_by_kind": by}
        contact_only = {k: float(np.mean([f["contact_exists"] and f["contact_frac_moving"] >= MIN_CONTACT
                                          for f in feats[k]])) for k in KINDS if k != "normal"}
        per_seed.append({"seed": seed, "grid": grid,
                         "summary": {k: summarize(feats[k]) for k in KINDS},
                         "contact_only_accept": contact_only})

    mean_grid = {}
    for key in per_seed[0]["grid"]:
        frr = np.array([s["grid"][key]["frr"] for s in per_seed])
        far = np.array([s["grid"][key]["far"] for s in per_seed])
        mean_grid[key] = {"frr_mean": float(frr.mean()), "frr_sd": float(frr.std()),
                          "far_mean": float(far.mean()), "far_sd": float(far.std())}
    best = {}
    for stat in ["max", "p99"]:
        cands = {k: v for k, v in mean_grid.items() if k.startswith(stat + "|")}
        k_best = min(cands, key=lambda k: cands[k]["frr_mean"] + cands[k]["far_mean"])
        best[stat] = {"key": k_best, **cands[k_best]}
    contact_only_mean = {k: float(np.mean([s["contact_only_accept"][k] for s in per_seed]))
                         for k in KINDS if k != "normal"}
    # 접촉 비율 하한 민감도: 최선 점(p99, 3.0 mm, 2.5 m/s^2)에서 0.90 / 0.95 / 0.98
    sens = {}
    for mc in [0.90, 0.95, 0.98]:
        vals = [rates(run_seed(sd), 3.0, 2.5, "p99", mc) for sd in SEEDS]
        sens[str(mc)] = {"frr_mean": float(np.mean([v[0] for v in vals])),
                         "far_mean": float(np.mean([v[1] for v in vals]))}
    res = {
        "config": {"fps": FPS, "T": T, "half_extent_m": HALF.tolist(), "finger_radius_m": R_FINGER,
                   "contact_thr_m": CONTACT_THR, "debounce_frames": DEBOUNCE, "min_contact_fraction": MIN_CONTACT,
                   "tau_jump_mm": TAU_JUMP_MM, "n_normal": N_NORMAL, "n_per_defect": N_PER_DEFECT,
                   "seeds": SEEDS, "pen_grid_mm": pen_grid, "acc_grid_mps2": acc_grid},
        "mean_over_seeds": mean_grid,
        "best_equal_weight_frr_plus_far": best,
        "contact_only_accept_rate_mean": contact_only_mean,
        "min_contact_sensitivity_p99_3.0_2.5": sens,
        "per_seed": per_seed,
        "runtime_s": round(time.time() - t0, 2),
    }
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({"best": best, "contact_only": contact_only_mean, "runtime_s": res["runtime_s"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

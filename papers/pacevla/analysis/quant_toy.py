"""시간 양자화 toy 검증 (CPU, numpy, 단일 스레드). 실행: python3 analysis/quant_toy.py

- T1: delta 회전을 블록으로 합칠 때 회전 벡터를 그냥 더하는 것과 SO(3) 합성(부록 식 6)의 차이
- T2: 2차원 손끝이 시연 경로를 따라가는 추적 toy — 고정 q=2, 적응형(정답 국면), 판정 잡음, 제어 이득
- T3: ES의 t_max 의존성 — Table 1 값으로 복원한 성공 궤적 길이에 t_max만 바꿔 ES 다시 계산
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
from pathlib import Path

import numpy as np

import table_check as tc

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)


# ── T1 SO(3) 집계 ─────────────────────────────────────────────────────────────
def exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def angle(R):
    return math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))


def t1():
    out = {}
    for q in (2, 4, 8):
        for th in (0.01, 0.03, 0.1):
            errs, net = [], []
            for _ in range(400):
                ws = [th * v / np.linalg.norm(v) for v in rng.normal(size=(q, 3))]
                R = np.eye(3)
                for w in ws:  # 공간 증분 규약: R_q ··· R_1
                    R = exp_so3(w) @ R
                errs.append(angle(exp_so3(np.sum(ws, axis=0)).T @ R))
                net.append(angle(R))
            out[f"q{q}_theta{th}"] = {"median_err_deg": round(math.degrees(float(np.median(errs))), 4),
                                      "median_net_rot_deg": round(math.degrees(float(np.median(net))), 3),
                                      "rel_err": round(float(np.median(errs) / np.median(net)), 4)}
    return out


# ── T2 추적 toy ───────────────────────────────────────────────────────────────
def demo_path(n=240):
    """reach(곡선) → grasp(멈춤에 가까운 정밀 구간) → carry → place(정밀 구간). 60 Hz 가정."""
    t = np.linspace(0, 1, n)
    g, p = np.array([0.30, 0.10]), np.array([0.05, 0.40])
    seg = []
    # reach: 원점에서 곡선으로 grasp 점까지 (0–0.35)
    for s in t[t < 0.35] / 0.35:
        seg.append(np.array([0.30 * s, 0.10 * s + 0.12 * math.sin(math.pi * s)]))
    # grasp: grasp 점 근처에서 천천히 접근·정지 (0.35–0.5)
    m = t[(t >= 0.35) & (t < 0.5)]
    for s in (m - 0.35) / 0.15:
        seg.append(g + np.array([0.0, -0.015 * (1 - s) ** 2]))
    # carry: grasp → place 위 (0.5–0.85)
    m = t[(t >= 0.5) & (t < 0.85)]
    for s in (m - 0.5) / 0.35:
        mid = (1 - s) * g + s * (p + np.array([0, 0.04]))
        seg.append(mid + np.array([0.0, 0.08 * math.sin(math.pi * s)]))
    # place: 천천히 내려놓기 (0.85–1)
    m = t[t >= 0.85]
    for s in (m - 0.85) / 0.15:
        seg.append(p + np.array([0, 0.04 * (1 - s) ** 2]))
    path = np.array(seg)
    grasp_idx = int(np.searchsorted(t, 0.5)) - 1
    place_idx = n - 1
    precise = np.zeros(n, bool)
    precise[int(np.searchsorted(t, 0.33)):grasp_idx + 1] = True
    precise[int(np.searchsorted(t, 0.83)):] = True
    return path, grasp_idx, place_idx, precise


def run(path, precise_mask, q_free, q_prec, kp, noise, eps=0.006, settle=12):
    """절대 위치 명령을 블록 끝값으로 양자화(식 5), 2차 PD 추적. 명령 1개 = 제어 1주기."""
    n = len(path)
    demo = path + noise * rng.normal(size=path.shape)  # 텔레옵 떨림
    cmds, src_idx, i = [], [], 0
    while i < n - 1:
        q = q_prec if precise_mask[i] else q_free
        j = min(n - 1, i + q)
        cmds.append(demo[j])
        src_idx.append(j)
        i = j
    kd = 2 * math.sqrt(kp)
    dt = 1.0 / 60
    x, v = demo[0].copy(), np.zeros(2)
    err_at = {}
    traj = []
    for c, j in zip(cmds, src_idx):
        a = kp * (c - x) - kd * v
        v = v + a * dt
        x = x + v * dt
        traj.append(x.copy())
        err_at[j] = float(np.linalg.norm(x - path[j]))
    for _ in range(settle):  # 마지막 명령 뒤 안착
        a = kp * (cmds[-1] - x) - kd * v
        v = v + a * dt
        x = x + v * dt
    return cmds, err_at, x, np.array(traj)


def episode_metrics(path, gi, pi, mask, qf, qp, kp, tol, n_ep=200):
    succ, steps = [], []
    for _ in range(n_ep):
        cmds, err_at, xf, _ = run(path, mask, qf, qp, kp, noise=0.0015)
        # 쥐는 순간: 그 시연 step이 명령에 없으면(블록 안에 묻히면) 가장 가까운 명령 시점에서 판정
        keys = np.array(sorted(err_at))
        eg = err_at[int(keys[np.argmin(np.abs(keys - gi))])]
        ep = float(np.linalg.norm(xf - path[pi]))
        succ.append(eg < tol and ep < tol)
        steps.append(len(cmds))
    return float(np.mean(succ)), float(np.mean(steps))


def window_mask(n, events, W, delay, after=4):
    """사건(쥐기·놓기) 앞 W step부터 세밀 모드. 판정이 delay step 늦으면 시작도 늦어짐."""
    m = np.zeros(n, bool)
    for e in events:
        m[max(0, e - W + delay):min(n, e + after)] = True
    return m


def t2():
    path, gi, pi, _ = demo_path()
    n = len(path)
    out = {"demo_steps": n, "tol_m": 0.004, "rows": {}}
    for kp in (150.0, 300.0, 600.0):
        sr_b, st_b = episode_metrics(path, gi, pi, np.ones(n, bool), 1, 1, kp, 0.004)
        sr_f, st_f = episode_metrics(path, gi, pi, np.zeros(n, bool), 2, 2, kp, 0.004)
        out["rows"][f"kp{int(kp)}/base q=1"] = {"SR": round(sr_b, 3), "speedup": 1.0}
        out["rows"][f"kp{int(kp)}/fixed q=2"] = {"SR": round(sr_f, 3), "speedup": round(st_b / st_f, 3)}
        for W in (6, 12, 24, 48):
            for d in (0, 6, 12):
                mask = window_mask(n, (gi, pi), W, d)
                sr, st = episode_metrics(path, gi, pi, mask, 2, 1, kp, 0.004)
                out["rows"][f"kp{int(kp)}/adaptive W{W} delay{d}"] = {
                    "SR": round(sr, 3), "speedup": round(st_b / st, 3), "fine_frac": round(float(mask.mean()), 3)}
    return out


# ── T3 ES의 t_max 의존성 ──────────────────────────────────────────────────────
def t3():
    out = {}
    g = tc.T1["GR00T N1.5"]
    for j, col in ((0, "RoboCasa PnP"), (1, "RoboCasa Others")):
        Lb = tc.harmonic_len(g["base"][j][0], g["base"][j][1], 1500)
        row = {}
        for m in tc.METHODS[1:]:
            L = tc.harmonic_len(g[m][j][0], g[m][j][1], 1500)
            row[m] = {str(tm): round(tc.T_per_success(g["base"][j][0], Lb, tm) / tc.T_per_success(g[m][j][0], L, tm), 3)
                      for tm in (0, 300, 720, 1500, 3000)}
        out[col] = row
    out["note"] = "t_max=0은 실패 비용이 없는 경우 = 성공 궤적 step 비. 720은 GR00T 저장소 RoboCasa 평가 기본 max-episode-steps"
    return out


def main() -> None:
    out = {"T1_so3_aggregation": t1(), "T2_tracking_toy": t2(), "T3_es_vs_tmax": t3()}
    path = ROOT / "analysis" / "quant_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])
    print("wrote", path)


if __name__ == "__main__":
    main()

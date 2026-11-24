"""토이: 이진 접촉 이벤트 보정(식 1-2)이 무엇을 맞추고 무엇을 못 맞추는지 보는 합성 실험 (numpy, 1분 미만).

논문 §3.1의 모델을 1차원 핀 배열로 줄였다.
  - 시뮬: 표면 노드의 접촉 압력 f_j = k * max(-sdf_j, 0), 가상 택셀 i의 전파 압력 f_i = sum_j exp(-alpha |p_j - t_i|) f_j,
          f_i > lam_pre 이면 활성 (식 1). SDF 임계 lam_con은 FSR 쪽 규칙이라 여기서는 노드 활성 임계로 쓴다.
  - 실제 센서(합성): 조건 A는 같은 형태(참 파라미터), 조건 B는 형태가 다른 센서(지연 + 히스테리시스 + 느린 해제).
  - 보정: 10 Hz로 기록한 9개 궤적에서 식 2의 l1 불일치를 격자 탐색으로 최소화하고, 보정에 쓰지 않은 궤적에서 평가한다.
지표는 논문 부록 B.2와 같은 세 가지: 접촉 시작 오차(ms), 활성 F1, FPR.
이 토이는 논문 실험의 재현이 아니다. "세 파라미터 격자 탐색이 어떤 오차를 줄이고, 센서 형태가 다를 때 어디서 멈추는가"만 본다.

실행: python3 analysis/toy_event_calibration.py   (결과: toy_event_calibration_results.json)
"""
import json
import itertools
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_event_calibration_results.json"
FS_SIM = 1000          # 시뮬 내부 시간 해상도(Hz)
FS_LOG = 10            # 보정 데이터 기록 주기(Hz): 논문 부록 A의 약 10 Hz
NODES = np.linspace(0, 28, 15)        # 표면 노드 위치(mm)
TAXELS = np.linspace(2, 26, 8)        # 가상 택셀 8개(TwinTac 하나)
K = 1.0


def make_traj(rng, dur):
    """손가락이 물체에 눌렀다 떼는 궤적: 노드별 침투 깊이(mm) d_j(t)를 만든다."""
    t = np.arange(0, dur, 1 / FS_SIM)
    depth = np.zeros((len(t), len(NODES)))
    n_press = rng.integers(2, 5)
    for _ in range(n_press):
        t0 = rng.uniform(0.5, dur - 4)
        rise, hold, fall = rng.uniform(0.3, 1.2), rng.uniform(0.3, 1.0), rng.uniform(0.3, 1.2)
        amp = rng.uniform(0.3, 2.0)       # 최대 침투(mm)
        c = rng.uniform(2, 26)            # 접촉 중심(mm)
        w = rng.uniform(2, 8)             # 접촉 폭(mm)
        prof = np.interp(t, [t0, t0 + rise, t0 + rise + hold, t0 + rise + hold + fall], [0, 1, 1, 0], left=0, right=0)
        depth += amp * prof[:, None] * np.exp(-0.5 * ((NODES - c) / w) ** 2)[None, :]
    return t, depth


def sim_taxels(depth, lam_con, alpha, lam_pre):
    """식 1: 노드 압력 -> 지수 전파 -> 임계. lam_con은 노드가 접촉으로 인정되는 최소 침투(mm)."""
    f = K * np.where(depth > lam_con, depth - lam_con, 0.0)
    W = np.exp(-alpha * np.abs(NODES[:, None] - TAXELS[None, :]))
    prop = f @ W
    return prop > lam_pre


def real_taxels(depth, rng, mode):
    """합성 '실제' 센서. A: 같은 형태, B: 지연 + 히스테리시스 + 느린 해제."""
    if mode == "A":
        return sim_taxels(depth, TRUE["lam_con"], TRUE["alpha"], TRUE["lam_pre"])
    f = K * depth
    W = np.exp(-0.25 * np.abs(NODES[:, None] - TAXELS[None, :]))
    prop = f @ W
    lag = int(0.08 * FS_SIM)                      # 80 ms 지연
    prop = np.vstack([np.zeros((lag, prop.shape[1])), prop[:-lag]])
    on, off = 0.9, 0.4                            # 켜는 임계 > 끄는 임계 (히스테리시스)
    state = np.zeros(prop.shape[1], bool)
    out = np.zeros_like(prop, bool)
    alpha_slow = 1 - np.exp(-1 / (0.12 * FS_SIM))  # 해제 시 120 ms 시상수
    filt = np.zeros(prop.shape[1])
    for i in range(len(prop)):
        filt = np.where(prop[i] > filt, prop[i], filt + alpha_slow * (prop[i] - filt))
        state = np.where(filt > on, True, np.where(filt < off, False, state))
        out[i] = state
    flip = rng.random(out.shape[1]) < 0.0
    return out ^ flip


TRUE = {"lam_con": 0.15, "alpha": 0.35, "lam_pre": 0.55}


def metrics(real, sim):
    """부록 B.2의 세 지표. 입력은 10 Hz로 줄인 (시간, 택셀) 이진 배열의 목록."""
    onsets, missed, total = [], 0, 0
    tp = fp = fn = tn = 0
    for r, s in zip(real, sim):
        tp += int((r & s).sum()); fp += int((~r & s).sum()); fn += int((r & ~s).sum()); tn += int((~r & ~s).sum())
        for i in range(r.shape[1]):
            if r[:, i].any():
                total += 1
                if not s[:, i].any():
                    missed += 1          # 시뮬이 한 번도 켜지지 않은 택셀은 시작 오차에서 빼고 따로 센다
                    continue
                a, b = int(np.argmax(r[:, i])), int(np.argmax(s[:, i]))
                onsets.append(abs(a - b) * 1000 / FS_LOG)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"onset_ms": round(float(np.mean(onsets)), 1), "onset_missed_pct": round(100 * missed / max(total, 1), 1), "F1_pct": round(100 * f1, 1), "FPR_pct": round(100 * fp / max(fp + tn, 1), 2),
            "active_share_pct": round(100 * (tp + fn) / (tp + fn + fp + tn), 1)}


def down(x):
    step = FS_SIM // FS_LOG
    return x[::step]


def run(mode, seed):
    rng = np.random.default_rng(seed)
    cal, test = [], []
    for k in range(9 + 30):
        t, depth = make_traj(rng, dur=rng.uniform(30, 60))
        real = real_taxels(depth, rng, mode)
        (cal if k < 9 else test).append((depth, real))
    grid = list(itertools.product(np.arange(0.0, 0.65, 0.05), np.arange(0.1, 0.65, 0.05), np.arange(0.1, 1.05, 0.05)))
    best, best_loss = None, 1e18
    losses = {}
    cal_ds = [(down(d), down(r)) for d, r in cal]
    for lc, al, lp in grid:
        loss = sum(int((sim_taxels(d, lc, al, lp) ^ r).sum()) for d, r in cal_ds)   # 식 2의 l1 (이진이므로 불일치 개수)
        losses[(lc, al, lp)] = loss
        if loss < best_loss:
            best, best_loss = (lc, al, lp), loss
    near = [g for g, v in losses.items() if v <= best_loss * 1.02]
    nominal = (0.0, 0.5, 0.2)    # 보정 전 '공칭' 값(임의로 정함)
    res = {"best_params": [round(float(x), 3) for x in best], "n_grid_points": len(grid), "n_within_2pct_of_best_loss": len(near)}
    for name, params in [("nominal", nominal), ("calibrated", best)]:
        for split, data in [("cal", cal), ("test", test)]:
            R = [down(r) for _, r in data]
            S = [down(sim_taxels(d, *params)) for d, _ in data]
            res[f"{name}_{split}"] = metrics(R, S)
    return res


if __name__ == "__main__":
    out = {"true_params_mode_A": TRUE, "log_rate_Hz": FS_LOG, "n_cal_traj": 9, "n_test_traj": 30,
           "A_same_form": [run("A", s) for s in range(3)], "B_different_form": [run("B", s) for s in range(3)]}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for mode in ("A_same_form", "B_different_form"):
        print(mode)
        for r in out[mode]:
            print(" ", r["best_params"], "grid_near_best", r["n_within_2pct_of_best_loss"], "| nominal test", r["nominal_test"], "| calibrated test", r["calibrated_test"])

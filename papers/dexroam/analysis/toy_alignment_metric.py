"""토이: 논문 그림 6의 MMD/SWD가 무엇을 재는가. 결과는 toy_alignment_metric_results.json.

실행: python3 analysis/toy_alignment_metric.py   (numpy, scipy, 약 20-40초)

논문 부록 E.1의 지표를 그대로 구현한다: 차원별 강건 정규화(중앙값, IQR/1.349, 하한 1e-3), 같은 수의 표본(여기서는 도메인당 2,000개,
논문은 최대 4,000개), 가우시안 RBF 편향 V-통계량 MMD(대역폭은 합친 표본의 제곱 거리 중앙값, 표본 1,500개로 추정),
SWD는 무작위 방향 L = 128개의 1차 슬라이스 W1. 논문은 56차원 행동 공간을 쓴다.

A) 보정: 56차원 표준 가우시안 두 집합에서 모든 차원을 s만큼 평행이동했을 때의 MMD와 SWD. s = 0은 같은 분포의 바닥값이다.
B) 구조: 사람과 로봇이 같은 행동(같은 과제 진행 곡선)을 하지만 (i) 사람의 절대 자세에 시연자별 오프셋(키, 팔 길이, 시작 위치)이 있고
   (ii) 사람 에피소드가 절반 길이일 때, 지표가 세 단계에서 어떻게 변하는가: 절대 행동(raw) -> 상대 행동(원래 시간축) -> 상대 행동 + 과제 진행 재표집.
   팔·몸통·베이스 30차원이 상대(변화량)가 되고 손·머리 26차원은 절대로 남는다(논문 표 VII의 구성).
C) 대조: B와 같지만 사람이 실제로 다른 행동을 하는 경우(곡선에 독립 부드러운 성분 delta를 더함). 정렬 뒤 지표가 이 차이를 읽는지 본다.
모든 모형은 우리가 만든 것이고 논문의 실제 데이터와 분포가 다르다. 지표 값의 절대 크기를 논문 값에 옮겨 읽으면 안 된다.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "toy_alignment_metric_results.json"
D = 56
HAND = slice(0, 24)
POSE = slice(24, 51)
HEAD = slice(51, 53)
BASE = slice(53, 56)
REL_DIMS = np.r_[24:51, 53:56]  # 상대(변화량)로 바꾸는 차원 30개
N_SAMPLE = 2000
L_PROJ = 128


def robust_normalize(x, y):
    pooled = np.concatenate([x, y])
    med = np.median(pooled, axis=0)
    scale = np.maximum((np.percentile(pooled, 75, axis=0) - np.percentile(pooled, 25, axis=0)) / 1.349, 1e-3)
    return (x - med) / scale, (y - med) / scale


def mmd(x, y, rng):
    pooled = np.concatenate([x, y])
    idx = rng.choice(len(pooled), size=min(1500, len(pooled)), replace=False)
    b = np.median(cdist(pooled[idx], pooled[idx], "sqeuclidean"))
    kxx = np.exp(-cdist(x, x, "sqeuclidean") / (2 * b)).mean()
    kyy = np.exp(-cdist(y, y, "sqeuclidean") / (2 * b)).mean()
    kxy = np.exp(-cdist(x, y, "sqeuclidean") / (2 * b)).mean()
    return float(np.sqrt(max(kxx + kyy - 2 * kxy, 0.0)))


def swd(x, y, rng):
    dirs = rng.standard_normal((L_PROJ, x.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    px = np.sort(x @ dirs.T, axis=0)
    py = np.sort(y @ dirs.T, axis=0)
    return float(np.abs(px - py).mean())


def metrics(x, y, rng):
    n = min(len(x), len(y), N_SAMPLE)
    x = x[rng.choice(len(x), n, replace=False)]
    y = y[rng.choice(len(y), n, replace=False)]
    xn, yn = robust_normalize(x, y)
    return mmd(xn, yn, rng), swd(xn, yn, rng)


def make_template(rng, K=3):
    amp = rng.standard_normal((D, K)) / np.sqrt(K)
    phase = rng.uniform(0, 2 * np.pi, (D, K))
    freq = np.arange(1, K + 1)

    def f(s):
        return np.sum(amp[None] * np.sin(2 * np.pi * freq[None, None] * s[:, None, None] + phase[None]), axis=2)

    return f


def to_stage_features(abs_traj, relative):
    """절대 궤적 (T, 56) -> 행동 특징. relative이면 30차원을 프레임 간 변화량으로 바꾼다(길이 T-1)."""
    if not relative:
        return abs_traj
    out = abs_traj[1:].copy()
    out[:, REL_DIMS] = np.diff(abs_traj[:, REL_DIMS], axis=0)
    return out


def resample(traj, n_new):
    s_old = np.linspace(0, 1, len(traj))
    s_new = np.linspace(0, 1, n_new)
    return np.stack([np.interp(s_new, s_old, traj[:, j]) for j in range(traj.shape[1])], axis=1)


def make_demos(rng, tmpl, n_demos, length_mean, length_sd, offset_sd, behavior_f=None, behavior_delta=0.0, tmpl2=None):
    demos = []
    for _ in range(n_demos):
        n = max(40, int(rng.normal(length_mean, length_sd)))
        s = np.linspace(0, 1, n)
        a = tmpl(s) * (1 + 0.05 * rng.standard_normal())
        if tmpl2 is not None:
            a = a + behavior_delta * tmpl2(s)
        a = a + 0.02 * rng.standard_normal(a.shape)
        off = np.zeros(D)
        off[POSE] = offset_sd * rng.standard_normal(27)
        off[BASE] = 2 * offset_sd * rng.standard_normal(3)
        demos.append(a + off)
    return demos


def run_stage(robot, human, rng, mode, n_robot_mean):
    if mode == "raw":
        rx = np.concatenate([to_stage_features(d, False) for d in robot])
        hx = np.concatenate([to_stage_features(d, False) for d in human])
    elif mode == "relative_native":
        rx = np.concatenate([to_stage_features(d, True) for d in robot])
        hx = np.concatenate([to_stage_features(d, True) for d in human])
    else:  # relative_resampled: 사람 궤적을 로봇 평균 길이로 먼저 재표집한 뒤 변화량
        rx = np.concatenate([to_stage_features(d, True) for d in robot])
        hx = np.concatenate([to_stage_features(resample(d, n_robot_mean), True) for d in human])
    return metrics(rx, hx, rng)


def main():
    res = {"A_calibration": {}, "B_structure": {}, "C_behavior_diff": {}}
    rng0 = np.random.default_rng(0)

    # ---- A) 보정: 56차원 가우시안, 모든 차원을 s만큼 평행이동
    for s in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        vals = []
        for seed in range(3):
            rng = np.random.default_rng(100 + seed)
            x = rng.standard_normal((N_SAMPLE, D))
            y = rng.standard_normal((N_SAMPLE, D)) + s
            vals.append(metrics(x, y, rng))
        v = np.array(vals)
        res["A_calibration"][f"shift_{s}"] = {"MMD": float(v[:, 0].mean()), "SWD": float(v[:, 1].mean())}

    # ---- B, C) 구조 토이
    n_robot_mean = 300
    for label, behavior_delta, off_sd in (("B_structure", 0.0, 0.5), ("C_behavior_diff", 0.35, 0.5)):
        stage_vals = {"raw": [], "relative_native": [], "relative_resampled": [], "floor_robot_split": []}
        for seed in range(3):
            rng = np.random.default_rng(10 + seed)
            tmpl = make_template(rng)
            tmpl2 = make_template(rng)
            robot = make_demos(rng, tmpl, 40, n_robot_mean, 30, 0.0)
            human = make_demos(rng, tmpl, 40, n_robot_mean / 2, 20, off_sd, behavior_delta=behavior_delta, tmpl2=tmpl2)
            for mode in ("raw", "relative_native", "relative_resampled"):
                stage_vals[mode].append(run_stage(robot, human, rng, mode, n_robot_mean))
            # 바닥값: 로봇 에피소드를 두 반으로 나눠 같은 분포끼리 비교(상대 행동)
            half = len(robot) // 2
            a = np.concatenate([to_stage_features(d, True) for d in robot[:half]])
            b = np.concatenate([to_stage_features(d, True) for d in robot[half:]])
            stage_vals["floor_robot_split"].append(metrics(a, b, rng))
        for k, v in stage_vals.items():
            arr = np.array(v)
            res[label][k] = {"MMD": float(arr[:, 0].mean()), "SWD": float(arr[:, 1].mean()),
                             "MMD_sd": float(arr[:, 0].std()), "SWD_sd": float(arr[:, 1].std())}
        for m in ("MMD", "SWD"):
            raw = res[label]["raw"][m]
            al = res[label]["relative_resampled"][m]
            nat = res[label]["relative_native"][m]
            res[label][f"{m}_reduction_raw_to_aligned"] = float(1 - al / raw)
            res[label][f"{m}_reduction_raw_to_relative_native"] = float(1 - nat / raw)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT)
    for k, v in res["A_calibration"].items():
        print("A", k, round(v["MMD"], 3), round(v["SWD"], 3))
    for lab in ("B_structure", "C_behavior_diff"):
        for k in ("raw", "relative_native", "relative_resampled", "floor_robot_split"):
            print(lab, k, round(res[lab][k]["MMD"], 3), round(res[lab][k]["SWD"], 3))
        print(lab, "reduction MMD/SWD", round(res[lab]["MMD_reduction_raw_to_aligned"], 3), round(res[lab]["SWD_reduction_raw_to_aligned"], 3))


if __name__ == "__main__":
    main()

"""장난감 검증: "디노이징 속도 갱신"과 "행동 잔차"는 선형 머리에서 같은 함수인가, 관측 없는 피드백 모듈은 이득이 있는가.

논문 식 (6): a^k = a^(k-1) + dk * v_hat,  학습 손실 식 (8): ||a^k - a_gt||^2 (속도가 아니라 최종 행동에 건 회귀).
논문 4.4 절은 속도 갱신이 행동 잔차 Delta a보다 낫다고 하고(그림 5a), 관측을 뺀 변형(No Obs)도 보고한다.

1차원 설정: 목표 위치가 2 s마다 속도 {0, -0.04} m/s 중 하나로 바뀌고, 계획기는 청크 시작 시점의 위치를 청크 전체의 목표로 낸다(낡은 청크).
피드백 모듈은 (근사 행동, 현재 관측 위치, 청크 내 위치)에서 선형 머리로 속도를 낸다.
세 가지를 본다.
  (1) 선형 머리와 최종 행동 회귀에서는 dk가 달라도(0.25, 0.5, 1.0) 닫힌형 최적해의 오차가 같다.
  (2) Adam으로 같은 스텝 수만큼 학습하면 dk가 작을수록 느리다(가중치가 1/dk배 커야 같은 함수가 된다).
  (3) 관측을 뺀 모듈도 계획기의 체계적 편향(여기서는 평균 표류)을 배워 오차를 줄인다. 따라서 No Obs 변형의 이득은
      "실시간 관측의 효과"가 아니라 "학습되는 보정기의 효과"이며, 관측의 순수 효과는 No Obs 대비로 따로 봐야 한다.
이 장난감은 실제 VLA 결과를 재현하지 않는다. 식 (6)과 (8)의 대수적 성질만 확인한다.
실행: python3 analysis/toy_feedback_param.py   (numpy, 수 초)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_feedback_param_results.json")
rng = np.random.default_rng(0)

H, DT_CTRL, SPEEDS = 16, 0.1, (0.0, -0.04)
SIGMA_PLAN, SIGMA_OBS = 0.002, 0.002  # 계획기 오차, 관측 잡음 (m)
N_CHUNKS = 4000


def make_data(n):
    x0 = rng.uniform(0.0, 0.3, n)                       # 청크 시작 목표 위치
    v = rng.choice(SPEEDS, size=(n, 2))                 # 2 s(20 스텝) 단위로 바뀌는 속도 두 구간
    steps = np.arange(H)
    vel = np.where(steps[None, :] < 8, v[:, [0]], v[:, [1]])   # 8 스텝(0.8 s) 뒤 속도 전환(낡은 청크가 생기도록)
    x = x0[:, None] + np.cumsum(vel, axis=1) * DT_CTRL        # 각 스텝의 목표 위치 = 정답 행동
    a_prev = x0[:, None] + rng.normal(0, SIGMA_PLAN, (n, 1)) + 0 * x   # 계획기: 시작 위치를 청크 전체에 사용
    obs = x + rng.normal(0, SIGMA_OBS, x.shape)               # 스텝별 현재 관측
    return a_prev, obs, x


def features(a_prev, obs, with_obs):
    n = a_prev.shape[0]
    pos = np.tile(np.arange(H) / H, (n, 1))
    cols = [a_prev, pos, np.ones_like(a_prev)]
    if with_obs:
        cols.insert(1, obs - a_prev)   # 관측과 근사 행동의 차이 (선형 머리가 쓸 수 있는 가장 단순한 신호)
    return np.stack(cols, axis=-1).reshape(n * H, -1)


a_tr, o_tr, x_tr = make_data(N_CHUNKS)
a_te, o_te, x_te = make_data(N_CHUNKS)
mse = lambda a, b: float(np.mean((a - b) ** 2))
res = {"planner_only_mse": mse(a_te, x_te)}


def fit_closed_form(dk, with_obs):
    """v_hat = F w,  a = a_prev + dk * v_hat,  최종 행동 회귀의 최소제곱 해."""
    F = features(a_tr, o_tr, with_obs)
    y = ((x_tr - a_tr) / dk).reshape(-1)      # 속도 목표 = 잔차/dk
    w = np.linalg.lstsq(F, y, rcond=None)[0]
    pred = a_te + dk * (features(a_te, o_te, with_obs) @ w).reshape(-1, H)
    return mse(pred, x_te)


res["closed_form_mse"] = {f"dk={dk}": fit_closed_form(dk, True) for dk in (0.25, 0.5, 1.0)}
res["closed_form_mse_no_obs"] = {f"dk={dk}": fit_closed_form(dk, False) for dk in (0.25, 0.5, 1.0)}


def fit_adam(dk, with_obs, steps=300, lr=1e-2, batch=2048, seed=1):
    r = np.random.default_rng(seed)
    F = features(a_tr, o_tr, with_obs)
    ytrue = (x_tr - a_tr).reshape(-1)
    base = a_tr.reshape(-1)
    w = r.normal(0, 0.1, F.shape[1])
    m = np.zeros_like(w); v = np.zeros_like(w)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        idx = r.integers(0, F.shape[0], batch)
        pred_res = dk * (F[idx] @ w)
        g = 2 * ((pred_res - ytrue[idx])[:, None] * dk * F[idx]).mean(0)   # d(최종 행동 MSE)/dw
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        w -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    pred = a_te + dk * (features(a_te, o_te, with_obs) @ w).reshape(-1, H)
    return mse(pred, x_te)


res["adam_mse_300steps"] = {f"dk={dk}": float(np.mean([fit_adam(dk, True, seed=s) for s in range(5)])) for dk in (0.25, 0.5, 1.0)}
res["note"] = ("closed_form: dk와 무관. adam: 같은 lr·스텝에서 dk가 작을수록 느림. "
               "no_obs도 planner_only보다 오차가 작다(체계적 편향 보정). 관측까지 쓰면 한 단계 더 줄어든다.")
res["no_obs_gain_ratio"] = res["closed_form_mse_no_obs"]["dk=0.5"] / res["planner_only_mse"]
res["obs_gain_ratio"] = res["closed_form_mse"]["dk=0.5"] / res["planner_only_mse"]

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))

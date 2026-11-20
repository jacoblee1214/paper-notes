"""토이: 궤적 유사도 보상(verified reward)이 성공을 가려내는가 (numpy, CPU, 1초 안팎).

우리가 만든 단순 모형이다. 논문의 실제 과제와 다르다.
- 2D 점 질량이 (0,0)에서 (1,0)으로 T=30 프레임 이동한다. 참조(goal-achieving) 궤적은 직선 단계 함수다.
- 성공 A: 참조와 같은 경로에 작은 잡음을 얹었다.
- 성공 B: 같은 목표에 도달하지만 옆으로 돌아가는 유효한 다른 경로다 (detour 진폭 d).
- 실패 F: 목표의 f(0.3–0.8) 지점에서 멈춘다.
- 세계모델 오차: 궤적 위치의 결정적 함수 e(x) = σ sin(2πx)·cos(2πy)로 모형화하고, 정책과 참조를 같은 세계모델로 굴린다
  (논문 reward type 3과 같은 구조). 보상 R = -mean_t |(x_pol + e(x_pol)) - (x_ref + e(x_ref))| (LPIPS 항은 생략).
- 비교용 보상 type 1 = -mean_t |x_pol - x_ref| (세계모델 없음, 참조 행동에 해당).

결과: 성공(A+B)과 실패(F)의 AUC, 그리고 유효한 다른 경로(B)가 실패의 중앙값보다 낮게 채점되는 비율.
실행: python3 analysis/toy_reward_validity.py → analysis/toy_reward_validity_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_reward_validity_results.json")
T, N = 30, 1500
s0, g = np.array([0.0, 0.0]), np.array([1.0, 0.0])
u = np.linspace(0, 1, T)
smooth = 3 * u ** 2 - 2 * u ** 3  # 참조 진행 함수


def path(kind, rng, detour=0.0, f=None):
    if kind == "ref":
        prog = smooth
    elif kind == "fail":
        prog = smooth * f  # 목표 일부만 간다
    else:
        prog = smooth
    x = s0 + (g - s0) * prog[:, None]
    if kind == "detour":
        x = x + detour * np.sin(np.pi * u)[:, None] * np.array([0.0, 1.0])
    return x


def wm_err(x, sigma):
    e = sigma * np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1])
    return np.stack([e, e], axis=-1)


def reward_wm(x, ref, sigma):
    a = x + wm_err(x, sigma)
    b = ref + wm_err(ref, sigma)
    return -np.mean(np.abs(a - b))


def reward_plain(x, ref):
    return -np.mean(np.abs(x - ref))


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1
    npos, nneg = len(pos), len(neg)
    return (r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def run(detour, sigma, seed=0):
    rng = np.random.default_rng(seed)
    ref = path("ref", rng)
    succ_a, succ_b, fail = [], [], []
    for _ in range(N // 3):
        xa = ref + rng.normal(0, 0.02, ref.shape)
        succ_a.append(reward_wm(xa, ref, sigma))
        xb = path("detour", rng, detour=detour) + rng.normal(0, 0.02, ref.shape)
        succ_b.append(reward_wm(xb, ref, sigma))
        xf = path("fail", rng, f=rng.uniform(0.3, 0.8)) + rng.normal(0, 0.02, ref.shape)
        fail.append(reward_wm(xf, ref, sigma))
    succ_a, succ_b, fail = map(np.array, (succ_a, succ_b, fail))
    fail_med = np.median(fail)
    return {
        "detour": detour, "sigma": sigma,
        "auc_success_vs_fail_wm_reward": round(float(auc(np.concatenate([succ_a, succ_b]), fail)), 3),
        "auc_detour_success_vs_fail": round(float(auc(succ_b, fail)), 3),
        "frac_detour_success_below_median_fail": round(float((succ_b < fail_med).mean()), 3),
        "frac_ref_like_success_below_median_fail": round(float((succ_a < fail_med).mean()), 3),
    }


grid = [run(d, s) for d in [0.0, 0.1, 0.3, 0.5] for s in [0.0, 0.05, 0.2]]
# 참조와 같은 경로만 쓰면 보상은 성공을 잘 가린다. 다른 유효 경로는 참조와 다르므로 낮게 채점된다.
res = {"setup": {"T": T, "n_per_class": N // 3, "seed": 0, "note": "우리가 만든 모형. 논문 과제와 다름"},
       "grid": grid}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
for row in grid:
    print(row)

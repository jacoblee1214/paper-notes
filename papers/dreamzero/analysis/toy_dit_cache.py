"""토이 1: DiT 캐시(속도 재사용)가 샘플 정확도를 얼마나 잃는지. 논문 3.2.3절 '16 → 4단계, 품질 손실 최소'의 방향을 본다.

설정: 2차원, 4개 가우시안 혼합(반지름 2, 표준편차 0.3)을 데이터로 둔다. 흐름 정합의 정확한 속도장
  v(x,t) = (E[x1|x_t] - x) / (1 - t)  (x_t = t x1 + (1-t) x0, x0 ~ N(0,I))
을 닫힌 식으로 계산해 '완벽한 DiT'로 쓴다. 실제 DiT는 이보다 오차가 크므로 이 토이의 손실은 아래한계다.
스케줄은 Euler, 균등 간격(코드의 sigma_shift=5는 넣지 않음). 참조해는 같은 잡음에서 512단계 Euler.
DiT 호출 방식: 호출하지 않은 단계는 직전 속도를 재사용한다(공개 코드의 캐시 방식과 같은 가정).

비교: 16단계 전부, 코드 기본 mask(8회), 코드 최소 mask(5회), 논문 수치(4회, 균등 배치, 코드에 없음), 코드 동적 규칙(코사인 0.95 → 4단계 건너뜀, 0.93 → 2단계).
실행: python3 analysis/toy_dit_cache.py  (numpy, 약 5초)  → analysis/toy_dit_cache_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_dit_cache_results.json")
SEED = 0
N_SAMPLES = 4000
MU = 2.0 * np.array([[np.cos(a), np.sin(a)] for a in np.arange(4) * np.pi / 2])
SIG = 0.3


def velocity(x, t):
    # 혼합 성분 k: x_t | k ~ N(t mu_k, (t^2 s^2 + (1-t)^2) I)
    var = t * t * SIG * SIG + (1 - t) ** 2                       # 스칼라
    d2 = ((x[:, None, :] - t * MU[None, :, :]) ** 2).sum(-1)     # (n, K)
    logw = -0.5 * d2 / var
    logw -= logw.max(1, keepdims=True)
    r = np.exp(logw)
    r /= r.sum(1, keepdims=True)
    # E[x1 | x_t, k] = mu_k + t s^2 (x - t mu_k) / var
    e1_k = MU[None, :, :] + (t * SIG * SIG / var) * (x[:, None, :] - t * MU[None, :, :])
    e1 = (r[:, :, None] * e1_k).sum(1)
    return (e1 - x) / max(1 - t, 1e-3)


def sample(x0, steps, dit_mask, dynamic=False):
    """dit_mask: 길이 steps의 bool 목록. True인 단계만 속도를 새로 계산."""
    x = x0.copy()
    dt = 1.0 / steps
    v_hist = []
    countdown = 0
    n_calls = 0
    for i in range(steps):
        t = i / steps
        if dynamic:
            if len(v_hist) < 2:
                run = True
            elif countdown > 1:
                countdown -= 1
                run = False
            elif countdown == 1:
                countdown = 0
                run = True
            else:
                vl = v_hist[-1].reshape(len(x), -1)
                vp = v_hist[-2].reshape(len(x), -1)
                sim = (vl * vp).sum(1) / (np.linalg.norm(vl, axis=1) * np.linalg.norm(vp, axis=1) + 1e-12)
                s = sim.mean()
                run = True
                for thr, cd in ((0.95, 4), (0.93, 2)):
                    if s > thr:
                        countdown = cd
                        run = False
                        break
        else:
            run = dit_mask[i]
        if run or not v_hist:
            v = velocity(x, t)
            n_calls += 1
        else:
            v = v_hist[-1]
        v_hist.append(v)
        x = x + dt * v
    return x, n_calls


def mask_from_count(steps, n_true, positions=None):
    m = [False] * steps
    if positions is None:
        positions = np.round(np.linspace(0, steps - 1, n_true)).astype(int)
    for p in positions:
        m[p] = True
    m[0] = True
    return m


# 코드의 mask (wan_flow_matching_action_tf.py 214–222행)
CODE_MASK = {
    5: [True, True, True, False, False, False, False, True, False, False, False, False, True, False, False, False],
    8: [True, True, True, False, False, False, True, False, False, False, True, False, False, True, True, True],
}

rng = np.random.default_rng(SEED)
x0 = rng.standard_normal((N_SAMPLES, 2))
ref, _ = sample(x0, 512, [True] * 512)


def assign_mode(x):
    d = ((x[:, None, :] - MU[None]) ** 2).sum(-1)
    return d.argmin(1)


configs = {
    "16단계 전부 (기준 스케줄)": ([True] * 16, False),
    "코드 mask 8회 (기본값)": (CODE_MASK[8], False),
    "코드 mask 5회 (최소 제공 mask)": (CODE_MASK[5], False),
    "논문 수치 4회 (균등 배치, 코드에 없음)": (mask_from_count(16, 4), False),
    "코드 동적 규칙 (코사인 0.95/0.93)": ([True] * 16, True),
}
res = {"assumptions": [
    "데이터: 2차원 4-가우시안 혼합(반지름 2, 표준편차 0.3), 정확한 속도장 사용 → DiT 근사 오차 0 (아래한계: 실제 캐시 오차는 이보다 큼)",
    "호출하지 않은 단계는 직전 속도를 재사용 (공개 코드 캐시 방식과 같은 가정)",
    "Euler, 균등 시간 간격. 코드의 sigma_shift = 5와 Flow UniPC는 넣지 않음",
    "참조: 같은 잡음에서 512단계 Euler. 샘플 4000개, 시드 0"],
    "results": {}}
ref_mode = assign_mode(ref)
for name, (mask, dyn) in configs.items():
    steps = len(mask)
    x, n_calls = sample(x0, steps, mask, dynamic=dyn)
    err = np.sqrt(((x - ref) ** 2).sum(1))
    res["results"][name] = {
        "dit_calls": int(n_calls),
        "rms_endpoint_error": round(float(np.sqrt((err ** 2).mean())), 4),
        "p90_endpoint_error": round(float(np.percentile(err, 90)), 4),
        "mode_agreement_with_ref_pct": round(100 * float((assign_mode(x) == ref_mode).mean()), 2),
        "mean_radius_error": round(float(np.abs(np.linalg.norm(x, axis=1) - np.linalg.norm(ref, axis=1)).mean()), 4),
    }
# 동적 규칙이 실제로 몇 회 호출되었는지 확인용: 평균 호출 수는 위 dit_calls에 있다.
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", OUT)
for k, v in res["results"].items():
    print(f"{k:45s} calls={v['dit_calls']:2d} rms={v['rms_endpoint_error']:.4f} p90={v['p90_endpoint_error']:.4f} mode%={v['mode_agreement_with_ref_pct']}")

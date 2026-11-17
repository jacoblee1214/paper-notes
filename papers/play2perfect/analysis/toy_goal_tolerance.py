"""Toy: 목표 허용오차 ε가 키포인트 6D 거리에서 허용하는 회전각 (Play2Perfect 식 (3), 부록 J).

질문: 논문의 ε = 1 cm, 5 cm, 10 cm가 같은 6D 목표에서 얼마나 다른 회전 정밀도를 요구하는가?
방법: s_rew = [0.14, 0.03, 0.03] m 직육면체의 8개 꼭짓점 대신 논문 식 (1)의 4개 키포인트를 쓴다.
      물체 중심에서 회전만 있을 때 d(R, I) = max_i ||R k_i - k_i||가 ε 이하인 최대 각 θ*를
      무작위 회전축 2000개에 대해 이분법으로 구한다. 축 방향 최솟값(worst case)이 허용 기준이다.
      코드(play/utils/obs_utils.py)는 성공 판정 허용오차를 ε × keypoint_scale(1.5)로 쓴다. 이 배율도 함께 본다.
출력: analysis/toy_goal_tolerance_results.json
실행: python3 -I analysis/toy_goal_tolerance.py  (numpy만, 1초 안팎)
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "toy_goal_tolerance_results.json")
rng = np.random.default_rng(0)

S_REW = np.array([0.14, 0.03, 0.03])  # 논문 J, 식 (3) 아래
KP = np.array([  # 식 (1): 네 키포인트
    [+S_REW[0] / 2, +S_REW[1] / 2, +S_REW[2] / 2],
    [+S_REW[0] / 2, -S_REW[1] / 2, -S_REW[2] / 2],
    [-S_REW[0] / 2, +S_REW[1] / 2, -S_REW[2] / 2],
    [-S_REW[0] / 2, -S_REW[1] / 2, +S_REW[2] / 2],
])
# 식 (1)은 열 벡터로 쓰였다. 위 행렬은 s/2 부호 조합 네 개이며, 원문 식의 부호 배치와 같은 집합이다.
# (원문은 x 성분 ±, y ±, z ± 가 대각 배치. 집합 비교는 아래 check에서 수행)


def rot(axis, theta):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K


def kp_dist(theta, axis):
    R = rot(axis, theta)
    return float(np.max(np.linalg.norm(KP @ R.T - KP, axis=1)))


def theta_star(eps, axis):
    """d(R_θ) = ε가 되는 최대 θ (θ ∈ [0, π])를 이분법으로 찾는다. 단조 증가 가정은 θ ≤ π에서 확인."""
    lo, hi = 0.0, math.pi
    if kp_dist(hi, axis) <= eps:
        return math.pi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if kp_dist(mid, axis) <= eps:
            lo = mid
        else:
            hi = mid
    return lo


def monotone_check(axis, n=200):
    th = np.linspace(0, math.pi, n)
    d = np.array([kp_dist(t, axis) for t in th])
    return bool(np.all(np.diff(d) >= -1e-12))


# 축 샘플: 좌표축 3개 + 무작위 2000개
axes = [np.array(a, float) for a in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]
axes += [v / np.linalg.norm(v) for v in rng.normal(size=(2000, 3))]

res = {"s_rew_m": S_REW.tolist(), "keypoint_scale_code": 1.5, "cases": {}}
res["monotone_all_axes_sampled"] = all(monotone_check(a) for a in axes[:50])

# 식 (1) 대각 배치 확인: 네 점은 s/2 부호의 대각 구조(부호 곱이 일정하지 않음)
signs = np.sign(KP)
res["keypoint_sign_pattern"] = signs.tolist()

labels = {
    "paper_eps_1cm": 0.01, "paper_eps_5cm": 0.05, "paper_eps_10cm": 0.10,
    "code_effective_1cm_x1.5": 0.015, "code_curriculum_start_2cm_x1.5": 0.03,
}
for name, eps in labels.items():
    th_all = np.array([theta_star(eps, a) for a in axes])
    res["cases"][name] = {
        "eps_m": eps,
        "theta_worst_deg": round(float(np.degrees(th_all.min())), 2),
        "theta_median_deg": round(float(np.degrees(np.median(th_all))), 2),
        "theta_best_deg": round(float(np.degrees(th_all.max())), 2),
        "axis_worst": [round(float(x), 3) for x in axes[int(th_all.argmin())]],
    }

# 해석용 해석해: 짧은 축(y, z) 회전에서 최악 반경 r = sqrt((s_x/2)^2 + (s_z/2)^2)
r_short = math.hypot(S_REW[0] / 2, S_REW[2] / 2)
analytic = {}
for name, eps in labels.items():
    analytic[name] = round(float(np.degrees(2 * math.asin(min(1.0, eps / (2 * r_short))))), 2)
res["analytic_short_axis_deg"] = analytic
res["note"] = (
    "최악 축은 가장 긴 키포인트 반경을 쓰는 방향이 아니라, 반경이 가장 작은 방향에서 나온다. "
    "θ* = 2 asin(ε/(2r)), r = 키포인트 반경. 수치와 해석해가 같은지 확인한다."
)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(json.dumps(res["cases"], ensure_ascii=False, indent=1))
print("analytic", analytic)
print("monotone", res["monotone_all_axes_sampled"])

"""토이 A: 이미지 목표의 목표 모호성(대칭 물체)과 연속 대조 목표의 최소 라벨 잡음.

가정(논문에 없는 설계 가정이며 논문 값에 맞추지 않음):
- 목표 방향은 항등 회전이라고 둔다(대칭 덕분에 일반성을 잃지 않음).
- 물체 외형이 대칭군 G에 대해 불변이면 q와 q·g(g ∈ G)는 같은 이미지를 낸다.
  * 정육면체: 회전 대칭군 O (24개 원소)
  * 몸체가 회전체인 병(예: 병 몸통): z축 회전 S^1 (연속 대칭). 무늬가 없다고 가정.
- 기하 성공: 측지 각 θ(q) < 0.4 rad (논문 §4의 임계값).
- 논문 연속 대조 목표: s* = exp(-α θ), α = 2.5 (부록 Table 11).
- 학생 이미지 인코더는 이미지 등가류마다 하나의 값을 낸다. MSE 최적해는 등가류 안 s*의 평균이므로,
  등가류 안 분산이 학습 목표에 남는 최소 잡음이다.

실행: python3 analysis/toy_symmetry.py  (numpy, 약 5초)
출력: analysis/toy_symmetry_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_symmetry_results.json"
TOL = 0.4          # 기하 성공 임계값 (rad)
ALPHA = 2.5        # 논문 Table 11의 α
N = 400_000
rng = np.random.default_rng(0)


def qmul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def angle(q):
    return 2 * np.arccos(np.clip(np.abs(q[..., 0]), 0.0, 1.0))


def haar(n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def octahedral_group():
    """정육면체 회전군 O (24원소)를 생성원으로부터 닫힘 계산으로 만든다."""
    gens = [np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]),                  # 90도 x축
            np.array([np.cos(np.pi / 3), *(np.sin(np.pi / 3) * np.ones(3) / np.sqrt(3))])]  # 120도 (1,1,1)
    elems = [np.array([1.0, 0, 0, 0])]
    frontier = list(elems)
    while frontier:
        new = []
        for a in frontier:
            for g in gens:
                b = qmul(a, g)
                if not any(abs(np.dot(b, e)) > 1 - 1e-9 for e in elems):
                    elems.append(b)
                    new.append(b)
        frontier = new
    assert len(elems) == 24, len(elems)
    return np.stack(elems)


def sstar(theta):
    return np.exp(-ALPHA * theta)


def cube_analysis(G):
    q = haar(N)
    th_geo = angle(q)                                              # 목표(항등)까지 거리
    # 같은 이미지를 내는 모든 자세 q·g 의 거리
    qg = qmul(q[:, None, :], G[None, :, :])                        # (N, 24, 4)
    th_all = angle(qg)                                             # (N, 24)
    th_min = th_all.min(axis=1)
    p_geo = float(np.mean(th_geo < TOL))
    p_img = float(np.mean(th_min < TOL))                           # "이미지상 목표와 같아 보이는" 자세 비율
    # 등가류 내 s* 분산 (MSE 최적화의 최소 잡음)
    s_all = sstar(th_all)
    class_var = s_all.var(axis=1)                                  # 등가류 안 분산
    s_tot_var = float(sstar(th_geo).var())
    # 실제 기하 성공과 이미지 성공이 어긋나는 비율: 이미지로는 성공인데 기하로는 실패
    mismatch = float(np.mean((th_min < TOL) & (th_geo >= TOL)))
    return {
        "P_geometric_success_random_goal": p_geo,
        "P_image_equivalent_success_random_goal": p_img,
        "ratio_image_over_geometric": p_img / p_geo,
        "P_image_success_but_geometric_fail": mismatch,
        "irreducible_MSE_of_sstar_target": float(class_var.mean()),
        "total_var_of_sstar_target": s_tot_var,
        "irreducible_fraction_of_target_variance": float(class_var.mean() / s_tot_var),
    }


def bottle_analysis(n=100_000, K=72):
    """회전체 병: 이미지는 자세의 z축 방향만 알려 준다고 본다. 같은 이미지 등가류 = {q·Rz(φ)}.
    등가류 안 s* 분산(잔여 잡음)을 전체 목표 분산과 비교한다. 등가류 K개를 격자로 근사한다."""
    q = haar(n)
    phis = np.linspace(0, 2 * np.pi, K, endpoint=False)
    rz = np.stack([np.cos(phis / 2), np.zeros(K), np.zeros(K), np.sin(phis / 2)], axis=1)  # z축 회전
    qz = qmul(q[:, None, :], rz[None, :, :])                       # (n, K, 4)
    th = angle(qz)
    s_all = sstar(th)
    within = s_all.var(axis=1).mean()                              # 등가류 안 분산 평균
    total = sstar(angle(q)).var()                                  # 전체 목표 분산
    # 축 정렬 조건부 기하 실패율: 목표 축과 정렬된 자세 q = Rz(φ)만 볼 때
    th_axis = np.abs(np.linspace(-np.pi, np.pi, 20001))
    p_geo_axis = float(np.mean(th_axis < TOL))
    return {
        "P_geometric_success_given_axis_aligned": p_geo_axis,
        "P_geometric_fail_given_axis_aligned": 1 - p_geo_axis,
        "irreducible_MSE_of_sstar_target_within_class": float(within),
        "total_var_of_sstar_target": float(total),
        "irreducible_fraction_of_target_variance": float(within / total),
        "class_size_K_grid": K,
    }


def main():
    G = octahedral_group()
    res = {
        "setup": {"tolerance_rad": TOL, "alpha": ALPHA, "n_samples": N, "seed": 0,
                  "cube_group_size": int(len(G))},
        "cube": cube_analysis(G),
        "bottle_of_revolution": bottle_analysis(),
        "note": "설계 가정 토이. 무늬와 조명이 없는 물체를 가정했다. 실제 GoalDex 물체의 대칭성과 텍스처는 논문에 기술되지 않았다.",
    }
    # 대칭 등가 자세가 목표에 가까운지 확인 (군 자체 점검): 항등 원소에서 거리 0
    res["check_identity_in_group"] = bool(np.any(angle(G) < 1e-9))
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

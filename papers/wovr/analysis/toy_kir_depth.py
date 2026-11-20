"""토이 T1: 롤아웃 깊이(유효 오차 깊이)와 가짜 성공(hallucinated success)의 관계 (numpy, 수 초).

모델 (논문 전체가 아니라 메커니즘만 본다):
- 실제 세계: 스칼라 진행도 x. 매 스텝 x += u_t + w_t (w_t는 아주 작은 잡음). 성공은 x_T >= 1.
- 세계모델: x̂ += u_t + b + e_t, e_t ~ N(0, s^2) (스텝마다 독립 오차, 편향 b).
  자기회귀 롤아웃이라 오차가 스텝 수만큼 쌓인다고 가정한다 (논문 §1의 두 원인 중 첫째).
- 기준선(초기 상태 롤아웃): 전체 H 스텝을 처음부터 상상한다.
- KIR: 임계 상태 근처 스텝 k에서 실제 상태를 알고 시작한다(첫 스텝 오차 ~ 0). 남은 H-k 스텝만 상상한다.
- 가짜 성공: 상상에서는 성공(x̂_T >= 1)이나 실제는 실패(x_T < 1).

주의: 이 토이는 오차가 독립 랜덤워크라는 가정에서 나오는 결과다. 논문의 실제 효과 크기를 검증하지 않는다.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_kir_depth_results.json")
H, K = 60, 40          # 전체 롤아웃 길이, KIR 시작 스텝 (남은 깊이 20)
N = 20000              # 에피소드 수
rng_seed = 0


def run(sigma, bias, kir, seed=rng_seed):
    rng = np.random.default_rng(seed)
    # 실제 진행: 목표 근처에서 실패/성공이 갈리도록 최종 진행도가 1 근처에 분포
    u = np.full((N, H), 1.0 / H)                       # 평균 진행 속도
    u += rng.normal(0, 0.004, size=(N, H))              # 실제 행동의 변동
    x = np.cumsum(u, axis=1)                           # 실제 궤적 x_t (t=1..H)
    x_real_T = x[:, -1] + rng.normal(0, 0.002, N)       # 아주 작은 실제 잡음
    real_success = x_real_T >= 1.0

    if not kir:
        start, x_hat = 0, np.zeros(N)                  # 처음부터 상상
    else:
        start = K
        x_hat = x[:, K - 1] + rng.normal(0, 0.002, N)  # 키프레임: 실제 상태에서 시작 (작은 잡음)
    e = rng.normal(0, sigma, size=(N, H - start)) + bias
    x_hat_T = x_hat + np.sum(u[:, start:], axis=1) + e.sum(axis=1)
    imag_success = x_hat_T >= 1.0
    hallucinated = imag_success & ~real_success
    return {
        "imagined_success_rate": float(imag_success.mean()),
        "real_success_rate": float(real_success.mean()),
        "hallucinated_rate_all_episodes": float(hallucinated.mean()),
        "false_positive_share_of_imagined_success": float(hallucinated.sum() / max(imag_success.sum(), 1)),
        "effective_depth": H - start,
        "accumulated_error_std": float(sigma * np.sqrt(H - start)),
    }


results = {"config": {"H_full": H, "K_kir": K, "N": N, "seed": rng_seed}, "sweep": {}}
for sigma in [0.002, 0.004, 0.008]:
    for bias in [0.0, 0.0002]:
        key = f"sigma={sigma}_bias={bias}"
        results["sweep"][key] = {"full_rollout": run(sigma, bias, False),
                                 "keyframe_rollout": run(sigma, bias, True)}

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
for k, v in results["sweep"].items():
    f, q = v["full_rollout"], v["keyframe_rollout"]
    print(f"{k:28s} full: depth={f['effective_depth']} halluc={f['hallucinated_rate_all_episodes']:.3f} "
          f"FP-share={f['false_positive_share_of_imagined_success']:.3f} | "
          f"kir: depth={q['effective_depth']} halluc={q['hallucinated_rate_all_episodes']:.3f} "
          f"FP-share={q['false_positive_share_of_imagined_success']:.3f}")

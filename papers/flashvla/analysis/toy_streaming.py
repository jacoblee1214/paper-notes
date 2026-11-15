"""FlashVLA 토이 (numpy, 단일 프로세스). 두 실험, 모두 논문 수치를 가정으로 쓴 산술 토이이며 논문 주장을 검증하지 않는다.

A. 지연 모델: 체인 10패스 디코딩 대 스트리밍 1패스(배치 N)의 스텝당 디코딩 시간. Fig 1b(미최적화)에서 캘리브레이션.
B. 비동기 경계 불연속: 추론 지연 d 스텝일 때 청크 경계에서의 행동 점프, 그리고 "진행 중 청크 조건화"가 오차를 줄이는 비율 rho (가정 파라미터).

실행: python3 analysis/toy_streaming.py  → analysis/toy_streaming_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_streaming_results.json"
RES = {}

# ---- A. 지연 모델 ----
# Fig 1b: 미최적화 π0.5 디코딩 97.4 ms = 10 패스 -> 패스당 9.74 ms. 스트리밍 1패스(N=4) 막대 4.9 ms.
p10 = 97.4 / 10.0
stream_bar = 4.9
VLM = 31.5           # Fig 1b VLM 부분 (대략 동일)
DEADLINE = 1000 / 30  # 30 Hz 주기 33.3 ms
RES["A_calibration"] = {"chunked_pass_ms": p10, "stream_bar_ms": stream_bar,
                        "required_per_pass_gain_if_stream_is_one_pass": round(p10 / stream_bar, 2)}

# 스트리밍 패스 비용 모델: c0 (패스 고정비, 커널 실행·가중치 읽기) + c1*N (청크 계산, 배치 비례)
# 미최적화 조건에서 c0 + c1*1 >= p10 이어야 한다고 가정 (N=1 패스는 체인 패스와 같다고 볼 때)
rows = []
for N in [1, 2, 4, 5, 6]:
    for c0_share in [0.0, 0.5, 0.9, 1.0]:
        c0 = p10 * c0_share
        c1 = (p10 - c0) / 1.0      # N=1 에서 c0 + c1 = p10
        pass_ms = c0 + c1 * N
        per_step_decode = pass_ms               # 스트리밍은 스텝당 패스 1회
        chunk_equiv = 10 * p10                  # 체인 10패스 = 청크 1개
        rows.append({"N": N, "c0_share": c0_share, "stream_pass_ms": round(pass_ms, 2),
                     "decode_speedup_per_chunk": round(chunk_equiv / pass_ms, 2) if pass_ms > 0 else None,
                     "work_ratio_flops_per_chunk": round(N / 10, 2)})
RES["A_grid"] = rows
RES["A_note"] = ("N=4 에서 패스가 체인 패스보다 빨라지려면 c0 비중이 커야 함. 그래도 Fig 1b 막대 4.9 ms(패스당 9.74 ms의 절반)에 맞추려면 "
                 "미최적화 조건에서 패스당 약 2배 단축이 필요하다. 이는 각주 1의 '커널 실행 직렬화 제거'와 맞고, '미최적화' 캡션과는 맞지 않는다.")

# 30 Hz 예산: VLM + 디코딩 <= 33.3 ms 여야 1스텝당 행동이 나온다 (단일 추론이 주기 안에 끝나야 할 때)
RES["A_deadline"] = {
    "deadline_ms": round(DEADLINE, 2),
    "decode_budget_if_vlm_31_5_ms": round(DEADLINE - VLM, 2),
    "chunked_pi05_fits": bool(VLM + 10 * p10 <= DEADLINE),
    "streaming_N4_fits": bool(VLM + stream_bar <= DEADLINE),
}

# ---- B. 비동기 경계 불연속 (가정 토이) ----
rng = np.random.default_rng(0)
T = 4000
t = np.arange(T)
ref = 0.6 * np.sin(2 * np.pi * t / 240.0)   # 부드러운 참조 궤적 (단위 임의)
vel = np.gradient(ref)


def run(d, E, rho, noise=0.02):
    """추론 지연 d 스텝. 청크 길이 E. 새 청크는 관측 시각 t-d 로 계산되어 실행 시각 t에 시작한다.
    naive: 첫 행동은 관측 시각 기준 참조값 -> 오차 = 속도 * d (잔여 편차 정도의 노이즈 포함).
    rho: 진행 중 청크를 조건화해 오차를 줄이는 비율 (가정, 0이면 naive, 1이면 완전 보정).
    반환: 청크 경계마다 이전 청크의 마지막 행동과 새 청크의 첫 행동 차이의 평균 절댓값."""
    jumps = []
    for k in range(d + E, T - E, E):
        stale_err = vel[k] * d
        new_first = ref[k] + (1 - rho) * stale_err + rng.normal(0, noise)
        old_last = ref[k - 1]
        jumps.append(abs(new_first - old_last))
    return float(np.mean(jumps))


RES["B_boundary_jump_mean_abs"] = {}
for E in [10, 20]:
    for d in [0, 1, 2, 4]:
        RES["B_boundary_jump_mean_abs"][f"E{E}_d{d}"] = {
            f"rho{rho}": round(run(d, E, rho), 4) for rho in [0.0, 0.5, 1.0]
        }
RES["B_note"] = ("토이 가정: 새 청크의 첫 행동 오차 = 속도 x d 의 (1-rho) 비율. rho는 '진행 중 청크 조건화'의 효과를 나타내는 "
                 "자유 파라미터이며 논문이 측정한 값이 아니다. 논문 Table 1/3의 성공률 차이는 이 토이로 설명하지 않는다.")

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(RES["A_calibration"], ensure_ascii=False), RES["A_deadline"])
print("B d=1,E=10:", RES["B_boundary_jump_mean_abs"]["E10_d1"])
print("wrote", OUT)

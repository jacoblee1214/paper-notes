"""토이 T1: "사람 루프는 좁은 힘 구간을 지키기엔 늦다"는 주장(서론, §1)이 어떤 조건에서 성립하는지 보는 1차원 접촉 모델.

실행: python3 analysis/toy_delay_window.py   (numpy, 1초 미만, 시드 고정)

논문은 접촉 강성, 닫힘 속도, 사람 반응 지연, 허용 힘 구간을 수치로 주지 않는다. 그래서 이 토이는 논문 수치를 재현하지 않고
"지연 × 속도 × 강성"이 구간 폭을 넘는 경계가 어디인지만 계산한다. 아래 값은 모두 가정이다.

모델: 그리퍼가 속도 v로 닫히다 접촉 위치부터 F = k × (침투 깊이)로 힘이 오른다. 조작자나 반사가 트리거 힘 F_trig에서
정지 명령을 내리면 지연 τ 동안 더 닫혀 힘이 F_trig + k·v·τ가 된다. 허용 구간은 [F_lo, F_hi]로 둔다.
  - 구간: 부록 D의 "눈에 띄는 변형 약 0.5 N, 1 N 미만 지속 하중에서 비가역 변형"에서 F_hi = 0.5 N, F_lo = 0.05 N(가정)
  - 닫힘 속도 v = 3.75 mm/s: 공개 반사 코드의 기본 close_step_um(150 µm) × 25 Hz (공개 코드 정적 값, 실측 아님)
  - 반사 지연 τ_reflex = 40 ms(25 Hz 한 주기) + 20 ms(처리, 가정) = 60 ms
  - 사람 지연 τ_h ~ 로그정규(중앙값 0.35 s, 로그 SD 0.3)(가정; 단순 반응 시간 0.2–0.3 s에 판단과 버튼 시간을 더한 값)
  - 트리거 F_trig = 0.15 N (사람과 반사 모두 같은 목표 힘)
강성 k는 모른다. 0.02–2 N/mm를 훑는다.
"""
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
HERE = Path(__file__).resolve().parent
F_LO, F_HI, F_TRIG = 0.05, 0.5, 0.15
V = 3.75  # mm/s
TAU_REFLEX = 0.060
N = 20000


def in_window(k, tau):
    f = F_TRIG + k * V * tau
    return (f >= F_LO) & (f <= F_HI)


def main():
    ks = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0])
    tau_h = rng.lognormal(mean=np.log(0.35), sigma=0.3, size=N)
    out = {"가정": {"F_lo": F_LO, "F_hi": F_HI, "F_trig": F_TRIG, "v_mm_s": V, "tau_reflex_s": TAU_REFLEX,
                  "tau_human": "로그정규 중앙값 0.35 s, 로그 SD 0.3"},
           "k_N_per_mm": ks.tolist(), "사람_구간내_비율": [], "반사_구간내_비율": [], "사람_평균_초과힘_N": [], "반사_초과힘_N": []}
    for k in ks:
        out["사람_구간내_비율"].append(round(float(in_window(k, tau_h).mean()), 3))
        out["반사_구간내_비율"].append(round(float(in_window(k, np.full(N, TAU_REFLEX)).mean()), 3))
        out["사람_평균_초과힘_N"].append(round(float((k * V * tau_h).mean()), 3))
        out["반사_초과힘_N"].append(round(k * V * TAU_REFLEX, 3))
    # 경계 강성: 사람 중앙값 지연에서 초과힘이 구간 폭 (F_hi - F_trig)에 이르는 k
    out["경계_강성_사람_중앙값(N/mm)"] = round((F_HI - F_TRIG) / (V * 0.35), 3)
    out["경계_강성_반사(N/mm)"] = round((F_HI - F_TRIG) / (V * TAU_REFLEX), 3)
    out["경계_강성_배율"] = round(out["경계_강성_반사(N/mm)"] / out["경계_강성_사람_중앙값(N/mm)"], 2)
    # 닫힘 속도에 대한 민감도: 속도를 5배로 올리면 경계가 5배 낮아진다
    out["경계_강성_v=1/5배_사람"] = round((F_HI - F_TRIG) / (V / 5 * 0.35), 3)
    out["경계_강성_v=5배_사람"] = round((F_HI - F_TRIG) / (V * 5 * 0.35), 3)
    (HERE / "toy_delay_window_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

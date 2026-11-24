"""Toy: 1축 준정적 접촉에서 가변 강성 + 가상 변위가 힘 오차를 얼마나 줄이는가 (numpy, 수 초).

논문 부록 B.2의 라벨 규칙(식 18, 값 500/200 N/m, 1/15 N)과 4.1절의 d = f / k_low 를 그대로 쓴다.
모델: 로봇 스프링 K가 목표를 추종하고, 목표가 표면보다 깊으면(e + d > 0) 환경 스프링 ke와 직렬로 눌린다.
  F = K * ke / (K + ke) * (e + d)   (e + d > 0),  아니면 0
  e : 목표 위치의 표면 높이 오차(+ 가 더 깊음), d : 가상 변위
세 구성을 비교한다.
  A  위치만, K = 500 고정 (논문 기준선 설정), 시연에서 누르는 힘을 만들려고 깊이 p 만큼 명령
  B  CompVLA 식: K = k_low(F*), d = F* / k_low(F*)
  C  고정 K = 500 + CompVLA의 d (표 3 고정 K 절제와 같은 구성)
1 자유도 이상화이며 논문 실험의 재현이 아니다. 접촉 충격, 감쇠, 마찰, 회전 축은 다루지 않는다.
실행: python3 analysis/toy_contact_1d.py -> analysis/toy_contact_1d_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_contact_1d_results.json")
KMAX, KMIN, FMIN, FMAX = 500.0, 200.0, 1.0, 15.0


def k_low(f):
    f = np.asarray(f, float)
    return np.where(f < FMIN, KMAX, np.where(f > FMAX, KMIN, KMAX - (KMAX - KMIN) * (f - FMIN) / (FMAX - FMIN)))


def force(K, ke, e, d):
    pen = np.maximum(e + d, 0.0)
    return K * ke / (K + ke) * pen


def main():
    R = {"label_map": {}, "cases": []}
    for f in (0.5, 1, 5, 10, 15, 30, 50):
        k = float(k_low(f))
        d = 0.0 if f < FMIN else f / k
        R["label_map"][str(f)] = {"k_low": round(k, 1), "d_mm": round(1000 * d, 1), "K_times_d": round(k * d, 2)}

    fstar = 10.0
    for ke in (2000.0, 5000.0, 20000.0):
        # A: 목표 힘을 만드는 명령 깊이 p (환경 강성을 안다고 가정한 이상적 시연)
        p = fstar * (KMAX + ke) / (KMAX * ke)
        kB = float(k_low(fstar))
        dB = fstar / kB
        # B에서는 d를 환경 강성을 모르고 K 와 f* 만으로 정한다 (실제 d = f*/k_low)
        row = {"ke": ke, "F_target": fstar, "A_penetration_cmd_mm": round(1000 * p, 2), "B_k_low": round(kB, 1),
               "B_d_mm": round(1000 * dB, 2)}
        for name, K, d in (("A", KMAX, p), ("B", kB, dB), ("C", KMAX, dB)):
            f0 = float(force(K, ke, 0.0, d))
            errs = {f"{1000 * e:+.0f}mm": round(float(force(K, ke, e, d)) - f0, 2) for e in (-0.005, 0.005, 0.01)}
            # 접촉이 끊기는 표면 높이 오차(목표가 표면 위로 d 이상 어긋남)
            row[name] = {"F_nominal": round(f0, 2), "dF_per_mm": round(float(K * ke / (K + ke)) / 1000, 3),
                         "force_change_for_error": errs, "contact_loss_if_error_above_mm": round(1000 * d, 1)}
        R["cases"].append(row)
    OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    print(json.dumps(R, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

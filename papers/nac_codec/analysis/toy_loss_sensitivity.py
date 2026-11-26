"""토이 T1: 재구성 손실이 "채널·부호·위치 어긋남"에 얼마나 민감한지 (합성 행동 청크, numpy/scipy, 수 초).

검증하려는 주장: 논문 4.2절과 부록 B.2는 "크기만 맞추는 스펙트로그램 손실은 위상을 맞추지 않아서, 평탄화한 행동에서
채널이 어긋나고 그리퍼 명령이 사라진다"고 설명한다. 여기서는 같은 종류의 오류를 합성 청크에 직접 넣어 손실별 값을 잰다.
이것은 논문의 학습을 재현한 것이 아니라 손실 함수만의 성질 검사이다.

설정(가정): 청크 32스텝 x 7차원, 앞 3차원은 저주파 사인 합(진폭 약 0.3), 다음 3차원은 진폭 0.05, 마지막은 그리퍼(±1, 청크 안에서 한 번 전환).
스펙트로그램은 공식 설정과 같은 n_fft 194, hop 64, 크기(power 1)이며 hann 창, 반사 패딩이다. 멜 필터는 쓰지 않았고
"로그 스펙트로그램"은 로그 멜 손실의 대용이다(실제 로그 멜 손실이 아니다).
실행: python3 analysis/toy_loss_sensitivity.py  → analysis/toy_loss_sensitivity_results.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import fft, signal

OUT = Path(__file__).resolve().parent / "toy_loss_sensitivity_results.json"
rng = np.random.default_rng(0)


def chunk():
    t = np.arange(32) / 32
    x = np.zeros((32, 7))
    for d in range(3):
        for k in (1, 2, 3):
            x[:, d] += 0.3 / k * np.sin(2 * np.pi * (k * t + rng.random())) * rng.random()
    for d in range(3, 6):
        x[:, d] = 0.05 * np.sin(2 * np.pi * (t + rng.random()))
    g = -1.0 if rng.random() < 0.5 else 1.0
    sw = rng.integers(8, 24)
    x[:, 6] = g
    x[sw:, 6] = -g
    return x


def spec(y, log=False):
    _, _, z = signal.stft(y, nperseg=194, noverlap=194 - 64, boundary="even", padded=True, window="hann")
    m = np.abs(z)
    return np.log(np.maximum(m, 1e-5)) if log else m


def losses(a, b):
    ya, yb = a.reshape(-1), b.reshape(-1)
    return {
        "mse": np.mean((ya - yb) ** 2),
        "l1": np.mean(np.abs(ya - yb)),
        "dct_l1": np.mean(np.abs(fft.dct(ya, norm="ortho") - fft.dct(yb, norm="ortho"))),
        "spec_l1": np.mean(np.abs(spec(ya) - spec(yb))),
        "logspec_l1": np.mean(np.abs(spec(ya, True) - spec(yb, True))),
    }


CASES = {
    "위치 오프셋 +0.3 (차원 0)": lambda x: x + np.eye(7)[0] * 0.3,
    "회전 오프셋 +0.3 (차원 3)": lambda x: x + np.eye(7)[3] * 0.3,
    "그리퍼 오프셋 +0.3 (차원 6)": lambda x: x + np.eye(7)[6] * 0.3,
    "그리퍼 부호 뒤집기 (차원 6)": lambda x: x * np.array([1, 1, 1, 1, 1, 1, -1]),
    "전체 부호 뒤집기": lambda x: -x,
    "평탄 신호를 1샘플 밀기 (채널 어긋남)": lambda x: np.roll(x.reshape(-1), 1).reshape(32, 7),
    "평탄 신호를 7샘플 밀기 (1스텝 밀기)": lambda x: np.roll(x.reshape(-1), 7).reshape(32, 7),
}


def main():
    xs = [chunk() for _ in range(200)]
    res = {"n_chunks": len(xs), "cases": {}}
    for name, fn in CASES.items():
        acc = {}
        for x in xs:
            for k, v in losses(x, fn(x)).items():
                acc.setdefault(k, []).append(v)
        res["cases"][name] = {k: float(np.mean(v)) for k, v in acc.items()}
    base = res["cases"]["위치 오프셋 +0.3 (차원 0)"]
    for name, d in res["cases"].items():
        d["relative_to_pos_offset"] = {k: d[k] / base[k] for k in ("mse", "l1", "dct_l1", "spec_l1", "logspec_l1")}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{'오류':36s} " + " ".join(f"{k:>10s}" for k in ("mse", "l1", "dct_l1", "spec_l1", "logspec_l1")))
    for name, d in res["cases"].items():
        print(f"{name:36s} " + " ".join(f"{d[k]:10.5f}" for k in ("mse", "l1", "dct_l1", "spec_l1", "logspec_l1")))
    print("위치 오프셋 대비 배율:")
    for name, d in res["cases"].items():
        print(f"{name:36s} " + " ".join(f"{d['relative_to_pos_offset'][k]:10.2f}" for k in ("mse", "l1", "dct_l1", "spec_l1", "logspec_l1")))


if __name__ == "__main__":
    main()

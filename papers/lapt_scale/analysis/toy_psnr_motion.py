"""작은 CPU 토이: PSNR 0.4 dB 차이가 움직임 정확도에 대해 말해 주는 것, 그리고 K개 코드의 변위 분해능.

논문의 중심 가설 둘을 합성 영상에서 계산한다(논문 수치를 예측하지 않는다).
  H1. 화소 지표는 큰 배경(과 센서 잡음)이 지배하므로 PSNR 차이가 작아도 움직임 정확도는 크게 다를 수 있다 (표 5, §4.4).
  H2. 이산 코드는 연속 변위를 거칠게 담는다 (§1, §4.6). 여기서는 코드 수 K가 2차원 변위를 얼마나 분해하는지만 본다.

실험 A: 64x64 영상, 무늬 배경 + 가우시안 점(반경 r)이 (dx, dy) 이동, 모든 프레임에 센서 잡음 sigma.
        예측기 4개의 다음 프레임 PSNR: 정답, 위치 오차 e 화소, 이동 없음(복사), 이동 방향 반대.
        점 면적 비율과 잡음 크기를 바꿔 "복사와 정답의 PSNR 차이"가 몇 dB인지 본다.
실험 B: [-3, 3]^2에서 균일한 변위를 K개 코드(k-means 중심)로 양자화했을 때의 R^2.

실행: python3 analysis/toy_psnr_motion.py   (numpy 필요, 약 5초) -> analysis/toy_psnr_motion_results.json
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
S = 64
RNG = np.random.default_rng(0)
YY, XX = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")


def blob(cy, cx, r):
    return np.exp(-(((YY - cy[:, None, None]) ** 2 + (XX - cx[:, None, None]) ** 2) / (2 * r ** 2)))


def psnr(a, b, peak=1.5):
    return 10 * math.log10(peak ** 2 / max(float(((a - b) ** 2).mean()), 1e-9))


def exp_a(r, sigma, n=1500, move=4.0):
    pos = RNG.uniform(16, S - 16, size=(n, 2))
    ang = RNG.uniform(0, 2 * np.pi, n)
    d = np.stack([np.sin(ang), np.cos(ang)], 1) * move
    bg = RNG.random((n, S, S)) * 0.3
    bg = (bg + np.roll(bg, 1, 1) + np.roll(bg, 1, 2) + np.roll(np.roll(bg, 1, 1), 1, 2)) / 4
    clean0 = bg + blob(pos[:, 0], pos[:, 1], r)
    clean1 = bg + blob(pos[:, 0] + d[:, 0], pos[:, 1] + d[:, 1], r)
    x1 = clean1 + RNG.normal(0, sigma, clean1.shape)  # 관측된 다음 프레임(잡음 포함)
    area = float((blob(np.array([32.0]), np.array([32.0]), r) > 0.05).mean())
    out = {"blob_area_pct": round(100 * area, 2), "sigma": sigma, "move_px": move}
    preds = {
        "perfect": clean1,
        "pos_err_1px": bg + blob(pos[:, 0] + d[:, 0] + 1.0, pos[:, 1] + d[:, 1], r),
        "pos_err_2px": bg + blob(pos[:, 0] + d[:, 0] + 2.0, pos[:, 1] + d[:, 1], r),
        "copy_no_motion": clean0,
        "wrong_direction": bg + blob(pos[:, 0] - d[:, 0], pos[:, 1] - d[:, 1], r),
    }
    for k, p in preds.items():
        out[f"psnr_{k}"] = round(psnr(p, x1), 3)
    out["gap_perfect_minus_copy_dB"] = round(out["psnr_perfect"] - out["psnr_copy_no_motion"], 3)
    out["gap_perfect_minus_wrong_dB"] = round(out["psnr_perfect"] - out["psnr_wrong_direction"], 3)
    return out


def kmeans(x, k, iters=30):
    c = x[RNG.choice(len(x), k, replace=False)]
    for _ in range(iters):
        idx = ((x[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            m = idx == j
            if m.any():
                c[j] = x[m].mean(0)
    idx = ((x[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
    return c, idx


def exp_b():
    d = RNG.uniform(-3, 3, size=(20000, 2))
    var = float(((d - d.mean(0)) ** 2).sum(1).mean())
    out = {}
    for k in (2, 4, 8, 16, 64, 256):
        c, idx = kmeans(d, k)
        mse = float(((d - c[idx]) ** 2).sum(1).mean())
        out[str(k)] = {"r2_of_quantized_displacement": round(1 - mse / var, 4), "rms_err_px": round(math.sqrt(mse), 3)}
    return out


def main():
    res = {"A": [], "B": exp_b()}
    for r, label in ((2.5, "점 큼"), (1.5, "점 중간"), (0.8, "점 작음")):
        for sigma in (0.005, 0.02, 0.05):
            row = exp_a(r, sigma)
            row["label"] = label
            res["A"].append(row)
            print(label, sigma, row["blob_area_pct"], row["gap_perfect_minus_copy_dB"], row["gap_perfect_minus_wrong_dB"])
    (HERE / "toy_psnr_motion_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(res["B"])


if __name__ == "__main__":
    main()

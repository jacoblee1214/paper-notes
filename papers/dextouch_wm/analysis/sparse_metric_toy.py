"""토이 B: 희소 접촉 촉각 맵에서 MSE/MAE/PSNR이 어떻게 오해를 부르는가.

320 taxel, 접촉 비율 p (기본 3%)인 정답 맵 y. 예측 후보:
  zero   : 항상 0 출력 (접촉을 전혀 예측하지 않음)
  blur   : 정답을 가우시안으로 번지게 한 출력 (접촉 위치는 맞지만 흐림, 최대값 낮음)
  shift  : 정답을 1 taxel 옮긴 출력 (접촉 강도는 맞지만 위치 틀림)
  noisy  : 정답 + 작은 잡음 (배경이 0이 아님)
지표: MSE, MAE, PSNR(peak=1), Contact-IoU(임계 0.1), Contact-F1, Contact-MAE(접촉 칸만 MAE).
결과: MSE 순위와 접촉 지표 순위가 뒤집히는 조건을 본다.

실행: python3 analysis/sparse_metric_toy.py -> analysis/sparse_metric_results.json (1스레드)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("sparse_metric_results.json")
H, W = 20, 16  # 320 taxel 근사 (20x16)
THR = 0.1


def gauss_blur(img, s=2.0):
    k = np.arange(-4, 5)
    g = np.exp(-(k ** 2) / (2 * s * s))
    g /= g.sum()
    out = img.copy()
    out = np.apply_along_axis(lambda r: np.convolve(r, g, mode="same"), 1, out)
    out = np.apply_along_axis(lambda c: np.convolve(c, g, mode="same"), 0, out)
    return out


def metrics(pred, y):
    mse = float(np.mean((pred - y) ** 2))
    mae = float(np.mean(np.abs(pred - y)))
    psnr = float(10 * np.log10(1.0 / max(mse, 1e-12)))
    pm, ym = pred > THR, y > THR
    inter = np.logical_and(pm, ym).sum()
    union = np.logical_or(pm, ym).sum()
    iou = float(inter / union) if union else 1.0
    prec = inter / pm.sum() if pm.sum() else 0.0
    rec = inter / ym.sum() if ym.sum() else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    cmae = float(np.mean(np.abs(pred - y)[ym])) if ym.any() else 0.0
    return {"MSE": round(mse, 5), "MAE": round(mae, 4), "PSNR": round(psnr, 2),
            "ContactIoU": round(iou, 3), "ContactF1": round(f1, 3), "ContactMAE": round(cmae, 4)}


def main():
    rng = np.random.default_rng(1)
    res = {}
    for p in (0.03, 0.10):
        acc = {}
        for trial in range(200):
            y = np.zeros((H, W))
            n = max(1, int(p * H * W))
            idx = rng.choice(H * W, size=n, replace=False)
            y.flat[idx] = rng.uniform(0.4, 1.0, size=n)
            y = gauss_blur(y, s=0.6)
            y = np.clip(y, 0, 1)
            cands = {
                "zero": np.zeros_like(y),
                "blur": gauss_blur(y, s=2.0) * 0.6,
                "shift": np.roll(y, 1, axis=1),
                "noisy": np.clip(y + 0.03 * rng.normal(size=y.shape), 0, 1),
            }
            for name, pred in cands.items():
                for k, v in metrics(pred, y).items():
                    acc.setdefault(name, {}).setdefault(k, []).append(v)
        res[f"contact_frac={p}"] = {name: {k: round(float(np.mean(v)), 4) for k, v in d.items()}
                                    for name, d in acc.items()}
    # MSE가 zero를 blur보다 좋게 보는지
    flags = {}
    for key, d in res.items():
        flags[key] = {"mse_prefers_zero_over_blur": d["zero"]["MSE"] < d["blur"]["MSE"],
                      "iou_prefers_blur_over_zero": d["blur"]["ContactIoU"] > d["zero"]["ContactIoU"]}
    out = {"settings": {"H": H, "W": W, "threshold": THR, "trials": 200,
                        "candidates": ["zero", "blur(s=2, x0.6)", "shift(1 taxel)", "noisy(0.03)"]},
           "results": res, "flags": flags}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(flags, indent=2))
    for key, d in res.items():
        print(key, {n: d[n]["MSE"] for n in d})


if __name__ == "__main__":
    main()

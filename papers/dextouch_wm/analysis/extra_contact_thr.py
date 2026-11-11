"""추가 검증 (DexTouch-WM): 접촉 임계값을 바꾸면 MSE와 접촉 지표의 순위 관계가 바뀌는가.

03 §5의 토이 B(sparse_metric_toy.py)를 임계값 0.05–0.30으로 확장한 가정 모델이다 (논문 실험 아님).
  - 320 taxel (20 x 16), 접촉 비율 3%와 10%, 접촉 값 U(0.4, 1.0), 정답은 sigma 0.6 가우시안으로 번짐.
  - 후보: zero(0 출력), blur(sigma 2로 번지고 0.6배), shift(1 taxel 이동), noisy(정답 + 잡음 0.03).
  - 임계값 thr: 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 (예측과 정답 모두 이진화).
  - 지표: MSE(임계값과 무관), Contact-IoU, Contact-F1.
  - 200회 반복, 시드 1 (토이 B와 같은 시드 규칙).
  - 확인할 것: (1) MSE에서 zero가 blur보다 나은지, (2) IoU와 F1에서 blur 대 zero, shift 대 zero 순서가
    임계값에 따라 뒤집히는지, (3) 임계값 0.1이 특별한지.

실행: python3 analysis/extra_contact_thr.py -> analysis/extra_contact_thr_results.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "extra_contact_thr_results.json"
H, W = 20, 16
THRS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
PRS = [0.03, 0.10]
TRIALS = 200
CANDS = ["zero", "blur", "shift", "noisy"]


def gauss_blur(img, s):
    k = np.arange(-4, 5)
    g = np.exp(-(k ** 2) / (2 * s * s))
    g /= g.sum()
    out = np.apply_along_axis(lambda r: np.convolve(r, g, mode="same"), 1, img)
    out = np.apply_along_axis(lambda c: np.convolve(c, g, mode="same"), 0, out)
    return out


def contact_scores(pred, y, thr):
    pm, ym = pred > thr, y > thr
    inter = np.logical_and(pm, ym).sum()
    union = np.logical_or(pm, ym).sum()
    iou = float(inter / union) if union else 1.0
    prec = inter / pm.sum() if pm.sum() else 0.0
    rec = inter / ym.sum() if ym.sum() else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return iou, f1


def main():
    rng = np.random.default_rng(1)
    res = {}
    for p in PRS:
        acc = {}
        for _ in range(TRIALS):
            y = np.zeros((H, W))
            n = max(1, int(p * H * W))
            idx = rng.choice(H * W, size=n, replace=False)
            y.flat[idx] = rng.uniform(0.4, 1.0, size=n)
            y = np.clip(gauss_blur(y, s=0.6), 0, 1)
            cands = {
                "zero": np.zeros_like(y),
                "blur": gauss_blur(y, s=2.0) * 0.6,
                "shift": np.roll(y, 1, axis=1),
                "noisy": np.clip(y + 0.03 * rng.normal(size=y.shape), 0, 1),
            }
            for name, pred in cands.items():
                mse = float(np.mean((pred - y) ** 2))
                acc.setdefault(name, {"MSE": []})["MSE"].append(mse)
                for t in THRS:
                    iou, f1 = contact_scores(pred, y, t)
                    d = acc[name].setdefault(f"thr={t:g}", {"IoU": [], "F1": []})
                    d["IoU"].append(iou)
                    d["F1"].append(f1)
        mean = {}
        for name in CANDS:
            mean[name] = {"MSE": round(float(np.mean(acc[name]["MSE"])), 5)}
            for t in THRS:
                key = f"thr={t:g}"
                mean[name][key] = {"IoU": round(float(np.mean(acc[name][key]["IoU"])), 3),
                                   "F1": round(float(np.mean(acc[name][key]["F1"])), 3)}
        # 순위 뒤집힘 요약
        flips = {}
        for t in THRS:
            key = f"thr={t:g}"
            flips[key] = {
                "MSE_prefers_zero_over_blur": mean["zero"]["MSE"] < mean["blur"]["MSE"],
                "IoU_order_blur_vs_zero": "blur>zero" if mean["blur"][key]["IoU"] > mean["zero"][key]["IoU"] else "blur<=zero",
                "F1_order_shift_vs_zero": "shift>zero" if mean["shift"][key]["F1"] > mean["zero"][key]["F1"] else "shift<=zero",
                "F1_ranking": sorted(CANDS, key=lambda c: -mean[c][key]["F1"]),
                "IoU_ranking": sorted(CANDS, key=lambda c: -mean[c][key]["IoU"]),
            }
        res[f"contact_frac={p:g}"] = {"means": mean, "flags_by_threshold": flips}
    out = {"settings": {"H": H, "W": W, "thresholds": THRS, "contact_fracs": PRS, "trials": TRIALS, "seed": 1,
                        "candidates": ["zero", "blur(s=2, x0.6)", "shift(1 taxel)", "noisy(0.03)"],
                        "note": "가정 모델. 논문의 접촉 임계값은 없음"},
           "results": res}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for key, d in res.items():
        print(key)
        for t in THRS:
            k = f"thr={t:g}"
            f = d["flags_by_threshold"][k]
            print(" ", k, "IoU", {c: d["means"][c][k]["IoU"] for c in CANDS},
                  "F1", {c: d["means"][c][k]["F1"] for c in CANDS},
                  "F1rank", f["F1_ranking"], "IoUrank", f["IoU_ranking"])
        print("  MSE", {c: d["means"][c]["MSE"] for c in CANDS})


if __name__ == "__main__":
    main()

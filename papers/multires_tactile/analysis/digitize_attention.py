"""부록 그림 9(과제별 센서 어텐션)의 곡선을 색으로 분리해 수치로 읽는다(그림 판독 값).

입력: paper/paper.pdf (arXiv:2606.06281 v2, 22쪽의 래스터 그림 5장). 받는 법은 README의 curl 명령.
출력: analysis/digitize_attention_results.json

방법: 각 그림에서 축의 눈금 위치를 검은 픽셀로 찾고(y 눈금 값은 아래 표에 손으로 적음), 파랑(Vision), 빨강(GelSight),
금색(Evetac) 선의 픽셀 중앙값 행을 시간 단계별로 읽어 값으로 바꾼다. 곡선이 서로 겹치는 칸은 한 색이 다른 색에 가려
몇 픽셀 어긋날 수 있다(판독 오차 약 ±0.01–0.02). 논문 값이 아니라 그림 판독 값이다.
"""
import json
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "analysis" / "digitize_attention_results.json"

# (과제, x축 끝 시간 단계, y 눈금 값(위에서 아래로), 범례가 있어 위쪽을 잘라야 하는 픽셀 수)
TASKS = [
    ("Gear Assembly", 119, [0.8, 0.6, 0.4, 0.2], 110),
    ("Board Wiping", 93, [0.75, 0.50, 0.25, 0.00], 0),
    ("Lamp Installation", 100, [0.75, 0.50, 0.25], 0),
    ("Key in Lock", 51, [0.6, 0.4, 0.2], 0),
    ("Lightbulb Connection", 136, [0.8, 0.6, 0.4, 0.2], 0),
]
COLORS = {"vision": (31, 119, 180), "gelsight": (214, 39, 40), "evetac": (218, 165, 32)}


def runs(idx):
    """연속한 인덱스 묶음의 중심."""
    if len(idx) == 0:
        return []
    groups, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            groups.append((start + a) / 2)
            start = b
    groups.append((start + idx[-1]) / 2)
    return groups


def load_images():
    doc = pymupdf.open(PDF)
    imgs = []
    for info in doc[21].get_images(full=True):
        xref = info[0]
        if info[1] != 0 or True:
            pix = pymupdf.Pixmap(doc, xref)
            if pix.n - pix.alpha >= 3 and pix.width > 1000:
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
                imgs.append(arr.astype(int))
    return imgs


def digitize(im, n_last, yvals, cut_top):
    sc = im.shape[1] / 3600                 # 그림 5(래스터 폭이 작음)는 창 크기를 같은 비율로 줄인다
    dark = im.sum(axis=2) < 200
    sx = int(dark.sum(axis=0).argmax())     # y축 선 열
    ax = int(dark.sum(axis=1).argmax())     # x축 선 행
    yt = runs(np.where(dark[:, sx - int(45 * sc):sx - int(8 * sc)].sum(axis=1) > 15 * sc)[0])
    xt = runs(np.where(dark[ax + int(8 * sc):ax + int(50 * sc), :].sum(axis=0) > 15 * sc)[0])
    assert len(yt) == len(yvals), (len(yt), yvals)
    # 행 → 값 (선형 맞춤)
    a, b = np.polyfit(yt, yvals, 1)
    x0, x1 = xt[0], xt[-1]
    series = {}
    for name, rgb in COLORS.items():
        m = np.abs(im - np.array(rgb)).sum(axis=2) < 60
        m[:int(cut_top * sc), :] = False
        m[ax - 2:, :] = False
        vals = []
        for t in range(n_last + 1):
            c = int(round(min(max(x0 + (x1 - x0) * t / n_last, x0 + 6 * sc), x1 - 6 * sc)))  # 축 선을 피해 안쪽으로
            w = max(int(3 * sc), 1)
            rows = np.where(m[:, c - w:c + w + 1].any(axis=1))[0]
            vals.append(float(a * np.median(rows) + b) if len(rows) else None)
        series[name] = vals
    for k, v in series.items():            # 다른 선에 가려진 칸은 양옆 값으로 선형 보간
        idx = [i for i, x in enumerate(v) if x is not None]
        series[k] = [x if x is not None else float(np.interp(i, idx, [v[j] for j in idx])) for i, x in enumerate(v)]
    return series, {"x_ticks": len(xt), "y_ticks": len(yt), "x_axis_row": ax}


def main():
    imgs = load_images()
    assert len(imgs) == 5, len(imgs)
    res = {}
    for im, (task, n_last, yvals, cut) in zip(imgs, TASKS):
        s, meta = digitize(im, n_last, yvals, cut)
        v, g, e = (np.array([x if x is not None else np.nan for x in s[k]]) for k in ("vision", "gelsight", "evetac"))
        tot = v + g + e
        res[task] = {
            "n_steps": n_last + 1,
            "meta": meta,
            "missing": {k: int(sum(x is None for x in s[k])) for k in s},
            "mean": {k: float(np.nanmean(np.array([x if x is not None else np.nan for x in s[k]]))) for k in s},
            "max_evetac": float(np.nanmax(e)),
            "sum_mean": float(np.nanmean(tot)),
            "sum_min": float(np.nanmin(tot)),
            "sum_max": float(np.nanmax(tot)),
            "dominant_counts": {k: int(np.sum(np.nanargmax(np.vstack([v, g, e]), axis=0) == i))
                                for i, k in enumerate(("vision", "gelsight", "evetac"))},
            "series": {k: [None if x is None else round(x, 4) for x in s[k]] for k in s},
        }
        print(task, {k: round(x, 3) for k, x in res[task]["mean"].items()},
              "합 평균", round(res[task]["sum_mean"], 3), "범위", round(res[task]["sum_min"], 3), round(res[task]["sum_max"], 3))
    # 토큰 수 비율: 비전 256, GelSight 192, Evetac 192, 합 640 (본문 N_cond = 644)
    res["token_share"] = {"vision": 256 / 644, "gelsight": 192 / 644, "evetac": 192 / 644, "other": 4 / 644}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

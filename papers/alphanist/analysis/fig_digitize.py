"""논문 그림 판독 (CPU). 실행: python3 analysis/fig_digitize.py

Fig. 5(a) heatmap 80칸을 색 막대로 역산하고, Fig. 6 학습 곡선 네 개를 선 색으로 따라가 25K·50K iteration 값을 읽는다.
원문 PDF를 300 dpi로 렌더링해서 쓰므로 paper/paper.pdf가 있어야 한다(README의 curl 명령).
"""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
DPI = 300
S = DPI / 110  # crop 좌표는 110 dpi 기준
TASKS = ["N4", "N8", "N12", "N16", "Gs", "Gm", "Gl", "R4", "R8", "R12", "R16",
         "P4", "P8", "P12", "P16", "USB", "WP", "BNC", "DS", "RJ"]
ROWS = ["shared_sparse", "separate_sparse", "shared_dense", "separate_dense"]
REPORTED = {"shared_sparse": 98.2, "separate_sparse": 81.4, "shared_dense": 17.8, "separate_dense": 78.3}


def page(n: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-r", str(DPI), "-f", str(n), "-l", str(n), "-png", "-singlefile",
                        str(PDF), f"{td}/p"], check=True, capture_output=True)
        return np.asarray(Image.open(f"{td}/p.png").convert("RGB")).astype(float)


def heatmap(img: np.ndarray) -> dict:
    a = img[int(70 * S):int(320 * S), int(80 * S):int(470 * S)]
    # 색 막대: 오른쪽 세로 띠. 위 100, 아래 0
    nonwhite = (a.sum(axis=2) < 700)
    cols = np.where(nonwhite[int(a.shape[0] * 0.5)])[0]
    # heatmap 본체와 색 막대를 가르는 흰 틈을 찾는다
    gaps = np.where(np.diff(cols) > 5)[0]
    hx0, hx1 = cols[0], cols[gaps[0]]
    bx0, bx1 = cols[gaps[0] + 1], cols[gaps[1]] if len(gaps) > 1 else cols[-1]
    bx = (bx0 + bx1) // 2
    rows_nonwhite = np.where(nonwhite[:, bx])[0]
    by0, by1 = rows_nonwhite[0], rows_nonwhite[-1]
    bar = a[by0:by1 + 1, bx]
    vals = np.linspace(100, 0, len(bar))
    ys = np.where(nonwhite[:, (hx0 + hx1) // 2])[0]
    hy0, hy1 = ys[0], ys[-1]
    w, h = (hx1 - hx0) / 20, (hy1 - hy0) / 4
    out = {}
    for r, name in enumerate(ROWS):
        row = []
        for c in range(20):
            cx, cy = int(hx0 + (c + 0.5) * w), int(hy0 + (r + 0.5) * h)
            patch = a[cy - 6:cy + 7, cx - 4:cx + 5].reshape(-1, 3).mean(axis=0)
            d = np.linalg.norm(bar - patch, axis=1)
            row.append(round(float(vals[d.argmin()]), 1))
        out[name] = dict(zip(TASKS, row))
    return out


# Fig. 6 선 색(범례에서 확인한 RGB)과 읽을 iteration(천 단위)
LINES = {"shared_sparse": (176, 140, 40), "separate_sparse": (190, 105, 75),
         "separate_dense": (85, 140, 120), "shared_dense": (60, 90, 130)}
READ_AT = [5, 10, 15, 20, 25, 30, 40, 50]


def curves(img: np.ndarray) -> dict:
    a = img[int(425 * S):int(690 * S), int(80 * S):int(470 * S)]
    gray = (np.abs(a - a.mean(axis=2, keepdims=True)).max(axis=2) < 8)
    dark = gray & (a.mean(axis=2) < 160)
    # 왼쪽 y축(어두운 세로선)과 아래 x축(어두운 가로선)
    xaxis_col = int(np.argmax(dark.sum(axis=0)))
    yaxis_row = int(np.argmax(dark.sum(axis=1)))
    # 가로 격자선(연회색, 25·50·75·100): x축보다 위에서 연회색이 긴 행
    light = gray & (a.mean(axis=2) > 200) & (a.mean(axis=2) < 245)
    cand = [r for r in range(a.shape[0]) if r < yaxis_row - 5 and light[r, xaxis_col + 5:].sum() > 0.5 * (a.shape[1] - xaxis_col)]
    groups = []
    for r in cand:
        if groups and r - groups[-1][-1] <= 2:
            groups[-1].append(r)
        else:
            groups.append([r])
    grid = sorted(np.mean(g) for g in groups)  # 위에서 아래: 100, 75, 50, 25
    y100, y0 = grid[0], yaxis_row
    # x=50K 눈금: 격자선이 끝나는 열
    x50 = int(np.where(light[int(round(grid[-1]))])[0].max())  # 25% 격자선 끝 = x 50K
    def val_y(row):
        return 100 * (y0 - row) / (y0 - y100)
    out = {"calib": {"x0_px": xaxis_col, "x50_px": x50, "y0_px": y0, "y100_px": float(y100)}}
    names = list(LINES)
    dist = np.stack([np.linalg.norm(a - np.array(LINES[n], float), axis=2) for n in names])
    nearest = dist.argmin(axis=0)  # 색이 비슷한 두 선(청록·파랑)을 가장 가까운 색으로 가른다
    for j, name in enumerate(names):
        mask = (nearest == j) & (dist[j] < 40)
        mask[:int(y100) - 3] = False  # 범례·제목 제외
        mask[y0 + 1:] = False
        vals = {}
        for k in READ_AT:
            cx = int(round(xaxis_col + (x50 - xaxis_col) * k / 50))
            for w in range(0, 12):
                cols = mask[:, max(cx - w, 0):cx + w + 1]
                rows = np.where(cols.any(axis=1))[0]
                if len(rows):
                    vals[f"{k}K"] = round(float(val_y(np.median(rows))), 1)
                    break
        out[name] = vals
    return out


def main():
    img = page(5)
    hm = heatmap(img)
    rowmeans = {k: round(float(np.mean(list(v.values()))), 1) for k, v in hm.items()}
    cv = curves(img)
    out = {"fig6_curves": cv, "fig5a_cells": hm, "fig5a_row_means": rowmeans, "reported_fig5b": REPORTED,
           "text_claim": "separate sparse: <=0.3% on M4 nut, 4 mm peg, rod insertion (Sec. V-A)",
           "separate_sparse_hard_cells": {k: hm["separate_sparse"][k] for k in ("N4", "P4", "R4")},
           "separate_dense_text": {"M4_45.0": hm["separate_dense"]["N4"], "Gm_93.6": hm["separate_dense"]["Gm"]}}
    path = ROOT / "analysis" / "fig_digitize_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(cv, ensure_ascii=False, indent=1))
    print(json.dumps({k: out[k] for k in ("fig5a_row_means", "reported_fig5b", "separate_sparse_hard_cells", "separate_dense_text")}, ensure_ascii=False, indent=1))
    for k, v in hm.items():
        print(k, v)


if __name__ == "__main__":
    main()

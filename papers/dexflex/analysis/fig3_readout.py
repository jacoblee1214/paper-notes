"""그림 3(벡터 그래프)의 점과 오차 막대 위치를 PDF에서 읽어 값으로 환산한다.

실행: python3 analysis/fig3_readout.py paper/paper.pdf analysis/fig3_readout.json
PyMuPDF(pymupdf)가 필요하다. 격자선(눈금) 위치로 선형 환산하므로 읽기 오차는 대략 눈금 간격의 1% 이내다.
"""
import json
import sys

import numpy as np
import pymupdf

pdf, out = sys.argv[1], sys.argv[2]
page = pymupdf.open(pdf)[5]  # 그림 3은 6쪽
dr = [x for x in page.get_drawings() if 60 < x["rect"].y0 and x["rect"].y1 < 290]

grid = sorted(x["rect"].x0 for x in dr
              if x["type"] == "s" and x["rect"].width == 0 and abs(x["rect"].height - 96.6) < 0.2)
panels, cur = [], [grid[0]]
for g in grid[1:]:
    if g - cur[-1] > 30:
        panels.append(cur)
        cur = [g]
    else:
        cur.append(g)
panels.append(cur)
ticks = [[5, 10, 15, 20, 25, 30], [.2, .4, .6, .8], [0, .2, .4, .6, .8]]  # 지속 시간(s), 진행도, 의도 오차
names = ["duration_s", "progress", "intent_error"]
fits = [np.polyfit(p, t, 1) for p, t in zip(panels, ticks)]
COL = {"raw": 0.5568, "filter": 0.3098, "dexgen": 0.8784, "notactile": 0.6627, "dexflex": 0.1804}


def method(c):
    return next((k for k, v in COL.items() if c and abs(c[0] - v) < 0.01), None)


def panel_of(x):
    return 0 if x < 290 else (1 if x < 395 else 2)


def val(x, pi):
    a, b = fits[pi]
    return float(a * x + b)


marks = []
for x in dr:
    r = x["rect"]
    if x["type"] == "fs" and 1.8 < r.width < 3.1 and r.height > 1.8 and x.get("color") and x["color"][0] < 0.2:
        m = method(x["fill"])
        if m and (r.y0 + r.y1) / 2 < 190:
            marks.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, m))
ys = sorted(set(round(y, 1) for _, y, _ in marks))
cl = []
for y in ys:
    if cl and y - cl[-1][-1] < 8:
        cl[-1].append(y)
    else:
        cl.append([y])
cen = [float(np.mean(c)) for c in cl]
assert len(cen) == 5, cen
conds = ["clean_expert", "additive_noise", "command_scaling", "delayed_command", "support_finger_removal"]

res = {n: {} for n in names}
for x, y, m in marks:
    pi = panel_of(x)
    i = int(np.argmin([abs(y - c) for c in cen]))
    res[names[pi]].setdefault(m, {})[conds[i]] = round(val(x, pi), 3)
err = {n: {} for n in names}
for x in dr:
    r = x["rect"]
    m = method(x.get("color")) if x["type"] == "s" else None
    if m and r.height == 0 and 3 < r.width < 30 and r.y0 < 190:
        pi = panel_of(r.x0)
        i = int(np.argmin([abs(r.y0 - c) for c in cen]))
        err[names[pi]].setdefault(m, {})[conds[i]] = [round(val(r.x0, pi), 3), round(val(r.x1, pi), 3)]
json.dump({"note": "PDF 벡터 좌표에서 환산한 값(읽기 오차 눈금의 약 1%). 오차 막대의 의미(SD, SE, CI)는 논문에 없다.",
           "values": res, "errorbars": err, "conditions": conds}, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", out)

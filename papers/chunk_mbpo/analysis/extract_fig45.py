"""논문 그림 4(b), 5(a), 5(b)의 값을 PDF 벡터에서 읽는다 (그림 판독 값, 원 데이터 아님).

PyMuPDF가 필요하다. 사용법: python3 analysis/extract_fig45.py
"""
import json, os
import numpy as np
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
doc = pymupdf.open(os.path.join(HERE, "..", "paper", "paper.pdf"))
page = doc[7]
words = page.get_text("words")
draw = page.get_drawings()


def rc(c):
    return tuple(round(x, 2) for x in c) if c else None


def yfit(pairs):
    v = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
    return np.polyfit(y, v, 1)


def labels(xmin, xmax, ymin, ymax):
    out = []
    for w in words:
        if xmin <= w[0] and w[2] <= xmax and ymin <= w[1] and w[3] <= ymax:
            try:
                out.append((float(w[4]), (w[1] + w[3]) / 2))
            except ValueError:
                pass
    return out


res = {}
# ---- 그림 4(b): 기울기 노름, 가로축 2,3,4 ----
a, b = yfit(labels(370, 388, 150, 215))
bars = {"MAAC": rc((0.84, 0.37, 0.0)), "CMPO": rc((0.0, 0.45, 0.7))}
r4 = {}
for name, col in bars.items():
    for d in draw:
        r = d["rect"]
        if rc(d["color"]) == col and d["type"] == "s" and len(d["items"]) == 1 and r.width < 0.01 and 395 < r.x0 < 495 and 120 < r.y0 < 215:
            h = round((r.x0 - 400.6) / 42.4) + 2 if name == "MAAC" else round((r.x0 - 403.6) / 42.4) + 2
            r4.setdefault(name, {})[h] = dict(max=a * r.y0 + b, min=a * r.y1 + b)
    for d in draw:  # 평균 마커(흰 채움 + 색 외곽)
        r = d["rect"]
        if d["type"] == "fs" and rc(d["fill"]) == col and rc(d["color"]) == (1.0, 1.0, 1.0) and 395 < r.x0 < 495 and 185 < r.y0 < 205 and r.width < 3:
            xc = (r.x0 + r.x1) / 2; yc = (r.y0 + r.y1) / 2
            h = int(round((xc - (401 if name == "MAAC" else 403.6)) / 42.4)) + 2
            r4.setdefault(name, {}).setdefault(h, {})["mean"] = a * yc + b
res["fig4b_grad_norm_by_horizon"] = r4

# ---- 그림 5(b): 잠재 MSE (x1e-3) ----
a, b = yfit(labels(370, 382, 435, 510))
a *= 1.0  # 눈금은 1e-3 단위
cols = {rc((0.0, 0.0, 0.0)): "Recurrent", rc((0.28, 0.17, 0.48)): "Chunk h=1", rc((0.2, 0.38, 0.55)): "Chunk h=2",
        rc((0.13, 0.57, 0.55)): "Chunk h=3", rc((0.25, 0.74, 0.45)): "Chunk h=4", rc((0.69, 0.87, 0.18)): "Chunk h=5"}
r5 = {}
for d in draw:
    r = d["rect"]
    if not (380 < r.x0 < 505 and 430 < r.y0 < 510):
        continue
    c = rc(d["color"]) if d["color"] else None
    if d["type"] == "s" and c in cols and len(d["items"]) >= 1 and d["items"][0][0] == "l" and r.width > 5:
        pts = [(d["items"][0][1].x, d["items"][0][1].y)] + [(it[2].x, it[2].y) for it in d["items"] if it[0] == "l"]
        # 선 경로: 점 사이 모두 직선, 각 꼭짓점이 예측 스텝 1..5
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        step = np.round((xs - 387.5) / 27.4).astype(int) + 1
        r5[cols[c]] = {int(s): float(a * y + b) for s, y in zip(step, ys)}
res["fig5b_latent_mse_x1e-3"] = r5

# ---- 그림 5(a): 호라이즌별 평균 곡선 (Walk로 추정) ----
a, b = yfit(labels(115, 130, 395, 520))
# 범례 선은 벡터에서 잡히지 않아, 그림의 색을 눈으로 대조해 H에 대응시켰다 (보라 H=1, 초록 H=2, 분홍 H=3, 노랑 H=4, 주황 H=5)
names = {"H=1": rc((0.42, 0.24, 0.6)), "H=2": rc((0.0, 0.62, 0.45)), "H=3": rc((0.8, 0.47, 0.65)),
         "H=4": rc((0.9, 0.62, 0.0)), "H=5": rc((0.84, 0.37, 0.0))}
r5a = {}
for d in draw:
    r = d["rect"]
    if d["type"] == "s" and len(d["items"]) > 20 and 120 < r.x0 < 300 and 420 < r.y0 < 520:
        c = rc(d["color"])
        for nm, col in names.items():
            if col == c:
                pts = [(d["items"][0][1].x, d["items"][0][1].y)] + [(it[2].x, it[2].y) for it in d["items"] if it[0] == "l"]
                ys = np.array([a * p[1] + b for p in pts])
                r5a[nm] = dict(n=len(ys), final=float(ys[-1]), last5_mean=float(ys[-5:].mean()), max=float(ys.max()))
res["fig5a_walk_horizon_reward"] = r5a
json.dump(res, open(os.path.join(HERE, "fig45_values.json"), "w"), indent=1)
print(json.dumps(res, indent=1)[:3500])

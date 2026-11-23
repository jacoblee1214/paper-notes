"""논문 PDF의 벡터 그래프(그림 2, 그림 7, 그림 4, 그림 5)에서 곡선 값을 읽어 JSON으로 저장한다.

PyMuPDF가 필요하다. 사용법: python3 analysis/extract_curves.py
읽은 값은 "그림 판독 값"이며 원 데이터가 아니다(선 경로를 축 눈금에 맞춰 환산).
"""
import json, os, re
import numpy as np
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "..", "paper", "paper.pdf")

METHODS = {  # 범례 순서와 색 (그림 7 범례: Ours, MAAC, TD-MPC2, DreamerV3, SAC, PPO)
    (0.0, 0.45, 0.7): "CMPO",
    (0.89, 0.09, 0.09): "MAAC",
    (0.9, 0.62, 0.0): "TD-MPC2",
    (0.0, 0.62, 0.45): "DreamerV3",
    (0.84, 0.37, 0.0): "SAC",
    (0.8, 0.47, 0.65): "PPO",
}
TITLES = ["Average", "Balance Hard", "Balance Simple", "Basketball",
          "Bookshelf Hard", "Bookshelf Simple", "Cube", "Door",
          "Hurdle", "Maze", "Pole", "Run",
          "Sit Hard", "Sit Simple", "Slide", "Spoon",
          "Stair", "Stand", "Walk", "Window"]


def rc(c):
    return tuple(round(x, 2) for x in c)


def stroke_points(path):
    pts = []
    for it in path["items"]:
        if it[0] == "l":
            if not pts:
                pts.append((it[1].x, it[1].y))
            pts.append((it[2].x, it[2].y))
    return pts


def panels(page):
    words = page.get_text("words")
    zeros = sorted([w for w in words if w[4] == "0.0"], key=lambda w: (round(w[1]), w[0]))
    ones = sorted([w for w in words if w[4] == "1.0"], key=lambda w: (round(w[1]), w[0]))
    out = []
    for z, o in zip(zeros, ones):
        out.append(dict(x0=(z[0] + z[2]) / 2, x1=(o[0] + o[2]) / 2, ybase=z[1]))
    for p in out:  # 위 행 패널과의 간격에 맞춰 검색 높이 제한
        above = [q["ybase"] for q in out if abs(q["x0"] - p["x0"]) < 5 and q["ybase"] < p["ybase"] - 5]
        p["h"] = min(90, p["ybase"] - max(above) - 12) if above else 90
    return out, words


def calibrate_y(panel, words, hlines):
    """패널 왼쪽 눈금 숫자와 가로 격자선을 짝지어 y 픽셀 -> 값 선형식을 만든다."""
    xl, xr = panel["x0"] - 20, panel["x0"] - 1
    ytop = panel["ybase"] - panel["h"]
    labs = []
    for w in words:
        if xl <= w[0] and w[2] <= panel["x0"] + 1 and ytop <= (w[1] + w[3]) / 2 <= panel["ybase"] + 2:
            if re.fullmatch(r"-?\d+(\.\d+)?", w[4]):
                labs.append((float(w[4]), (w[1] + w[3]) / 2))
    gl = [y for (y, xa, xb) in hlines if xa >= panel["x0"] - 2 and xb <= panel["x1"] + 3 and ytop <= y <= panel["ybase"] + 2]
    pairs = []
    for v, yc in labs:
        if gl:
            yg = min(gl, key=lambda y: abs(y - yc))
            if abs(yg - yc) < 3:
                pairs.append((v, yg))
    if len(pairs) < 2:
        pairs = labs
    v = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
    a, b = np.polyfit(y, v, 1)
    return a, b, len(pairs)


def extract(page):
    pans, words = panels(page)
    dr = page.get_drawings()
    hl = []
    for d in dr:
        if d.get("color") and rc(d["color"]) == (0.69, 0.69, 0.69):
            for it in d["items"]:
                if it[0] == "l" and abs(it[1].y - it[2].y) < 0.01 and abs(it[1].x - it[2].x) > 20:
                    hl.append((it[1].y, min(it[1].x, it[2].x), max(it[1].x, it[2].x)))
    res = {}
    for pi, pn in enumerate(pans):
        a, b, n = calibrate_y(pn, words, hl)
        pn["a"], pn["b"], pn["ncal"] = a, b, n
    for d in dr:
        if d.get("color") is None or d["type"] != "s" or len(d["items"]) <= 5:
            continue
        m = METHODS.get(rc(d["color"]))
        if m is None:
            continue
        pts = stroke_points(d)
        if not pts:
            continue
        px = np.mean([p[0] for p in pts]); py = np.mean([p[1] for p in pts])
        for pi, pn in enumerate(pans):
            if pn["x0"] - 3 <= pts[0][0] <= pn["x1"] + 3 and pn["ybase"] - pn["h"] - 5 <= py <= pn["ybase"] + 2:
                xs = [(p[0] - pn["x0"]) / (pn["x1"] - pn["x0"]) * 1.0 for p in pts]
                ys = [pn["a"] * p[1] + pn["b"] for p in pts]
                res.setdefault(pi, {})[m] = dict(x_M=xs, y=ys)
    return pans, res


def extract_bands(page, pans):
    """신뢰구간 음영(채운 다각형)의 마지막 x에서 위·아래 값을 읽는다."""
    out = {}
    for d in page.get_drawings():
        if d.get("fill") is None or d["type"] != "f" or len(d["items"]) <= 5:
            continue
        m = METHODS.get(rc(d["fill"]))
        if m is None:
            continue
        pts = []
        for it in d["items"]:
            if it[0] == "l":
                pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
        if not pts:
            continue
        py = np.mean([p[1] for p in pts])
        for pi, pn in enumerate(pans):
            if pn["x0"] - 3 <= pts[0][0] <= pn["x1"] + 3 and pn["ybase"] - pn["h"] - 5 <= py <= pn["ybase"] + 2:
                xmax = max(p[0] for p in pts)
                ys = [pn["a"] * p[1] + pn["b"] for p in pts if abs(p[0] - xmax) < 0.05]
                out.setdefault(pi, {})[m] = dict(lo=float(min(ys)), hi=float(max(ys)))
    return out


FIG2_TITLES = ["Average", "Balance Hard", "Hurdle", "Pole", "Maze", "Run", "Sit Hard", "Slide", "Stair", "Walk"]


def fig7(doc):
    pans, res = extract(doc[15])
    out = {}
    for pi, t in enumerate(TITLES):
        out[t] = res.get(pi, {})
    return pans, out


if __name__ == "__main__":
    doc = pymupdf.open(PDF)
    pans, f7 = fig7(doc)
    bands = extract_bands(doc[15], pans)
    pans2, res2 = extract(doc[5])
    f2 = {t: res2.get(i, {}) for i, t in enumerate(FIG2_TITLES)}
    summary = {}
    for t, ms in f7.items():
        summary[t] = {}
        for m, c in ms.items():
            y = np.array(c["y"]); x = np.array(c["x_M"])
            bd = bands.get(TITLES.index(t), {}).get(m, {})
            summary[t][m] = dict(band_lo_end=bd.get("lo"), band_hi_end=bd.get("hi"), n_pts=len(y), x_end=float(x[-1]), y_final=float(y[-1]),
                                 y_max=float(y.max()), x_at_max=float(x[y.argmax()]),
                                 y_last5_mean=float(y[-5:].mean()))
    json.dump(dict(panels=[{k: float(v) for k, v in p.items()} for p in pans], summary=summary, curves=f7, curves_fig2=f2),
              open(os.path.join(HERE, "curves_fig7.json"), "w"))
    json.dump(summary, open(os.path.join(HERE, "curves_fig7_summary.json"), "w"), indent=1)
    for t in TITLES:
        print(t, {m: (round(v["y_final"], 1), round(v["y_max"], 1)) for m, v in summary[t].items()})

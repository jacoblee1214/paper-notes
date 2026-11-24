"""논문 PDF(paper/paper.pdf)의 벡터 막대 그림(그림 3, 6, 7, 8, 9, 10)에서 막대 높이와 오차 막대 끝을 읽는다.

막대 = 채움색이 있는 사각형(검은 채움 그림자는 제외), 오차 막대 = 막대 중심 x의 세로선 한 개와 위아래 가로 캡 두 개.
세로 눈금은 회색 가로 격자선(0%와 100%)으로 보정한다. 읽은 값은 '그림 판독 값'이며 원자료가 아니다.
실행: python3 -I analysis/extract_figures.py   → analysis/fig_readings.json (수 초, PyMuPDF 필요)
"""
import json
from collections import defaultdict
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = Path(__file__).with_name("fig_readings.json")

# 그림별 페이지(0부터)와 y 범위(pt). 격자선은 이 범위 안에서 찾는다.
PANELS = {
    "fig6": (5, 330, 425), "fig7_top": (6, 40, 115), "fig7_bottom": (6, 125, 200),
    "fig8": (7, 60, 165), "fig9": (7, 295, 380), "fig10": (8, 60, 165),
}


def is_gray(c):
    return c is not None and abs(c[0] - c[1]) < 0.01 and 0.45 < c[0] < 0.55


def read(page, y0, y1):
    dr = [d for d in page.get_drawings() if y0 < d["rect"].y0 < y1 and d["rect"].y1 < y1 + 2]
    grid = defaultdict(list)
    for d in dr:
        r = d["rect"]
        if is_gray(d.get("color")) and abs(r.y0 - r.y1) < 1e-3 and d.get("fill") is None:
            grid[(round(r.x0), round(r.x1))].append(r.y0)
    bars, vlines, caps = [], [], []
    for d in dr:
        r = d["rect"]
        f, c = d.get("fill"), d.get("color")
        kind = d["items"][0][0]
        if kind == "re" and f and f != (0.0, 0.0, 0.0) and c is not None and r.width > 3 and r.height > 0.3:
            bars.append(d)
        elif kind == "l" and abs(r.x0 - r.x1) < 1e-3 and c == (0.0, 0.0, 0.0):
            vlines.append(r)
        elif kind == "l" and abs(r.y0 - r.y1) < 1e-3 and c == (0.0, 0.0, 0.0) and f is not None:
            caps.append(r)
    return grid, bars, vlines


def panels_from(grid):
    """x 구간이 겹치는 격자선 묶음 = 한 패널. 패널별 (y0%, y100%)."""
    rows = sorted(grid.items())
    out = []
    for (xa, xb), ys in rows:
        ys = sorted(set(round(y, 2) for y in ys))
        out.append({"x": (xa, xb), "y0": max(ys), "y100": min(ys), "ny": len(ys)})
    return out


def fig3(doc):
    """그림 3: 지연 곡선 점(원 마커) 중심 y를 0.0 s와 0.5 s 눈금으로 보정. 초 단위."""
    pg = doc[3]
    ticks = {}
    for w in pg.get_text("words"):
        if 245 < w[1] < 380 and w[0] < 90 and w[4] in ("0.0", "0.5"):
            ticks[w[4]] = (w[1] + w[3]) / 2
    # 눈금선(검은 짧은 가로선)이 글자 높이와 같지 않을 수 있어 선 y를 직접 쓴다
    ys = sorted(d["rect"].y0 for d in pg.get_drawings()
                if 245 < d["rect"].y0 < 380 and abs(d["rect"].x1 - d["rect"].x0 - 3.0) < 0.01 and abs(d["rect"].y0 - d["rect"].y1) < 1e-3
                and d["rect"].x0 > 84 and d["rect"].x0 < 86)
    y_hi, y_lo = ys[-1], ys[-2]  # 0.0 s, 0.5 s
    per_s = (y_hi - y_lo) / 0.5
    xs = [d["rect"].x0 + 1.75 for d in pg.get_drawings()
          if 245 < d["rect"].y0 < 380 and abs(d["rect"].x1 - d["rect"].x0 - 3.0) < 0.01 and abs(d["rect"].y0 - d["rect"].y1) < 1e-3 and d["rect"].y1 > 376 and d["rect"].y0 < 377]
    out = {"without_encoder_s": {}, "with_encoder_s": {}}
    marks = [(d["fill"], d["rect"]) for d in pg.get_drawings() if d["items"][0][0] == "c" and 245 < d["rect"].y0 < 380 and d["rect"].width < 4
             and d["rect"].x0 > 90 and d["rect"].y0 > 280 - 30 and not (d["rect"].x0 < 100 and d["rect"].y0 < 275)]
    return {"y0_pt": y_hi, "pt_per_s": per_s, "n_marks": len(marks),
            "marks": [{"x": round((r.x0 + r.x1) / 2, 2), "series": "without" if f[0] == 0.0 else "with",
                       "sec": round((y_hi - (r.y0 + r.y1) / 2) / per_s, 3)} for f, r in marks]}


def main():
    doc = pymupdf.open(PDF)
    res = {"fig3": fig3(doc)}
    for name, (pn, y0, y1) in PANELS.items():
        grid, bars, vlines = read(doc[pn], y0, y1)
        pans = panels_from(grid)
        items = []
        for b in bars:
            r = b["rect"]
            cx = (r.x0 + r.x1) / 2
            # 이 막대가 속한 패널: x 범위가 가장 가까운 격자 묶음(막대 아래 끝이 0% 선과 같은 y)
            cand = [p for p in pans if p["x"][0] - 6 <= cx <= p["x"][1] + 6]
            if not cand:
                continue
            p = min(cand, key=lambda q: abs(q["y0"] - r.y1))
            scale = (p["y0"] - p["y100"]) / 100.0
            val = (p["y0"] - r.y0) / scale
            err = [v for v in vlines if abs((v.x0 + v.x1) / 2 - cx) < 1.0 and abs(v.y1 - r.y0) < 40]
            e = None
            if err:
                v = min(err, key=lambda v: abs(v.y0 - r.y0) + abs(v.y1 - r.y0))
                e = [round((p["y0"] - v.y1) / scale, 2), round((p["y0"] - v.y0) / scale, 2)]
            items.append({"x": round(cx, 1), "panel_x": p["x"], "fill": [round(c, 3) for c in b["fill"]],
                          "value": round(val, 2), "err_lo_hi": e})
        items = [i for i in items if i["value"] <= 100.5 and i["fill"] != [0.941, 0.937, 0.91]]
        items.sort(key=lambda i: (i["panel_x"], i["x"]))
        res[name] = {"n_panels": len(pans), "grid": pans, "bars": items}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("fig3 marks", res["fig3"]["n_marks"])
    for k, v in res.items():
        if k != "fig3":
            print(k, "panels", v["n_panels"], "bars", len(v["bars"]))


if __name__ == "__main__":
    main()

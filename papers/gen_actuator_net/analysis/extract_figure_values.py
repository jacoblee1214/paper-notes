"""논문 PDF(v4)의 벡터 그림에서 막대 높이·오차막대 끝을 읽어 값으로 환산한다.

그림 판독 값이므로 각 값은 약 ±0.1 화소(그림 눈금 기준 0.1~0.3% 내외) 오차를 가진다.
눈금 라벨의 위치로 y축을 선형 보정한다(라벨 중심 y좌표와 라벨 값을 최소제곱으로 맞춤).

실행: python3 extract_figure_values.py [paper.pdf]   (PyMuPDF 필요, 결과는 figure_values.json)
"""
import json
import sys
from pathlib import Path

import numpy as np

try:
    import pymupdf
except ImportError:  # 구버전 이름
    import fitz as pymupdf

HERE = Path(__file__).resolve().parent
PDF = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "paper" / "paper.pdf"


def fit_axis(page, label_box, scale=1.0):
    """label_box=(x0,x1,y0,y1) 안의 숫자 라벨로 y(화소)->값 선형 보정 (a*y+b)."""
    xs, vs = [], []
    for w in page.get_text("words"):
        x0, y0, x1, y1, t = w[:5]
        if label_box[0] <= x0 and x1 <= label_box[1] and label_box[2] <= (y0 + y1) / 2 <= label_box[3]:
            try:
                v = float(t.replace("−", "-"))
            except ValueError:
                continue
            xs.append((y0 + y1) / 2)
            vs.append(v * scale)
    a, b = np.polyfit(xs, vs, 1)
    return a, b, len(xs)


def bars_in(page, frame, min_w=5):
    out = []
    for d in page.get_drawings():
        r = d["rect"]
        if d["type"] == "fs" and d.get("fill") and tuple(d["fill"]) != (1, 1, 1) and r.width > min_w and r.height > 0.3:
            if frame[0] <= r.x0 and r.x1 <= frame[2] and frame[1] - 1 <= r.y0 and r.y1 <= frame[3] + 1:
                out.append(r)
    return sorted(out, key=lambda r: r.x0)


def caps_for(page, rect, frame):
    """막대 중심 x에 맞는 오차막대 가로 끝선 두 개(y) 반환."""
    cx = (rect.x0 + rect.x1) / 2
    ys = []
    for d in page.get_drawings():
        r = d["rect"]
        if d["type"] == "s" and r.height < 0.01 and 3 < r.width < 12 and abs((r.x0 + r.x1) / 2 - cx) < 0.6:
            if frame[1] - 1 <= r.y0 <= frame[3] + 1 and d.get("color") and abs(d["color"][0] - 0.41) < 0.02:
                ys.append(r.y0)
    return sorted(set(round(y, 2) for y in ys))


def panel(page, frame, label_box, scale=1.0, exclude=None):
    a, b, n = fit_axis(page, label_box, scale)
    res = []
    for r in bars_in(page, frame):
        if exclude and exclude[0] <= r.x0 and r.x1 <= exclude[2] and exclude[1] <= r.y0 and r.y1 <= exclude[3]:
            continue
        off = a * r.y1 + b  # 막대 바닥이 0이 되도록 눈금 라벨 중심과 격자선 사이의 오프셋 제거
        val = a * r.y0 + b - off
        caps = caps_for(page, r, frame)
        ci = [a * caps[-1] + b - off, a * caps[0] + b - off] if len(caps) >= 2 else None  # 아래 끝=작은 값
        res.append({"value": val, "ci": ci})
    return res


def main():
    doc = pymupdf.open(str(PDF))
    out = {}

    # Fig.5 (p8): 라벨 값(90 97 ...)이 막대 위에 적혀 있어 y축 보정 후 값을 대조
    p = doc[7]
    fr = (134.9, 78.2, 406.9, 179.7)
    out["fig5"] = panel(p, fr, (115, 134, 70, 190))

    # Fig.4 (p7): 본 축 + inset 두 개
    p = doc[6]
    out["fig4a_main"] = panel(p, (146.9, 82.6, 293.9, 173.4), (130, 147, 70, 180), 1e-2, exclude=(158, 88, 221, 153))
    out["fig4b_main"] = panel(p, (339.8, 82.6, 486.8, 173.4), (318, 340, 70, 180), exclude=(351, 88, 414, 151))
    out["fig4a_inset"] = panel(p, (169.4, 88.8, 208.7, 141.7), (148, 162.5, 90, 150), 1e-2)
    out["fig4b_inset"] = panel(p, (362.4, 88.8, 401.6, 141.7), (348, 357, 90, 150))

    # Fig.8, 9 (p22), Fig.10 (p24)
    p = doc[21]
    out["fig8a"] = panel(p, (142.0, 82.8, 298.8, 179.7), (120, 141, 75, 190), 1e-3)
    out["fig8b"] = panel(p, (332.6, 82.8, 489.4, 179.7), (320, 332, 75, 190))
    out["fig9a"] = panel(p, (146.7, 325.5, 303.5, 422.4), (125, 146, 320, 430), 1e-2)
    out["fig9b"] = panel(p, (334.9, 325.5, 491.7, 422.4), (320, 334, 320, 430))
    p = doc[23]
    out["fig10a"] = panel(p, (142.0, 310.8, 298.8, 407.8), (120, 141, 305, 415), 1e-3)
    out["fig10b"] = panel(p, (332.6, 310.8, 489.4, 407.8), (320, 332, 305, 415))

    # Fig.11 (p28), Fig.13 (p30)
    p = doc[27]
    out["fig11"] = panel(p, (185.4, 72.2, 462.6, 243.5), (160, 185, 65, 250))
    p = doc[29]
    out["fig13"] = panel(p, (196.1, 106.8, 453.5, 265.9), (170, 196, 100, 272))

    # Fig.6 (p9): 선 그래프의 표식(마커) 중심 → 값. x는 로그 축(50~2500 8점)
    p = doc[8]
    a1, b1, _ = fit_axis(p, (118, 130, 85, 175), 1e-3)  # 6a 왼쪽 축(1 step)
    a2, b2, _ = fit_axis(p, (272, 286, 85, 175))  # 6a 오른쪽 축(500 steps)
    a3, b3, _ = fit_axis(p, (325, 345, 70, 175))  # 6b 성공률

    def markers(color, frame_x):
        pts = []
        for d in p.get_drawings():
            r = d["rect"]
            if d["type"] == "fs" and d.get("fill") and all(abs(c - k) < 0.02 for c, k in zip(d["fill"], color)):
                in_legend = 403 <= r.x0 <= 430 and 138 <= r.y0 <= 160
                if frame_x[0] <= r.x0 and r.x1 <= frame_x[1] and r.width < 6 and r.y1 < 170 and not in_legend:
                    pts.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
        return sorted(pts)

    sizes = [50, 100, 200, 500, 1000, 1500, 2000, 2500]
    orange = markers((1.0, 0.5, 0.05), (135, 270))
    green = markers((0.17, 0.63, 0.17), (135, 270))
    blue = markers((0.12, 0.47, 0.71), (349, 482))
    red = markers((0.84, 0.15, 0.16), (349, 482))
    base6a, base6b = 165.8, 165.7  # 6a, 6b 축 바닥 y
    out["fig6"] = {
        "sizes": sizes,
        "pos_err_1step_deg": [a1 * y + b1 - (a1 * base6a + b1) for _, y in orange],
        "pos_err_500step_deg": [a2 * y + b2 - (a2 * base6a + b2) for _, y in green],
        "success_ensemble_pct": [a3 * y + b3 - (a3 * base6b + b3) for _, y in blue],
        "success_no_ensemble_pct": [a3 * y + b3 - (a3 * base6b + b3) for _, y in red],
    }
    Path(HERE / "figure_values.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for k, v in out.items():
        if k == "fig6":
            print(k)
            for kk, vv in v.items():
                print("  ", kk, [round(x, 4) if isinstance(x, float) else x for x in vv])
        else:
            print(k, [round(x["value"], 4) for x in v], "ci:", [None if x["ci"] is None else [round(c, 4) for c in x["ci"]] for x in v][:3])


if __name__ == "__main__":
    main()

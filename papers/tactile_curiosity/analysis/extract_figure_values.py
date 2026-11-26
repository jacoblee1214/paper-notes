"""논문 PDF의 벡터 그래프에서 수치를 읽어 figure_values.json으로 저장한다.

논문(arXiv v3)에는 결과 표가 없고 그림 2, 3, 4, 6만 결과를 담는다. 이 그림들의 막대, 오차 막대, 선, 점은 벡터로 들어 있어서
축 눈금(회색 격자선이나 눈금 글자)의 위치로 좌표를 보정하면 값을 읽을 수 있다.
결과는 그림 판독 값이며 저자가 표로 보고한 수치가 아니다(읽기 오차는 축 한 칸의 1% 안팎).

읽는 값:
  fig2b       : 단일 물체 오프라인 반환 (막대 7개 평균과 SEM, DrQ 점선)
  fig3b, fig3c: 다중 물체 오프라인 반환, 유효 물체 수 (막대와 SEM)
  fig4        : 8개 과제 학습 곡선의 끝점 (DSRL-SAC, DSRL-TacEx 세 가지)
  fig6        : 5개 과제 최종 반환(Final return)의 점과 표준오차 (방법 11가지)

실행: python3 analysis/extract_figure_values.py   (pymupdf 필요, 입력 paper/paper.pdf = arXiv v3)
출력: analysis/figure_values.json
"""
import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = Path(__file__).resolve().parent / "figure_values.json"


def text_spans(page, y0, y1, x0=0, x1=700):
    out = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                t = s["text"].strip()
                bx = s["bbox"]
                if t and y0 <= bx[1] and bx[3] <= y1 and x0 <= bx[0] and bx[2] <= x1:
                    out.append((t, (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2))
    return out


def linear(p0, v0, p1, v1):
    k = (v1 - v0) / (p1 - p0)
    return lambda p: v0 + (p - p0) * k


def bars_and_errors(page, region, labels, to_val):
    """region 안의 채워진 막대와 같은 x의 검은 수직 오차 막대를 읽는다."""
    x0, y0, x1, y1 = region
    dr = page.get_drawings()
    bars = sorted(
        (p["rect"] for p in dr
         if p["type"] == "f" and p.get("fill") is not None and len(p["items"]) == 1
         and x0 <= p["rect"].x0 and p["rect"].x1 <= x1 and y0 <= p["rect"].y0 and p["rect"].y1 <= y1
         and 8 < p["rect"].width < 20),
        key=lambda r: r.x0)
    errs = []
    for p in dr:
        if p.get("color") == (0.0, 0.0, 0.0) and len(p["items"]) == 1 and p["items"][0][0] == "l":
            a, b = p["items"][0][1], p["items"][0][2]
            if abs(a.x - b.x) < 0.01 and x0 <= a.x <= x1:
                errs.append((a.x, min(a.y, b.y), max(a.y, b.y)))
    out = {}
    for lab, r in zip(labels, bars):
        xc = (r.x0 + r.x1) / 2
        top = min(r.y0, r.y1)
        mean = to_val(top)
        e = [t for t in errs if abs(t[0] - xc) < 1.0]
        sem = None
        if e:
            sem = abs(to_val(e[0][1]) - to_val(e[0][2])) / 2
        out[lab] = {"mean": round(mean, 2), "sem": None if sem is None else round(sem, 2)}
    return out


def fig2b(doc):
    p = doc[5]
    # 격자선 y: 0, 100, 200, 300 (306.46, 284.17, 261.87, 239.58)
    to_val = linear(306.46, 0.0, 239.58, 300.0)
    res = bars_and_errors(p, (376, 225, 482, 308), ["T", "TV", "TVL", "V", "L", "VL", "RE"], to_val)
    drq = None
    for d in p.get_drawings():
        if d.get("dashes") and "3.21" in str(d["dashes"]) and 230 < d["rect"].y0 < 250:
            drq = round(to_val(d["rect"].y0), 1)
    return {"bars": res, "drq_dashed": drq}


def fig3(doc):
    p = doc[6]
    b = bars_and_errors(p, (276, 325, 366, 408), ["T", "TV", "V", "RE"], linear(405.63, 0.0, 336.33, 600.0))
    c = bars_and_errors(p, (413, 325, 503, 408), ["T", "TV", "V", "RE"], linear(405.64, 0.0, 334.57, 4.0))
    return {"fig3b": b, "fig3c": c}


FIG4_COLORS = {
    (0.0, 0.0, 0.0): "DSRL-SAC",
    (0.0, 0.427, 0.859): "TacEx tactile",
    (0.847, 0.106, 0.376): "TacEx vision+tactile",
    (0.902, 0.624, 0.0): "TacEx latent+tactile",
}
FIG4_TASKS = [  # (과제 이름, LIBERO-90 과제 번호, 열, 행)
    ("pick up the ketchup and put it in the tray", 58, 0, 0),
    ("put the chocolate pudding to the left of the plate", 69, 1, 0),
    ("pick up the chocolate pudding and put it in the tray", 61, 2, 0),
    ("pick up the book and place it in the back compartment of the caddy", 77, 3, 0),
    ("pick up the alphabet soup and put it in the tray", 55, 0, 1),
    ("turn on the stove and put the frying pan on it", 21, 1, 1),
    ("pick up the butter and put it in the tray", 56, 2, 1),
    ("turn on the stove", 20, 3, 1),
]


def fig4(doc):
    p = doc[7]
    # y 보정: 위 행 0.00 라벨 중심 161.95, 1.00 라벨 중심 111.05 / 아래 행 271.95, 221.05
    yv = [linear(161.95, 0.0, 111.05, 1.0), linear(271.95, 0.0, 221.05, 1.0)]
    # x 보정: (눈금 0의 x, 눈금 한 칸의 x, 한 칸의 값). 열별, 행별
    xs = {
        (0, 0): (147.7, 179.25, 2e5), (1, 0): (238.15, 257.85, 0.25e6), (2, 0): (328.55, 360.05, 2e5), (3, 0): (418.95, 450.5, 2e5),
        (0, 1): (147.7, 179.25, 2e5), (1, 1): (238.15, 269.65, 2e5), (2, 1): (328.55, 360.05, 2e5), (3, 1): (418.95, 440.85, 1e5),
    }
    col_edges = [229, 322, 413]

    def col(x):
        return sum(x > e for e in col_edges)

    best = {}
    # 같은 선이 다른 칸에 잘려 보이지 않는 사본으로 여러 번 들어 있다. 클립 영역이 비지 않고 선과 겹치는 경로만 쓴다.
    clips = {}
    for d in p.get_drawings(extended=True):
        if d["type"] == "clip":
            for lv in [lv for lv in clips if lv >= d["level"]]:
                del clips[lv]
            clips[d["level"]] = tuple(d["scissor"])
            continue
        if d["type"] != "s" or len(d["items"]) < 15:
            continue
        key = FIG4_COLORS.get(tuple(round(c, 3) for c in d["color"]))
        if key is None:
            continue
        r = d["rect"]
        if r.width < 40 or not clips:
            continue
        x0c = max(s_[0] for s_ in clips.values())
        y0c = max(s_[1] for s_ in clips.values())
        x1c = min(s_[2] for s_ in clips.values())
        y1c = min(s_[3] for s_ in clips.values())
        if not (x0c < x1c and y0c < y1c and r.y1 > y0c and r.y0 < y1c and r.x1 > x0c):
            continue
        c = col((r.x0 + r.x1) / 2)
        row = 0 if (r.y0 + r.y1) / 2 < 200 else 1
        end = d["items"][-1][2]
        k = (c, row, key)
        if k not in best or r.x1 > best[k][0]:
            best[k] = (r.x1, end, d["items"][0][1], len(d["items"]))
    out = {}
    for name, tid, c, row in FIG4_TASKS:
        x0, x1, dx = xs[(c, row)]
        res = {}
        for key in FIG4_COLORS.values():
            if (c, row, key) in best:
                _, end, start, nseg = best[(c, row, key)]
                res[key] = {"final": round(yv[row](end.y), 3), "segments": nseg,
                            "x_spacing": round(((end.x - x0) / (x1 - x0) * dx) / nseg),
                            "last_step": round((end.x - x0) / (x1 - x0) * dx)}
        out[str(tid)] = {"task": name, **res}
    return out


FIG6_COLORS = {
    (0.0, 0.427, 0.859): "Tactile",
    (0.847, 0.106, 0.376): "Tactile+vision",
    (0.902, 0.624, 0.0): "Tactile+latent",
    (0.6, 0.6, 0.6): "Latent",
    (0.337, 0.706, 0.914): "Image",
    (0.302, 0.302, 0.302): "State",
    (0.2, 0.133, 0.533): "DPPO",
    (0.067, 0.067, 0.067): "DSRL-PPO or DSRL-SAC",
    (0.478, 0.239, 0.722): "Summed taxels: info gain",
    (0.651, 0.337, 0.157): "Summed taxels: reward",
}
FIG6_ROWS = ["Ketchup in tray", "Soup in tray", "Butter in tray", "Book to back caddy", "Stove and frying pan"]


def fig6(doc):
    """5쪽이 아니라 17쪽의 그림 6. 왼쪽 열(Final return)만 읽는다."""
    p = doc[16]
    sp = text_spans(p, 118, 332, 170, 470)
    # 눈금 글자: y별로 묶어 x 보정. 왼쪽 열은 x < 321.
    rows_y = sorted({round(y, 0) for t, x, y in sp if t in ("0.0", "0.5", "1.0", "0.2", "0.4", "0.6")})
    axis_y = [129.5, 178.0, 226.5, 275.0, 323.5]   # 눈금 글자 y(행 아래)
    cal = []
    for ay in axis_y:
        lab = [(float(t), x) for t, x, y in sp if abs(y - ay) < 3 and x < 321]
        lab.sort()
        cal.append(linear(lab[0][1], lab[0][0], lab[-1][1], lab[-1][0]))
    dr = p.get_drawings()
    # 점: 방법 색으로 채워진 작은 도형(legend 제외: y < 332)의 중심. 오차 막대: 같은 색의 수평 선 경로
    pts = {}
    for d in dr:
        r = d["rect"]
        if d["type"] != "fs" or r.width > 12 or r.width < 5 or r.y1 > 332 or r.x0 > 321:
            continue
        for ck, name in FIG6_COLORS.items():
            fc = d.get("fill") if d.get("fill") != (1.0, 1.0, 1.0) else d.get("color")
            if fc and all(abs(a - b) < 0.02 for a, b in zip(fc, ck)):
                cy = (r.y0 + r.y1) / 2
                row = min(range(5), key=lambda i: abs((axis_y[i] - 40) - cy) if False else 0)
                pts.setdefault(name, []).append(((r.x0 + r.x1) / 2, cy))
    # 행 구분: y 구간
    bounds = [(80, 125.5), (126, 174), (175, 222), (223, 271), (272, 320)]

    def row_of(y):
        for i, (a, b) in enumerate(bounds):
            if a <= y <= b:
                return i
        return None

    err = {}
    for d in dr:
        if d["type"] != "s" or d.get("fill") is not None:
            continue
        r = d["rect"]
        if r.height > 6 or r.width < 1.0 or r.x0 > 321 or r.y1 > 332:
            continue
        for ck, name in FIG6_COLORS.items():
            if d.get("color") and all(abs(a - b) < 0.02 for a, b in zip(d["color"], ck)):
                err.setdefault(name, []).append((r.x0, r.x1, (r.y0 + r.y1) / 2))
    res = {rn: {} for rn in FIG6_ROWS}
    for name, lst in pts.items():
        for x, y in lst:
            i = row_of(y)
            if i is None:
                continue
            mean = cal[i](x)
            se = None
            for (a, b, ey) in err.get(name, []):
                if abs(ey - y) < 0.8 and a - 1 <= x <= b + 1 and b - a > 1.5:
                    se = max((cal[i](b) - cal[i](a)) / 2, 0.0)
            res[FIG6_ROWS[i]].setdefault(name, []).append({"mean": round(mean, 3), "se": None if se is None else round(se, 3)})
    return res


CURVE_COLORS = {
    (0.098, 0.463, 0.824): "T", (0.847, 0.106, 0.376): "TV", (0.882, 0.604, 0.0): "TVL",
    (0.337, 0.706, 0.914): "V", (0.6, 0.6, 0.6): "L", (0.0, 0.62, 0.451): "VL", (0.2, 0.133, 0.533): "DrQ",
}


def grasp_curves(doc):
    """그림 2a, 3a의 '물체를 잡는 단계 비율' 곡선. 마지막 10% 구간의 평균과 마지막 점을 읽는다."""
    jobs = {
        "fig2a": (5, (150, 330, 230, 310), linear(153.8, 0.0, 323.15, 3.0), linear(302.05, 0.0, 240.8, 0.5)),
        "fig3a": (6, (130, 232, 320, 410), linear(136.55, 0.0, 228.15, 5.0), linear(401.45, 0.0, 339.65, 0.8)),
    }
    out = {}
    for name, (pi, (x0, x1, y0, y1), fx, fy) in jobs.items():
        res = {}
        for d in doc[pi].get_drawings():
            r = d["rect"]
            if d["type"] != "s" or len(d["items"]) < 100 or not (x0 <= r.x0 and r.x1 <= x1 + 3 and y0 <= r.y0 and r.y1 <= y1 + 3):
                continue
            key = CURVE_COLORS.get(tuple(round(c, 3) for c in d["color"]))
            if key is None:
                continue
            pts = [(it[1].x, it[1].y) for it in d["items"] if it[0] == "l"] + [(d["items"][-1][2].x, d["items"][-1][2].y)]
            xs = [fx(x) for x, _ in pts]
            ys = [fy(y) for _, y in pts]
            xmax = max(xs)
            tail = [y for x, y in zip(xs, ys) if x >= 0.9 * xmax]
            res[key] = {"last": round(ys[-1], 3), "mean_last10pct": round(sum(tail) / len(tail), 3),
                        "max": round(max(ys), 3), "x_end": round(xmax, 2)}
        out[name] = res
    return out


def main():
    doc = pymupdf.open(PDF)
    out = {"_note": "그림 판독 값(벡터 좌표 보정). 저자가 표로 보고한 수치가 아니다.",
           "fig2b": fig2b(doc), **fig3(doc), "grasp_fraction_curves": grasp_curves(doc), "fig4_final": fig4(doc), "fig6_final_return": fig6(doc)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

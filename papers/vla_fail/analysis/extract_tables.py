"""논문 PDF(paper/paper.pdf, arXiv v2)의 표 1, 2, 4-11 숫자를 좌표 기반으로 읽어 tables_data.json 으로 저장한다.

글자 크기 10 = 평균, 크기 5 = 95% 신뢰구간 위·아래 값(위가 상한, 아래가 하한)이다.
논문 PDF는 저장소에 올리지 않으므로 먼저 `curl -L -o paper/paper.pdf https://arxiv.org/pdf/2606.21386v2` 로 받아야 한다.
실행: python3 analysis/extract_tables.py   (PyMuPDF 필요)
"""
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
doc = fitz.open(ROOT / "paper" / "paper.pdf")
DOC_V1 = ROOT / "paper" / "paper_v1.pdf"  # 있으면 v1 표 1·2도 읽는다
NUM = re.compile(r"\d\.\d+")


def words(page, d=None):
    out = []
    for w in (d or doc)[page - 1].get_text("words"):
        if NUM.fullmatch(w[4]):
            out.append({"x": w[0], "y": w[1], "h": w[3] - w[1], "v": float(w[4])})
    return out


def table(page, y_lo, y_hi, x_lo=0, d=None):
    """페이지의 [y_lo, y_hi) 구간에서 큰 숫자를 행·열로 묶고, 작은 숫자를 가장 가까운 큰 숫자에 붙인다."""
    ws = [w for w in words(page, d) if y_lo <= w["y"] < y_hi and w["x"] >= x_lo]
    big = [w for w in ws if w["h"] > 8]
    small = [w for w in ws if w["h"] <= 8]
    # 행: y 군집
    ys = sorted({round(w["y"], 1) for w in big})
    rows, cur = [], [ys[0]]
    for y in ys[1:]:
        if y - cur[-1] < 3.0:
            cur.append(y)
        else:
            rows.append(cur)
            cur = [y]
    rows.append(cur)
    row_of = {}
    for i, grp in enumerate(rows):
        for y in grp:
            row_of[y] = i
    xs = sorted({round(w["x"], 1) for w in big})
    cols, cur = [], [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] < 12:
            cur.append(x)
        else:
            cols.append(cur)
            cur = [x]
    cols.append(cur)
    col_of = {}
    for j, grp in enumerate(cols):
        for x in grp:
            col_of[x] = j
    grid = {}
    for w in big:
        r, c = row_of[round(w["y"], 1)], col_of[round(w["x"], 1)]
        grid[(r, c)] = {"mean": w["v"], "hi": None, "lo": None, "_x": w["x"], "_y": w["y"]}
    attach = {}
    for s in small:
        best, bd = None, 1e9
        for k, g in grid.items():
            dx, dy = s["x"] - g["_x"], s["y"] - g["_y"]
            if 5 < dx < 40 and -1 < dy < 14 and dx + abs(dy) < bd:
                best, bd = k, dx + abs(dy)
        if best is not None:
            attach.setdefault(best, []).append(s)
    for k, lst in attach.items():
        lst.sort(key=lambda s: s["y"])
        if len(lst) == 2:
            grid[k]["hi"], grid[k]["lo"] = lst[0]["v"], lst[1]["v"]
        elif len(lst) == 1:
            grid[k]["hi"] = lst[0]["v"]
    nr, nc = len(rows), len(cols)
    out = []
    for r in range(nr):
        out.append([[grid[(r, c)]["mean"], grid[(r, c)]["lo"], grid[(r, c)]["hi"]] if (r, c) in grid else None for c in range(nc)])
    return out


SPEC = {
    # 이름: (쪽, y 시작, y 끝, x 시작)
    "t1": (6, 345, 700, 200),
    "t2": (8, 140, 395, 200),
    "t4": (16, 110, 430, 200),
    "t5": (17, 125, 495, 200),
    "t6": (17, 505, 700, 200),
    "t7": (20, 140, 505, 200),
    "t8": (21, 135, 380, 200),
    "t9": (22, 220, 700, 200),
    "t10": (23, 220, 700, 200),
}


def main():
    res = {}
    for name, (pg, y0, y1, x0) in SPEC.items():
        tb = table(pg, y0, y1, x0)
        res[name] = tb
        print(name, len(tb), len(tb[0]))
    Path(ROOT / "analysis" / "tables_raw.json").write_text(json.dumps(res, ensure_ascii=False))
    if DOC_V1.exists():
        d1 = fitz.open(DOC_V1)
        res1 = {}
        for name, key in [("t1", "Table 1: Threshold-independent"), ("t2", "Table 2: Threshold-independent")]:
            for i, pg in enumerate(d1):
                if key in pg.get_text():
                    hdr = [w for w in pg.get_text("words") if w[4] == "ACE"]
                    y0 = max(h[1] for h in hdr) + 8
                    res1[name] = table(i + 1, y0, 780, 200, d1)
                    print("v1", name, "page", i + 1, len(res1[name]), len(res1[name][0]))
                    break
        Path(ROOT / "analysis" / "tables_raw_v1.json").write_text(json.dumps(res1, ensure_ascii=False))


if __name__ == "__main__":
    main()

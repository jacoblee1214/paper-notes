"""Extract bar heights and error-bar half-widths of Figure 4 (real-world system throughput)
directly from the vector drawing of the paper PDF (page 6).

Usage: python3 extract_fig4.py <paper.pdf> [out.json]
Needs PyMuPDF (pymupdf). Reads the PDF only; nothing is executed from it.
"""
import json
import sys

import pymupdf

pdf = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "fig4_extracted.json"
page = pymupdf.open(pdf)[5]
dr = page.get_drawings()

FAST = (0.2157, 0.6392, 0.8235)   # blue (fast tier)
SLOW = (0.4667, 0.4824, 0.4980)   # gray (slow tier)


def close(a, b, tol=0.01):
    return a is not None and all(abs(x - y) < tol for x, y in zip(a, b))


# y-axis calibration from the grid lines of panel 1: 0 at y0, 100 at y100
y0 = 178.4
grid = sorted({round(d["rect"].y0, 1) for d in dr
               if d["type"] == "s" and abs(d["rect"].y0 - d["rect"].y1) < 0.05
               and d["rect"].width > 100 and d["rect"].x0 < 250})
# grid y's (excluding axis y0): 163.5 -> 20 ... 103.7 -> 100
y100 = min(grid)
scale = (y0 - y100) / 100.0  # pt per unit


def val(y):
    return (y0 - y) / scale


bars = []
for d in dr:
    r = d["rect"]
    if d["type"] == "f" and d["fill"] and r.width > 10 and r.width < 25 and r.height > 0.3:
        if close(d["fill"], FAST):
            bars.append(("fast", r.x0, r.x1, val(r.y1), val(r.y0)))
        elif close(d["fill"], SLOW):
            bars.append(("slow", r.x0, r.x1, val(r.y1), val(r.y0)))
whisk = []
for d in dr:
    r = d["rect"]
    if d["type"] == "s" and r.width < 0.05 and r.height > 3 and d["color"] and close(d["color"], (0.2, 0.2, 0.2)):
        whisk.append((r.x0, val(r.y1), val(r.y0)))

rows = []
for x0 in sorted({round(b[1], 0) for b in bars}):
    cell = [b for b in bars if round(b[1], 0) == x0]
    seg = {b[0]: b for b in cell}
    top = max(b[4] for b in cell)
    xc = (cell[0][1] + cell[0][2]) / 2
    w = [k for k in whisk if abs(k[0] - xc) < 1.0]
    row = {
        "x": round(xc, 1),
        "total_from_bars": round(top, 2),
        "fast_segment": round(seg["fast"][4] - seg["fast"][3], 2) if "fast" in seg else None,
        "slow_segment": round(seg["slow"][4] - seg["slow"][3], 2) if "slow" in seg else None,
    }
    if w:
        lo, hi = w[0][1], w[0][2]
        row["err_low"], row["err_high"] = round(lo, 2), round(hi, 2)
        row["err_halfwidth"] = round((hi - lo) / 2, 2)
        row["err_center"] = round((hi + lo) / 2, 2)
    rows.append(row)
json.dump(rows, open(out, "w"), indent=1)
for r in rows:
    print(r)

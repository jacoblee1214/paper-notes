"""논문 그림 5·6의 마커 위치를 색으로 찾아 값으로 환산한다(그림 판독 값, 오차 약 ±0.005).

사용: python3 analysis/digitize_fig.py <paper.pdf>   (poppler의 pdfimages 필요)
결과: analysis/fig_digitized.json  — 그림 5(과제별 WHIRL 대 w/o WM)와 그림 6(Pick LEGO, Pull Drawer 세 곡선)의
      (상호작용 스텝, 굴림 개입 비율) 점 목록. 축 보정은 눈금 라벨 위치를 손으로 읽어 넣은 값이다.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
pdf = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "paper" / "paper.pdf")

COL = {"blue": (16, 78, 146), "red": (184, 69, 69), "orange": (232, 135, 55)}


def markers(img, color, r_open, xmin=0, ymin=0):
    a = np.asarray(img.convert("RGB")).astype(int)
    d = np.abs(a - np.array(COL[color])).sum(axis=2)
    m = d < 70
    yy, xx = np.ogrid[-r_open:r_open + 1, -r_open:r_open + 1]
    disk = (xx ** 2 + yy ** 2) <= r_open ** 2
    mo = ndi.binary_opening(m, structure=disk)
    lab, n = ndi.label(mo)
    cs = ndi.center_of_mass(mo, lab, range(1, n + 1))
    return sorted([(c[1], c[0]) for c in cs if c[1] > xmin and c[0] > ymin])  # (x, y) 픽셀. 축 밖(범례, 이웃 패널)은 뺀다


def cal(px, p0, p1, v0, v1):
    return v0 + (px - p0) * (v1 - v0) / (p1 - p0)


with tempfile.TemporaryDirectory() as td:
    subprocess.run(["pdfimages", "-png", "-f", "7", "-l", "8", pdf, f"{td}/a"], check=True, capture_output=True)
    f5 = Image.open(f"{td}/a-000.png")
    f6 = Image.open(f"{td}/a-002.png")
    w, h = f5.size
    # 그림 5: 5개 패널. 각 패널 = (x 픽셀 두 점과 값, y 픽셀 두 점과 값), 패널은 가로 w/5씩 자른 조각 기준
    CAL5 = {
        "Pick Cube":    dict(left=185, x=((210, 0), (1148, 7500)),   y=((880, 0.0), (343, 0.3))),
        "Pick LEGO":    dict(left=172, x=((192, 0), (1033, 6000)),   y=((885, 0.0), (265, 0.45))),
        "Pick & Place": dict(left=157, x=((504, 4000), (858, 8000)), y=((866, 0.0), (221, 0.20))),
        "Pull Drawer":  dict(left=124, x=((461, 2500), (1146, 7500)), y=((888, 0.0), (165, 0.8))),
        "Long Horizon": dict(left=112, x=((418, 20000), (1002, 60000)), y=((872, 0.0), (255, 0.24))),
    }
    out = {"fig5": {}, "fig6": {}}
    for i, (name, c) in enumerate(CAL5.items()):
        crop = f5.crop((i * (w // 5), 0, (i + 1) * (w // 5), h))
        (xa, va), (xb, vb) = c["x"]
        (ya, wa), (yb, wb) = c["y"]
        rec = {}
        for col, key in (("blue", "WHIRL"), ("red", "wo_WM")):
            pts = markers(crop, col, 8, xmin=c['left'], ymin=170)
            rec[key] = [[round(cal(x, xa, xb, va, vb)), round(cal(y, ya, yb, wa, wb), 3)] for x, y in pts]
        out["fig5"][name] = rec
    # 그림 6: 위 Pick LEGO, 아래 Pull Drawer (한 이미지, 위아래 패널)
    W6, H6 = f6.size
    CAL6 = {
        "Pick LEGO":   dict(box=(0, 120, W6, 712),   x=((180, 0), (1187, 7500)), y=((702, 0.0), (227, 0.45))),
        "Pull Drawer": dict(box=(0, 785, W6, 1370), x=((180, 0), (1187, 7500)), y=((1360, 0.0), (806, 0.75))),
    }
    # 위 두 패널의 픽셀 보정은 그림 6 원본 해상도 기준이다(아래 확인 단계에서 점 수로 검증).
    for name, c in CAL6.items():
        x0, y0, x1, y1 = c["box"]
        crop = f6.crop(c["box"])
        (xa, va), (xb, vb) = c["x"]
        (ya, wa), (yb, wb) = c["y"]
        rec = {}
        for col, key in (("blue", "Full"), ("orange", "alpha_i0"), ("red", "wo_WM")):
            pts = markers(crop, col, 6)
            rec[key] = [[round(cal(x + x0, xa, xb, va, vb)), round(cal(y + y0, ya, yb, wa, wb), 3)] for x, y in pts]
        out["fig6"][name] = rec

(ROOT / "analysis" / "fig_digitized.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
for fig in out:
    for t, r in out[fig].items():
        print(fig, t, {k: len(v) for k, v in r.items()}, {k: v[-1] for k, v in r.items()})

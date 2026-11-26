"""그림 6d(원문 8쪽)의 막대 높이를 픽셀에서 읽는다. 막대 수 세기와 (Ours, 최적, Class prior) 판독 값(N)을 fig6d_readings_raw.json으로 저장한다.
방법: 500 dpi 렌더링 -> 살구색 마스크로 막대, 검정 가로선으로 최적 힘, 회색 점선으로 Class prior. 눈금 3.0 N(상단 틀)과 0 N(하단 틀)로 환산. 해상도 약 0.007 N, 겹침 때문에 17, 18, 24번째 막대는 신뢰하지 않는다.
실행: python3 analysis/read_fig6d.py   (pdftoppm, numpy, Pillow)
"""
import json, subprocess, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as td:
    subprocess.run(["pdftoppm", "-r", "500", "-png", "-f", "8", "-l", "8", "-x", "150", "-y", "2100", "-W", "3900", "-H", "700",
                    str(ROOT / "paper" / "paper.pdf"), f"{td}/f"], check=True, capture_output=True)
    a = np.array(Image.open(next(Path(td).glob("f*.png"))).convert("RGB")).astype(int)
salmon = (a[..., 0] > 230) & (a[..., 1] > 130) & (a[..., 1] < 180) & (a[..., 2] > 110) & (a[..., 2] < 160)
dark = a.sum(2) < 200
gray = (abs(a[..., 0]-a[..., 1]) < 12) & (abs(a[..., 1]-a[..., 2]) < 12) & (a[..., 0] > 90) & (a[..., 0] < 190)
cols = np.where(salmon.sum(0) > 20)[0]
groups, s, p = [], cols[0], cols[0]
for c in cols[1:]:
    if c != p+1:
        groups.append((s, p)); s = c
    p = c
groups.append((s, p)); groups = [g for g in groups if g[1]-g[0] > 40]
y0, sc = 540.0, (540-110.5)/3   # 하단 틀 = 0 N, 상단 틀 = 3 N
f = lambda y: None if y is None else round((y0-y)/sc, 2)
out = []
for x0, x1 in groups:
    w = x1-x0+1; xs = slice(x0+int(.15*w), x1-int(.15*w))
    top = np.where(salmon[:, xs].sum(1) > 0.6*w*.7)[0].min()
    d = np.where(dark[120:535, xs].sum(1) > 0.6*w*.7)[0]+120
    g = np.where(gray[120:535, xs].sum(1) > 0.3*w*.7)[0]+120
    out.append((f(top), f(np.median(d)) if len(d) else None, f(np.median(g)) if len(g) else None))
print("bars:", len(out))
json.dump(out, open(ROOT / "analysis" / "fig6d_readings_raw.json", "w"))

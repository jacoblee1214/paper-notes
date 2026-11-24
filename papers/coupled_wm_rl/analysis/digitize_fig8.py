"""논문 그림 8(a-c)과 그림 5의 래스터 이미지에서 곡선의 끝점과 최종값을 픽셀로 읽는다.
이 값은 그림 판독 값이며 오차가 크다(약 ±1 픽셀 = 축 단위의 1-2%).
실행: python3 analysis/digitize_fig8.py   (numpy, pillow 필요)
입력: analysis/fig_crops/paper_fig8a.png, paper_fig8b.png, paper_fig8c.png, paper_fig5.png
      (논문 PDF에서 pdfimages로 추출한 래스터. 논문 그림, © 저자, CC BY 4.0, 출처 arXiv:2602.06219)
"""
import json, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {"DMO": (0, 114, 178), "NoDiffusion": (230, 159, 0), "PPO": (0, 158, 115),
       "SAC": (213, 94, 0), "NoRSSM": (204, 121, 167)}

def load(name):
    return np.array(Image.open(os.path.join(HERE, "fig_crops", name)).convert("RGB")).astype(int)

def _unused_gridlines(im, lo, hi):
    g = (abs(im[:, :, 0] - im[:, :, 1]) < 6) & (im[:, :, 0] > 200) & (im[:, :, 0] < 240)
    gc = g.sum(0); gr = g.sum(1)
    xs = [i for i in range(lo, hi) if gc[i] > 60]
    # 연속 픽셀 묶음의 중심
    def group(v):
        out = []; cur = []
        for i in v:
            if cur and i - cur[-1] > 2:
                out.append(np.mean(cur)); cur = []
            cur.append(i)
        if cur: out.append(np.mean(cur))
        return out
    return group(xs), group([i for i in range(im.shape[0]) if gr[i] > 100])

def curve_stats(im, x0, x1, y0, y1, xcal, ycal):
    """xcal=(px0, px1, v0, v1) 두 점 선형 보정. 반환: 곡선별 끝점 x, 끝 y(마지막 5열 중앙값)"""
    out = {}
    for k, c in COL.items():
        d = np.sqrt(((im - np.array(c)) ** 2).sum(2))
        m = d < 40
        m[:, :x0 + 2] = False; m[:, x1 - 1:] = False; m[:y0 + 2] = False; m[y1 - 1:] = False
        # 범례 영역 제외: 아래쪽(축 밖)은 위에서 y1로 잘림
        cols = np.where(m.sum(0) > 0)[0]
        if len(cols) == 0:
            continue
        xe = cols.max()
        ys = []
        for xx in range(xe - 4, xe + 1):
            ys += list(np.where(m[:, xx])[0])
        ye = float(np.median(ys))
        vx = xcal[2] + (xe - xcal[0]) * (xcal[3] - xcal[2]) / (xcal[1] - xcal[0])
        vy = ycal[2] + (ye - ycal[0]) * (ycal[3] - ycal[2]) / (ycal[1] - ycal[0])
        out[k] = {"x_end": round(float(vx), 3), "y_end": round(float(vy), 2)}
    return out

# 눈금 보정: (기준 픽셀, 기준 값) 두 점으로 선형 보정. 격자선 위치는 이미지에서 직접 읽었다.
# 이미지별: 패널 경계(왼쪽 x0,x1 / 오른쪽 x0,x1), 위아래 y0,y1, x(샘플, 1e7) 보정, x(시간, s) 보정, y 보정
CAL = {
  "Push-T":      dict(img="paper_fig8a.png", L=(56, 352), R=(389, 685), xs=((69.5, 0), (348, 5)),     xt=((402, 0), (628, 60000)), y=((38, 40), (204.5, 0))),
  "Push Cube":   dict(img="paper_fig8b.png", L=(56, 350), R=(387, 681), xs=((70, 0), (341, 2.5)),     xt=((400, 0), (667, 80000)), y=((63, 60), (197.5, 0))),
  "G1 Grab Box": dict(img="paper_fig8c.png", L=(56, 345), R=(383, 672), xs=((69, 0), (337, 2.5)),     xt=((396, 0), (669, 80000)), y=((49.5, 25), (197, 0))),
}

def lin(p, cal):
    (p0, v0), (p1, v1) = cal
    return v0 + (p - p0) * (v1 - v0) / (p1 - p0)

def curves(im, x0, x1, xcal, ycal, yt=34, yb=205):
    out = {}
    for k, c in COL.items():
        d = np.sqrt(((im - np.array(c)) ** 2).sum(2))
        m = d < 40
        m[:, :x0 + 2] = False; m[:, x1 - 1:] = False; m[:yt + 2] = False; m[yb - 1:] = False
        cols = np.where(m.sum(0) > 0)[0]
        if len(cols) == 0:
            continue
        xe = int(cols.max())
        ys = []
        for xx in range(xe - 4, xe + 1):
            ys += list(np.where(m[:, xx])[0])
        out[k] = {"x_end": round(float(lin(xe, xcal)), 3), "y_end": round(float(lin(float(np.median(ys)), ycal)), 1)}
    return out

res = {}
for task, c in CAL.items():
    im = load(c["img"])
    r = {"samples_e7": curves(im, c["L"][0], c["L"][1], c["xs"], c["y"]),
         "time_s": curves(im, c["R"][0], c["R"][1], c["xt"], c["y"])}
    # IQL 파선(회색 약 85,85,85)의 높이
    m = (abs(im[:, :, 0] - 85) < 25) & (abs(im[:, :, 1] - 85) < 25) & (abs(im[:, :, 2] - 85) < 25)
    rows = m[:, c["L"][0] + 5:c["L"][1] - 5].sum(1)
    rows[:40] = 0; rows[203:] = 0
    r["iql_line"] = round(float(lin(int(np.argmax(rows)), c["y"])), 1)
    res[task] = r
with open(os.path.join(HERE, "digitize_fig8_results.json"), "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))

# ---- 그림 5 (집계 곡선) ----
im5 = load("paper_fig5.png")
yc = ((70.5, 1.0), (424.0, 0.0))
res5 = {}
for side, (x0, x1, xc) in {"samples": (118, 703, ((144.5, 0.0), (676.5, 1.0))), "time": (784, 1369, ((810.5, 0.0), (1342.5, 1.0)))}.items():
    out = {}
    for k, c in COL.items():
        d = np.sqrt(((im5 - np.array(c)) ** 2).sum(2))
        m = d < 40
        m[:, :x0 + 3] = False; m[:, x1 - 2:] = False; m[:72] = False; m[447:] = False
        m[340:, x0 + 100:x1] = False  # 범례 영역 제외(곡선은 y>0.25 위쪽)
        cols = np.where(m.sum(0) > 0)[0]
        if len(cols) == 0: continue
        xe = int(cols.max())
        ys = []
        for xx in range(xe - 6, xe + 1): ys += list(np.where(m[:, xx])[0])
        out[k] = {"x_end": round(float(lin(xe, xc)), 3), "y_end": round(float(lin(float(np.median(ys)), yc)), 2)}
    res5[side] = out
mm = (abs(im5[:, :, 0] - 85) < 25) & (abs(im5[:, :, 1] - 85) < 25) & (abs(im5[:, :, 2] - 85) < 25)
rows = mm[:, 150:690].sum(1); rows[:72] = 0; rows[447:] = 0
res5["iql_line"] = round(float(lin(int(np.argmax(rows)), yc)), 3)
res["fig5_aggregate"] = res5
with open(os.path.join(HERE, "digitize_fig8_results.json"), "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps(res5, ensure_ascii=False, indent=1))


# ---- 문턱 도달 표본·시간 (그림 8, 판독) ----
def curve_xy(im, x0, x1, color, xcal, ycal, yt=34, yb=205):
    d = np.sqrt(((im - np.array(color)) ** 2).sum(2))
    m = d < 40
    m[:, :x0 + 2] = False; m[:, x1 - 1:] = False; m[:yt + 2] = False; m[yb - 1:] = False
    xs, ys = [], []
    for xx in range(x0 + 2, x1 - 1):
        r = np.where(m[:, xx])[0]
        if len(r):
            xs.append(lin(xx, xcal)); ys.append(lin(float(np.median(r)), ycal))
    return np.array(xs), np.array(ys)

def first_cross(xs, ys, thr, k=3):
    ok = ys >= thr
    for i in range(len(ok) - k + 1):
        if ok[i:i + k].all():
            return float(xs[i])
    return None

cross = {}
for task, c in CAL.items():
    im = load(c["img"])
    iql = res[task]["iql_line"]
    ppo_final = res[task]["samples_e7"]["PPO"]["y_end"]
    row = {}
    for side, (x0, x1, xc) in {"samples_e7": (c["L"][0], c["L"][1], c["xs"]), "time_s": (c["R"][0], c["R"][1], c["xt"])}.items():
        out = {}
        for meth in ["DMO", "PPO", "SAC", "NoRSSM"]:
            xs, ys = curve_xy(im, x0, x1, COL[meth], xc, c["y"])
            out["IQL선 도달_" + meth] = None if len(xs) == 0 else (None if first_cross(xs, ys, iql) is None else round(first_cross(xs, ys, iql), 3))
            out["PPO최종값 도달_" + meth] = None if len(xs) == 0 else (None if first_cross(xs, ys, ppo_final - 0.5) is None else round(first_cross(xs, ys, ppo_final - 0.5), 3))
        row[side] = out
    cross[task] = row
res["threshold_crossings"] = cross
with open(os.path.join(HERE, "digitize_fig8_results.json"), "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps(cross, ensure_ascii=False, indent=1))

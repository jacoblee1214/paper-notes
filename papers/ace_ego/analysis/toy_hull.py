"""토이: 같은 분포에서 뽑아도 에피소드 수가 많으면 볼록 껍질 면적이 커지는가 (논문 그림 6, 5.5절의 "4.8배" 해석용).

34개 에피소드(로봇)와 419개 에피소드(사람)의 손 위치 점들이 같은 분포에서 나왔다고 가정하고 면적비를 잰다.
에피소드 하나 = 시작점 + 랜덤 워크 20점. 분포는 2차원 정규(sigma=0.25)와 두꺼운 꼬리 t(3)의 두 가지.
가정이 강한 모형이다. 이 값은 논문의 면적비 4.8(0.296/0.062)을 설명하거나 부정하지 않고, 면적비가 표본 수에 얼마나 민감한지만 보인다.
실행: python3 analysis/toy_hull.py   (numpy, 20초 미만) -> analysis/toy_hull_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_hull_results.json")


def hull_area(p):
    pts = sorted(map(tuple, p))
    def half(points):
        h = []
        for q in points:
            while len(h) >= 2 and (h[-1][0]-h[-2][0])*(q[1]-h[-2][1]) - (h[-1][1]-h[-2][1])*(q[0]-h[-2][0]) <= 0:
                h.pop()
            h.append(q)
        return h
    lo, up = half(pts), half(pts[::-1])
    hull = np.array(lo[:-1] + up[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def episodes(rng, n, kind, jitter=0.0):
    if kind == "normal":
        start = rng.normal(0, 0.25, (n, 1, 2))
    else:
        start = 0.25 * rng.standard_t(3, (n, 1, 2)) / np.sqrt(3)  # 분산을 맞춘 t(3)
    steps = rng.normal(0, 0.25 / 6, (n, 20, 2))
    pts = start + np.cumsum(steps, axis=1)
    pts = pts + rng.normal(0, jitter, pts.shape)
    return pts.reshape(-1, 2)


rng = np.random.default_rng(0)
res = {}
for kind in ["normal", "t3"]:
    for jitter in [0.0, 0.03]:
        ratios = []
        for _ in range(300):
            a = hull_area(episodes(rng, 34, kind))
            b = hull_area(episodes(rng, 419, kind, jitter))
            ratios.append(b / a)
        r = np.array(ratios)
        res[f"{kind}_humanjitter{jitter}"] = {
            "mean_ratio": round(float(r.mean()), 2), "median": round(float(np.median(r)), 2),
            "p5": round(float(np.percentile(r, 5)), 2), "p95": round(float(np.percentile(r, 95)), 2),
            "share_ratio_ge_4.8": round(float((r >= 4.8).mean()), 3),
        }
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))

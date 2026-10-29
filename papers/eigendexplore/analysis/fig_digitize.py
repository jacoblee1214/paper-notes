"""Fig. 5(DexMachina) 막대 높이 픽셀 판독. 실행: python3 analysis/fig_digitize.py

원문 PDF 6쪽을 300 dpi로 렌더링해 파란(Joint-Space)·노란(EigenDEXplore) 막대 윗변을 색으로 찾고,
각 패널의 y축선(회색, 0–1 구간)으로 값을 환산한다. 본문에 적힌 네 값으로 판독 정확도를 확인한다.
원문 PDF가 paper/paper.pdf에 있어야 한다(README의 curl 명령).
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
BLUE, GOLD, AXIS = np.array([112, 152, 183]), np.array([225, 184, 78]), np.array([171, 171, 171])
HANDS = ["Allegro", "XHand", "Schunk", "Inspire"]
# 300 dpi 쪽 좌표의 패널 상자 (x0, y0, x1, y1): 축선과 막대 4쌍이 들어가게
PANELS = {
    ("box", "curriculum"): (600, 480, 960, 840),
    ("box", "task_only"): (1080, 480, 1440, 840),
    ("notebook", "curriculum"): (600, 1005, 960, 1365),
    ("notebook", "task_only"): (1080, 1005, 1440, 1365),
}
TEXT = {("box", "task_only", "Allegro"): (0.46, 0.80), ("box", "curriculum", "Schunk"): (0.59, 0.84)}  # Sec. 4.2


def render() -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "p6"
        subprocess.run(["pdftoppm", "-r", "300", "-f", "6", "-l", "6", "-png", "-singlefile", str(PDF), str(out)],
                       check=True, capture_output=True)
        return np.array(Image.open(f"{out}.png").convert("RGB")).astype(int)


def mask(img, color, tol=18):
    return (np.abs(img - color).max(axis=-1) <= tol)


def runs(cols):
    """True 열들을 연속 구간으로 묶는다."""
    out, start = [], None
    for i, v in enumerate(list(cols) + [False]):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i))
            start = None
    return out


def panel_values(img, box):
    x0, y0, x1, y1 = box
    sub = img[y0:y1, x0:x1]
    # y축선: 막대 왼쪽에서 가장 긴 회색 세로선. 그 왼쪽의 짧은 가로 눈금 5개(0, .25, .5, .75, 1)로 보정
    ax = mask(sub, AXIS, 6)
    col_len = ax.sum(axis=0)
    xa = int(np.argmax(col_len[: sub.shape[1] // 4]))
    tick_rows = np.nonzero(((sub[:, max(0, xa - 12):xa - 1].max(axis=-1) < 200).sum(axis=1)) >= 4)[0]
    groups = [list(g) for g in np.split(tick_rows, np.nonzero(np.diff(tick_rows) > 1)[0] + 1) if len(g)]
    ticks = sorted(float(np.mean(g)) for g in groups)[-5:]
    assert len(ticks) == 5, ticks
    slope, icpt = np.polyfit(ticks, [1.0, 0.75, 0.5, 0.25, 0.0], 1)
    y_top, y_zero = ticks[0], ticks[-1]
    vals = {}
    for name, color in (("joint", BLUE), ("ours", GOLD)):
        m = mask(sub, color)
        cols = m.sum(axis=0) > 20
        bars = [r for r in runs(cols) if r[1] - r[0] >= 8]
        tops = []
        for a, b in bars:
            yy = np.nonzero(m[:, a + 2:b - 2].any(axis=1))[0]
            tops.append(float(slope * yy.min() + icpt))
        vals[name] = tops
    return vals, {"tick_px": [round(t + y0, 1) for t in ticks]}


def main():
    img = render()
    res, check = {}, {}
    for (task, setting), box in PANELS.items():
        vals, meta = panel_values(img, box)
        assert len(vals["joint"]) == 4 and len(vals["ours"]) == 4, (task, setting, vals)
        res[f"{task}/{setting}"] = {h: {"joint": round(j, 3), "ours": round(o, 3)}
                                     for h, j, o in zip(HANDS, vals["joint"], vals["ours"])}
        res[f"{task}/{setting}"]["_tick_px"] = meta["tick_px"]
    for (task, setting, hand), (j, o) in TEXT.items():
        r = res[f"{task}/{setting}"][hand]
        check[f"{task}/{setting}/{hand}"] = {"text": [j, o], "read": [r["joint"], r["ours"]],
                                             "abs_err": [round(abs(r["joint"] - j), 3), round(abs(r["ours"] - o), 3)]}
    # 본문 "task reward만 쓸 때 평균 object-tracking 정확도 28% 향상"(Sec. 1) 재계산
    summ = {}
    for setting in ("curriculum", "task_only"):
        pairs = [(res[f"{t}/{setting}"][h]["joint"], res[f"{t}/{setting}"][h]["ours"]) for t in ("box", "notebook") for h in HANDS]
        j, o = np.array(pairs).T
        summ[setting] = {
            "mean_joint": round(float(j.mean()), 3), "mean_ours": round(float(o.mean()), 3),
            "ratio_of_means_pct": round(100 * (o.mean() / j.mean() - 1), 1),
            "mean_of_ratios_pct": round(100 * float((o / j - 1).mean()), 1),
            "per_cell_gain_pct": [round(100 * (b / a - 1), 1) for a, b in pairs],
        }
    out = {"values": res, "text_check": check, "summary": summ,
           "note": "8개 막대(손 4 × 과제 2)마다 판독. 오차막대는 판독하지 않음"}
    path = ROOT / "analysis" / "fig_digitize_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

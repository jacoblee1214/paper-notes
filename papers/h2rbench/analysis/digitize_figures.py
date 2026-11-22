#!/usr/bin/env python3
"""논문 그림에서 점 좌표를 읽어 digitized_points.json으로 저장한다.

- Fig. 4 (쪽 8, 벡터): 사람 데모 수별 진행도(Prog.) 곡선의 점. 눈금선(0과 100)에서 축을 보정한다.
- Fig. 5 (쪽 8, 벡터): 시뮬 대 실세계 산점도(로봇 + 최대 사람 데모). 대각선 끝점으로 축을 보정한다.
- Fig. 15a (쪽 24, 래스터): 로봇 전용 산점도. 색과 모양으로 표식을 찾고 눈금선으로 축을 보정한다.

필요한 것: poppler(pdftocairo, pdfimages), numpy, Pillow, scipy.
실행: python3 digitize_figures.py [paper.pdf]
Fig. 15a에서 다른 표식에 가려진 세 점은 눈으로 추정했다(json의 manual 표시).
읽기 오차는 벡터 그림이 0.1 퍼센트포인트 안팎, 래스터 그림이 0.5 퍼센트포인트 안팎이다.
"""
import json, os, re, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDF = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "paper" / "paper.pdf"

METHOD_R = {"Phantom": 12.94, "AMPLIFY": 30.2, "Point Policy": 90.19, "GHOST": 75.29}


def method_of(r):
    for m, v in METHOD_R.items():
        if abs(r - v) < 0.2:
            return m
    return None


def parse_svg_markers(svg_text):
    out = []
    for m in re.finditer(r"<path ([^>]*)/>", svg_text):
        a = m.group(1)
        fc = re.search(r'fill="rgb\(([\d.]+)%, ([\d.]+)%, ([\d.]+)%\)"', a)
        if not fc or 'fill-opacity="1"' not in a:
            continue
        meth = method_of(float(fc.group(1)))
        if not meth:
            continue
        d = re.search(r' d="([^"]*)"', a).group(1)
        tr = re.search(r'transform="matrix\(([^)]*)\)"', a)
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", d)]
        xs, ys = nums[0::2], nums[1::2]
        if tr:
            sc, _, _, _, tx, ty = [float(x) for x in tr.group(1).split(",")]
            cx = tx + sc * (min(xs) + max(xs)) / 2
            cy = ty - abs(sc) * (min(ys) + max(ys)) / 2
            size = (max(xs) - min(xs)) * abs(sc)
        else:
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            size = max(xs) - min(xs)
        shape = "circle" if "C" in d else ("tri" if d.count("L") == 2 else "quad")
        out.append(dict(method=meth, shape=shape, cx=cx, cy=cy, size=size))
    return out


def fig4_and_fig5(pdf, tmp):
    svg = Path(tmp) / "p8.svg"
    subprocess.run(["pdftocairo", "-svg", "-f", "8", "-l", "8", str(pdf), str(svg)],
                   check=True, stderr=subprocess.DEVNULL)
    pts = parse_svg_markers(svg.read_text())
    # Fig. 4: 처음 64개 표식(작업 4개 x 방법 4개 x 사람 데모 수준 4개)
    y0 = 193.803 - 0.40524 * 65.996462      # 값 0의 쪽 좌표
    y100 = 193.803 - 0.40524 * 251.139448   # 값 100
    tasks = ["Pick-and-Place", "Stacking", "Insertion", "Sequential"]
    fig4 = {}
    order = {}
    for i, p in enumerate(pts[:64]):
        meth = p["method"]
        k = order.get(meth, 0)
        order[meth] = k + 1
        task, lvl = tasks[k // 4], k % 4
        val = (y0 - p["cy"]) / (y0 - y100) * 100
        fig4.setdefault(meth, {}).setdefault(task, []).append(round(val, 2))
    # Fig. 5: 그다음 16개 표식(크기 4.98과 7.05)
    x0 = 317.88 + 0.47519 * 53.232983
    x100 = 317.88 + 0.47519 * 366.068375
    ay0 = 532.582 - 0.47519 * 44.825754
    ay100 = 532.582 - 0.47519 * 357.661146
    fig5 = {}
    for p in pts[68:84]:
        if p["size"] > 6:
            task = "Stacking"
        else:
            task = {"circle": "Insertion", "quad": "Sequential", "tri": "Pick-and-Place"}[p["shape"]]
        fig5.setdefault(p["method"], {})[task] = [
            round((p["cx"] - x0) / (x100 - x0) * 100, 2),
            round((ay0 - p["cy"]) / (ay0 - ay100) * 100, 2)]
    return fig4, fig5


def fig15a(pdf, tmp):
    import numpy as np
    from PIL import Image
    from scipy import ndimage as ndi
    base = Path(tmp) / "im"
    subprocess.run(["pdfimages", "-png", "-f", "24", "-l", "24", str(pdf), str(base)],
                   check=True, stderr=subprocess.DEVNULL)
    im = np.array(Image.open(str(base) + "-000.png").convert("RGB")).astype(int)
    cols = {"Phantom": (33, 102, 172), "AMPLIFY": (77, 172, 38),
            "Point Policy": (230, 132, 26), "GHOST": (192, 57, 43)}
    x0, x100, y0, y100 = 148.0, 1005.5, 939.0, 81.0   # 눈금선에서 읽은 픽셀 좌표
    found = {}
    for meth, c in cols.items():
        mask = np.abs(im - np.array(c)).sum(2) < 60
        mask[:, :150] = False
        mask[940:, :] = False
        mask[:60, :] = False
        lab, n = ndi.label(mask)
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            if len(ys) < 300:
                continue
            w = xs.max() - xs.min() + 1
            fill = len(ys) / (w * (ys.max() - ys.min() + 1))
            if w > 33:
                task = "Stacking"
            elif fill < 0.6:
                task = "Pick-and-Place"
            elif fill < 0.9:
                task = "Insertion"
            else:
                task = "Sequential"
            bx, by = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            found.setdefault(meth, {})[task] = [
                round((bx - x0) / (x100 - x0) * 100, 1),
                round((y0 - by) / (y0 - y100) * 100, 1)]
    manual = {  # 다른 표식에 가려져 눈으로 추정한 점
        ("Point Policy", "Insertion"): [62.6, 61.6],
        ("Point Policy", "Stacking"): [93.5, 80.9],
        ("GHOST", "Pick-and-Place"): [93.5, 80.0],
    }
    flags = []
    for (meth, task), v in manual.items():
        found.setdefault(meth, {})[task] = v
        flags.append(f"{meth}/{task}")
    return found, flags


def main():
    with tempfile.TemporaryDirectory() as tmp:
        fig4, fig5 = fig4_and_fig5(PDF, tmp)
        fig15a_pts, manual = fig15a(PDF, tmp)
    out = dict(
        note="Fig. 4: 사람 데모 수준별 Prog. (선택 순서 = 그림의 x 눈금). Fig. 5와 Fig. 15a: [시뮬 x, 실세계 y].",
        fig4_prog=fig4, fig5_max_human=fig5, fig15a_robot_only=fig15a_pts,
        fig15a_manual=manual)
    (HERE / "digitized_points.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("saved digitized_points.json; manual:", manual)


if __name__ == "__main__":
    main()

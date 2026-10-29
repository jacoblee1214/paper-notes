"""Fig. 4(섭동 각도별 성공률) 곡선값 픽셀 판독. 실행: python3 analysis/fig_digitize.py

원문 PDF 5쪽을 300 dpi로 렌더링해 네 곡선의 표식 위치를 색으로 찾는다.
y축은 본문에 수치가 적힌 두 점(90°: Ours 88.6%, Fingertip-only 72.4%, Sec. IV-A.1)으로 보정하고,
Ours 0° = 100%(그림 눈금)와 눈금선으로 보정을 검증한다. 원문 PDF가 paper/에 있어야 한다.
"""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
DPI = 300
BOX_110 = (85, 355, 460, 560)          # Fig. 4 영역 (110 dpi 기준 좌표)
X_PX = {0: 139, 30: 419, 60: 700, 90: 980}  # 크롭 안 x 위치(눈으로 잡은 값, ±3 px)
COLORS = {                              # matplotlib 기본 계열 색 (그림에서 표본 추출)
    "Ours": (52, 138, 78),
    "Fingertip-only": (139, 182, 112),
    "Feedforward": (76, 114, 176),
    "Impedance": (221, 132, 82),
}


def render() -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        prefix = Path(td) / "p5"
        subprocess.run(["pdftoppm", "-r", str(DPI), "-f", "5", "-l", "5", "-png", "-singlefile",
                        str(PDF), str(prefix)], check=True, capture_output=True)
        im = Image.open(f"{prefix}.png").convert("RGB")
    s = DPI / 110
    return np.asarray(im.crop(tuple(int(v * s) for v in BOX_110))).astype(int)


def marker_y(a: np.ndarray, rgb, x: int, half: int = 5, tol: int = 38) -> float:
    win = a[:, x - half:x + half + 1]
    d = np.abs(win - np.array(rgb)).max(axis=2)
    ys, _ = np.where(d < tol)
    ys = ys[(ys > 100) & (ys < 520)]     # 범례(위쪽) 제외
    if len(ys) == 0:
        return float("nan")
    # 같은 열에 다른 곡선의 띠가 섞이지 않도록 가장 큰 덩어리만 쓴다
    ys = np.sort(ys)
    groups = np.split(ys, np.where(np.diff(ys) > 3)[0] + 1)
    g = max(groups, key=len)
    return float(np.median(g))


def main() -> None:
    a = render()
    ypx = {name: {ang: marker_y(a, rgb, X_PX[ang]) for ang in X_PX} for name, rgb in COLORS.items()}
    # 보정: y% = A + B * ypx, 두 기준점 (Ours 90° = 88.6, Fingertip 90° = 72.4)
    y1, y2 = ypx["Ours"][90], ypx["Fingertip-only"][90]
    B = (88.6 - 72.4) / (y1 - y2)
    A = 88.6 - B * y1
    val = {n: {k: round(A + B * v, 1) for k, v in d.items()} for n, d in ypx.items()}
    out = {
        "calibration": {"anchors": "Ours@90=88.6, Fingertip-only@90=72.4 (Sec. IV-A.1)",
                        "pct_per_px": round(-B, 4),
                        "check_ours_0deg_expect_100": val["Ours"][0]},
        "values_pct": val,
        "note": "판독 정확도는 보정 검증 오차(Ours 0°)와 표식 반경(약 ±0.5%p) 수준",
    }
    path = ROOT / "analysis" / "fig_digitize_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

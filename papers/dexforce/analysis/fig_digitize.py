"""Fig. 9(k_f 허용 범위 막대와 오차 막대)를 픽셀로 판독한다. 실행: python3 analysis/fig_digitize.py

paper/paper.pdf 7쪽을 300 dpi로 그려 그림 영역을 자르고, 하늘색 막대의 위·아래 끝과 검은 오차 막대 cap의 위치를
y축 격자(0.000 = 261 px, 0.025 = 56.5 px, 300 dpi 기준)로 환산한다. 빨간 점선(k_f = 0.0045)을 검증점으로 쓴다.
결과: analysis/fig9_digitize_results.json
"""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
S = 300 / 110  # 110 dpi에서 정한 상자를 300 dpi로
BOX = (80, 510, 460, 630)
Y0, Y25 = 261.0, 56.5  # 크롭 좌표에서 k_f = 0, 0.025 격자선
BARS = [(168, 277, "unscrew nut"), (306, 415, "open AirPods"), (444, 553, "slide cube"),
        (582, 691, "reorient smiley"), (720, 829, "grasp battery"), (858, 966, "flip box")]


def kf(y: float) -> float:
    return (Y0 - y) * 0.025 / (Y0 - Y25)


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-r", "300", "-f", "7", "-l", "7", "-png", "-singlefile", str(PDF),
                        f"{td}/p7"], check=True, capture_output=True)
        im = Image.open(f"{td}/p7.png").crop(tuple(int(v * S) for v in BOX))
    a = np.asarray(im.convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    blue = (abs(r - 135) < 25) & (abs(g - 206) < 25) & (abs(b - 235) < 25)
    black = (r < 70) & (g < 70) & (b < 70)
    red_rows = np.where(((r > 200) & (g < 80) & (b < 90)).sum(1) > a.shape[1] * 0.3)[0]
    out = {"calibration": {"red_line_kf_read": round(kf(red_rows.mean()), 5), "red_line_kf_paper": 0.0045,
                           "kf_per_px": round(0.025 / (Y0 - Y25), 7)}, "bars": {}}
    for x0, x1, name in BARS:
        cols = [x0 + 6, x0 + 15, x1 - 15, x1 - 6]
        top = np.median([np.where(blue[:, c])[0].min() for c in cols]) - 0.5
        bot = np.median([np.where(blue[:, c])[0].max() for c in cols]) + 0.5
        c = (x0 + x1) // 2
        caps, groups = [y for y in range(a.shape[0]) if black[y, c - 8:c + 9].sum() >= 14], []
        for y in caps:
            if groups and y - groups[-1][-1] <= 2:
                groups[-1].append(y)
            else:
                groups.append([y])
        # 첫·마지막 묶음은 그림 테두리다
        capk = sorted((kf(float(np.mean(gp))) for gp in groups[1:-1]), reverse=True)
        mx, mn = kf(top), kf(bot)
        row = {"max_mean": round(mx, 5), "min_mean": round(mn, 5)}
        if len(capk) >= 2:
            row["max_sd"] = round((capk[0] - capk[1]) / 2, 5)
        if len(capk) >= 4:
            row["min_sd"] = round((capk[2] - capk[3]) / 2, 5)
        row["caps_read"] = [round(v, 5) for v in capk]
        out["bars"][name] = row
    path = ROOT / "analysis" / "fig9_digitize_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

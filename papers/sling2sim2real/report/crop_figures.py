"""논문 PDF(paper/paper.pdf)에서 그림 네 장을 잘라 report/figures/paper_fig*.png로 저장한다.
좌표는 200 dpi 기준 픽셀(x, y, 너비, 높이). 쪽 번호는 1-based. 실행: python3 report/crop_figures.py
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
FIG = ROOT / "report" / "figures"
TMP = ROOT / ".scratch" / "crop"

CROPS = [  # (출력 이름, 쪽, x, y, 너비, 높이)
    ("paper_fig1_teaser", 1, 883, 440, 683, 690),
    ("paper_fig2_pipeline", 3, 120, 127, 1480, 653),
    ("paper_fig3_bands", 6, 50, 60, 767, 500),
    ("paper_fig4_trajectories", 7, 90, 140, 1440, 232),
]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    for name, page, x, y, w, h in CROPS:
        base = TMP / name
        subprocess.run(["pdftoppm", "-r", "200", "-png", "-f", str(page), "-l", str(page),
                        "-x", str(x), "-y", str(y), "-W", str(w), "-H", str(h), str(PDF), str(base)],
                       check=True)
        produced = sorted(TMP.glob(f"{name}-*.png"))[-1]
        produced.replace(FIG / f"{name}.png")
        print("wrote", FIG / f"{name}.png")


if __name__ == "__main__":
    main()

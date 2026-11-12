"""논문 PDF에서 그림을 잘라 리포트·PPT·게시본에 쓴다.

실행: python3 report/crop_figures.py  -> report/figures/paper_fig2_overview.png
라이선스: arXiv abs 페이지 기준 CC BY 4.0. PDF 본문에 충돌하는 저작권 고지는 없음(pdftotext 확인).
  → 게시본에 출처와 라이선스를 표기하고 넣는다. Figure 1(Humanoid-X 렌더)과 Figure 4(실물 사진·참조 모션)는
    제3자 유래 이미지가 있어 게시본에 넣지 않는다.
좌표는 150 dpi 렌더링 기준으로 눈으로 잡았다.
"""
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
DPI = 150
# 2페이지 상단 Figure 2 (논문 원본 좌표계, 150 dpi)
BOX = (110, 96, 1212, 474)

OUT.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as td:
    prefix = Path(td) / "p2"
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-f", "2", "-l", "2", "-singlefile",
                    str(PDF), str(prefix)], check=True)
    im = Image.open(f"{prefix}.png").crop(BOX)
    im.save(OUT / "paper_fig2_overview.png")
    print("paper_fig2_overview.png", im.size)

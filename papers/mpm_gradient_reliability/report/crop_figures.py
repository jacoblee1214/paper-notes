"""논문 PDF에서 리포트·PPT·게시본용 그림을 잘라낸다. 실행: python3 report/crop_figures.py

arXiv 2609.34666v1은 CC BY 4.0이고, PDF에 다른 권리 표기가 없다(2026-11-11 확인: 1쪽과 본문 grep).
그래서 논문 그림은 © 저자, CC BY 4.0 표기와 출처를 붙여 게시본에도 쓴다.
좌표는 110 dpi 기준 (x0, y0, x1, y1). 실제 출력은 DPI/110 배로 키운다.
"""
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
DPI = 220
# (이름, 쪽, 110 dpi 기준 box)
CROPS = [
    ("paper_fig1_teaser", 1, (478, 366, 862, 562)),
    ("paper_fig2_findings", 3, (80, 82, 868, 292)),
]

OUT.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as td:
    for name, page, box in CROPS:
        prefix = Path(td) / f"p{page}"
        if not Path(f"{prefix}.png").exists():
            subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-f", str(page), "-l", str(page),
                            "-singlefile", str(PDF), str(prefix)], check=True)
        s = DPI / 110
        im = Image.open(f"{prefix}.png").crop(tuple(int(v * s) for v in box))
        im.save(OUT / f"{name}.png")
        print(name, im.size)

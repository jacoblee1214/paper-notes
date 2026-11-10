"""논문 PDF에서 리포트·PPT·게시본용 그림을 잘라낸다. 실행: python3 report/crop_figures.py

Environmental PDEs(arXiv:2608.00272v1)는 arXiv 라이선스 CC BY 4.0이고, PDF 1쪽과 전체 쪽에 IEEE 등 다른 권리 표기가 없다(2026-11-10 확인).
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
    ("paper_fig1_teaser", 1, (478, 222, 862, 420)),
    ("paper_fig2_swe", 5, (500, 92, 930, 968)),
    ("paper_fig3_adv", 6, (497, 72, 928, 758)),
    ("paper_fig4_swe_rmse", 7, (88, 72, 462, 283)),
    ("paper_fig5_adv_rmse", 7, (88, 388, 462, 603)),
    ("paper_table1", 7, (85, 826, 462, 919)),
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

"""논문 PDF에서 리포트·PPT용 그림을 잘라낸다 (개인 학습용; 논문이 CC BY가 아니므로 게시본에는 쓰지 않는다). 실행: python3 report/crop_figures.py"""
import subprocess, tempfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
DPI = 220
# (이름, 페이지, 60dpi 기준 box: x0, y0, x1, y1)
CROPS = [
    ("paper_fig1_teaser", 1, (262, 138, 470, 400)),
    ("paper_fig2_objects", 2, (22, 22, 490, 268)),
    ("paper_fig3_observations", 4, (15, 40, 505, 245)),
    ("paper_fig5_pipeline", 4, (262, 292, 500, 425)),
]

OUT.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as td:
    for name, page, box in CROPS:
        prefix = Path(td) / f"p{page}"
        subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-f", str(page), "-l", str(page),
                        "-singlefile", str(PDF), str(prefix)], check=True)
        s = DPI / 60
        im = Image.open(f"{prefix}.png").crop(tuple(int(v * s) for v in box))
        im.save(OUT / f"{name}.png")
        print(name, im.size)

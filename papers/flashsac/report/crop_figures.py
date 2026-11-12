"""논문 PDF에서 리포트·PPT·게시본용 그림을 잘라낸다. 실행: python3 report/crop_figures.py

FlashSAC(arXiv:2604.04539v2)는 arXiv 라이선스 CC BY 4.0이고, PDF에 다른 권리 표기가 없다(2026-11-12 확인).
그래서 아래 그래프와 구조도는 © 저자, CC BY 4.0 표기와 출처를 붙여 공개본에도 쓴다.
Fig. 1(시뮬레이터 장면 합성)과 Fig. 6(실로봇 사진)은 제3자 장면과 제품 사진이 섞여 있어 자르지 않는다.
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
    ("paper_fig2_architecture", 5, (285, 84, 830, 222)),
    ("paper_fig3_gpu_state", 7, (115, 78, 832, 344)),
    ("paper_fig4_cpu_state", 8, (110, 80, 832, 232)),
    ("paper_fig8_scaling", 11, (110, 84, 832, 218)),
    ("paper_fig9_architecture", 11, (118, 282, 832, 402)),
    ("paper_fig10_exploration", 12, (285, 84, 650, 225)),
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

"""논문 그림 두 장을 원문 PDF에서 잘라 report/figures/에 저장한다.

  paper_fig2.png : 그림 2 (원격조작 패러다임, 독립 설정과 협업 설정; 원문 3쪽)
  paper_fig3.png : 그림 3 (시스템 구조도; 원문 5쪽)

고른 이유: 저작자가 만든 도식이고, 실로봇 사진(그림 1, 4, 5의 장면), 사람 손 사진(그림 7),
시뮬레이터 화면 캡처(그림 8, 9)는 뺀다.
라이선스: arXiv abs 페이지 CC BY 4.0, 원문 PDF에 저작권 표시 없음. 출처와 라이선스를 적는다.
잘라낸 영역은 쪽 안의 pt 좌표이며 캡션 글자는 포함하지 않는다.

실행: python3 report/crop_figures.py   (pymupdf, Pillow)
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
JOBS = {
    "paper_fig2.png": (2, (50, 62, 575, 272)),   # 0-based 쪽 번호, 자를 영역(pt)
    "paper_fig3.png": (4, (50, 56, 560, 171)),
}


def trim(img: Image.Image) -> Image.Image:
    bg = Image.new("RGB", img.size, (255, 255, 255))
    box = ImageChops.difference(img.convert("RGB"), bg).getbbox()
    return img.crop(box) if box else img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PDF)
    for name, (page_idx, (x0, y0, x1, y1)) in JOBS.items():
        pix = doc[page_idx].get_pixmap(dpi=300, clip=pymupdf.Rect(x0, y0, x1, y1))
        tmp = OUT / (name + ".tmp.png")
        pix.save(tmp)
        img = trim(Image.open(tmp))
        img.save(OUT / name)
        tmp.unlink()
        print("wrote", OUT / name, img.size)


if __name__ == "__main__":
    main()

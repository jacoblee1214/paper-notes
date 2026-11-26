"""논문 그림 세 장(막대와 산점도 차트)을 원문 PDF에서 잘라 report/figures/에 저장한다.

  paper_fig3.png : 그림 3 (원자별 성공률, 원문 6쪽)
  paper_fig4.png : 그림 4 (쌍 과제의 전이 격차 TG, 원문 7쪽)
  paper_fig8.png : 그림 8 (Pi0.5 섭동 성공률, 원문 19쪽)

고른 이유: 저작자가 matplotlib으로 그린 차트이며 로봇 사진, 화면 캡처, 시뮬레이션 프레임, 로고가 없다.
쪽 머리의 기관 로고(이미지 객체)는 잘라낸 영역에 들어가지 않는다. 사진이 든 그림 1, 2, 5, 6, 7은 쓰지 않는다.
라이선스: arXiv abs 페이지 CC BY 4.0, 원문 PDF 첫 쪽과 끝 쪽에서 저작권 표시를 찾지 못했다.
잘라낸 영역은 이미지 객체의 경계 상자(pymupdf get_image_info)이며 캡션 글자는 포함하지 않는다.

실행: python3 report/crop_figures.py   (pymupdf, Pillow; paper/paper.pdf가 있어야 함)
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
JOBS = {
    "paper_fig3.png": (5, (72.0, 88.4, 540.0, 272.9)),   # 0-based 쪽 번호, 자를 영역(pt)
    "paper_fig4.png": (6, (189.0, 88.4, 423.0, 307.3)),
    "paper_fig8.png": (18, (83.7, 315.2, 528.3, 489.9)),
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

"""논문 그림에서 도식·그래프 패널만 잘라 report/figures/에 저장한다.

  paper_fig1_strip.png : 그림 1 아래쪽의 접근 방식 도식과 토큰 수 비교 (원문 1쪽, 위쪽 로봇 사진은 제외)
  paper_fig2_mask.png  : 그림 2 오른쪽의 인과 어텐션 마스크 도식 (원문 3쪽, 왼쪽의 사진 포함 패널은 제외)
  paper_fig6.png       : 그림 6 지연-성공률 버블 차트 (원문 9쪽)

고른 이유: 저작자가 만든 도식과 차트이고 실로봇 사진, 시뮬레이션 화면, 데이터셋 프레임은 자른 영역에 없다.
라이선스: arXiv abs 페이지 CC BY 4.0, 원문 PDF에 저작권 표시 없음. 출처는 아래 CREDIT과 같이 적는다.

실행: python3 report/crop_figures.py   (pymupdf, Pillow)
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
JOBS = {
    "paper_fig1_strip.png": (0, (115, 345, 530, 404)),
    "paper_fig2_mask.png": (2, (400, 72, 580, 215)),
    "paper_fig6.png": (8, (300, 70, 520, 190)),
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
        trim(Image.open(tmp)).save(OUT / name)
        tmp.unlink()
        print("wrote", name)


if __name__ == "__main__":
    main()

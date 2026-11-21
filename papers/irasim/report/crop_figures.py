"""논문 그림 세 장을 원문 PDF에서 잘라 report/figures/에 저장한다.

  paper_fig2.png : 그림 2의 (b)와 (c) 패널 (조건 방식 도식, 원문 4쪽). (a) 패널은 데이터셋 프레임 사진을 담고 있어 뺌
  paper_fig4.png : 그림 4 (사람 선호 평가 막대, 원문 9쪽)
  paper_fig5.png : 그림 5 (스케일링 곡선, 원문 9쪽)

고른 이유: 저작자가 만든 도식과 차트이고, 데이터셋 영상 프레임, 실로봇 사진, 시뮬레이터 화면 캡처가 없다.
그림 1, 3, 6–9, 10–15는 데이터셋 또는 생성 영상 프레임을 담고 있어 공개본에서 뺀다.
라이선스: arXiv abs 페이지 CC BY 4.0 (arXiv v2 PDF에 저작권 표시 없음). 출처와 라이선스를 적는다.
잘라낸 영역은 그림 벡터와 이미지 경계 상자의 합집합이며, 캡션은 제외한다.

실행: python3 report/crop_figures.py   (pymupdf, Pillow)
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
# 0-based 쪽 번호, 자를 영역 (pt, x0 y0 x1 y1)
JOBS = {
    "paper_fig2.png": (3, (270, 76, 542, 338)),
    "paper_fig4.png": (8, (70, 76, 542, 202)),
    "paper_fig5.png": (8, (68, 242, 542, 407)),
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

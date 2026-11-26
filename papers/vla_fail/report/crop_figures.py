"""논문 그림 두 장을 원문 PDF에서 잘라 report/figures/에 저장한다.

  paper_fig2.png : 그림 2 (성공·실패 에피소드의 예측 EEF x 위치와 청크 겹침, 원문 5쪽)
  paper_fig3.png : 그림 3 (X-VLA Blocks의 정밀도-재현율 곡선과 PDT-정밀도 곡선, 원문 15쪽)

고른 이유: 저자가 그린 그래프이고 사진, 사람, 시뮬레이터 화면, 상표 이미지가 없다. 그림 1(a)는 로봇 카메라 화면이 있어 쓰지 않는다.
라이선스: arXiv abs 페이지 CC BY 4.0, 원문 PDF에 © 등 저작권 표시 없음. 공개본에도 넣되 출처와 라이선스를 적는다.
캡션 글자는 잘라내지 않고 포함하지 않는다(영역은 그림 본체만).

실행: python3 report/crop_figures.py   (pymupdf, Pillow). 원문은 `curl -L -o paper/paper.pdf https://arxiv.org/pdf/2606.21386v2` 로 받는다.
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
JOBS = {
    "paper_fig2.png": (4, (100, 64, 512, 172)),   # 0-based 쪽 번호, 자를 영역(pt)
    "paper_fig3.png": (14, (104, 74, 516, 222)),
}


def trim(img: Image.Image) -> Image.Image:
    bg = Image.new("RGB", img.size, (255, 255, 255))
    box = ImageChops.difference(img.convert("RGB"), bg).getbbox()
    return img.crop(box) if box else img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PDF)
    for name, (page_idx, (x0, y0, x1, y1)) in JOBS.items():
        pix = doc[page_idx].get_pixmap(dpi=250, clip=pymupdf.Rect(x0, y0, x1, y1))
        tmp = OUT / (name + ".tmp.png")
        pix.save(tmp)
        img = trim(Image.open(tmp))
        img.save(OUT / name)
        tmp.unlink()
        print("wrote", OUT / name, img.size)


if __name__ == "__main__":
    main()

"""원문 PDF(arXiv v3)에서 차트와 도식 패널만 잘라 report/figures/에 저장한다.

  paper_fig1.png        : 그림 1 도식 (입력 영상 세 장은 자르고 모델 블록만, 원문 2쪽)
  paper_fig2_charts.png : 그림 2의 (a), (b) 차트 (위쪽 시뮬 렌더 프레임은 제외, 원문 6쪽)
  paper_fig3_charts.png : 그림 3의 (a), (b), (c) 차트 (위쪽 시뮬 렌더 프레임은 제외, 원문 7쪽)
  paper_fig4.png        : 그림 4 학습 곡선 8장 (원문 8쪽, 범례 포함, 캡션 글자는 제외)

고른 이유: 저작자 본인이 만든 도식과 그래프이고 사진, 사람, 시뮬 렌더 프레임이 없다.
라이선스: arXiv abs 페이지의 라이선스 링크가 CC BY 4.0이고 원문 PDF에서 ©, copyright, All rights reserved 문구를 찾지 못했다.
공개본에는 "© 저자, CC BY 4.0, 출처 arXiv:2609.40134 (v3)"를 붙여 싣는다.

실행: python3 report/crop_figures.py   (pymupdf, Pillow)
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
JOBS = {
    "paper_fig1.png": (1, (167, 72, 510, 236)),         # 0-based 쪽 번호, 자를 영역(pt)
    "paper_fig2_charts.png": (5, (105, 160, 512, 335)),
    "paper_fig3_charts.png": (6, (100, 283, 512, 432)),
    "paper_fig4.png": (7, (95, 62, 520, 346)),
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

"""논문 그림 네 장을 원문 PDF에서 잘라 report/figures/에 저장한다.

  paper_fig2a.png : 그림 2(a) (Dubins TPR/TNR 산점도, 원문 6쪽). 패널 (b)의 렌더 프레임은 뺀다.
  paper_fig5.png  : 그림 5 (조종 회수율 막대, 원문 7쪽)
  paper_fig8.png  : 그림 8 (정책 성공률 막대, 원문 8쪽)
  paper_fig14.png : 그림 14 (근사 기울기 대 전체 기울기 보상 곡선, 원문 32쪽)

고른 이유: 저작자 본인이 만든 그래프이고 사진, 사람, 시뮬레이터·렌더 프레임, 상표 이미지가 없다.
과제 아이콘(이모지)이 들어간 그림 22, 26은 쓰지 않았다.
라이선스: arXiv abs 페이지의 license 링크가 CC BY 4.0이고 원문 PDF에 저작권 표시가 없다.
공개본에도 출처를 적고 넣는다(© 저자, CC BY 4.0, 출처 arXiv:2606.00267).

실행: python3 report/crop_figures.py   (pymupdf, Pillow)
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
JOBS = {
    "paper_fig2a.png": (5, (306, 244, 412, 341)),   # 0-based 쪽 번호, 자를 영역(pt)
    "paper_fig5.png": (6, (425, 600, 510, 668)),
    "paper_fig8.png": (7, (443, 540, 508, 603)),
    "paper_fig14.png": (31, (343, 72, 506, 188)),
}


def trim(img: Image.Image) -> Image.Image:
    bg = Image.new("RGB", img.size, (255, 255, 255))
    box = ImageChops.difference(img.convert("RGB"), bg).getbbox()
    if not box:
        return img
    x0, y0, x1, y1 = box
    pad = 6
    return img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PDF)
    for name, (page_idx, (x0, y0, x1, y1)) in JOBS.items():
        pix = doc[page_idx].get_pixmap(dpi=400, clip=pymupdf.Rect(x0, y0, x1, y1))
        tmp = OUT / (name + ".tmp.png")
        pix.save(tmp)
        img = trim(Image.open(tmp))
        img.save(OUT / name)
        tmp.unlink()
        print("wrote", OUT / name, img.size)


if __name__ == "__main__":
    main()

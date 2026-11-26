"""논문 그림 한 장을 원문 PDF에서 잘라 report/figures/에 저장한다.

  paper_fig7.png : 그림 7 (효율-품질 개념도, 원문 9쪽)

고른 이유: 저작자 본인이 그린 개념 도식(수치 축 없음)이다. 그림 1, 2, 3, 5, 6은 사진·영상 프레임·로봇 사진이 들어 있고,
그림 4는 3D 렌더 이미지라서 넣지 않았다.
라이선스: arXiv abs 페이지의 라이선스 링크가 CC BY 4.0이고 원문 PDF에 저작권 표시가 없다. 출처와 라이선스를 적어 쓴다.
잘라낸 영역은 쪽 좌표(pt)이며 캡션 글자는 포함하지 않는다.

실행: python3 report/crop_figures.py   (pymupdf, Pillow; paper/paper.pdf 필요: curl -L -o paper/paper.pdf https://arxiv.org/pdf/2609.11753v1)
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
JOBS = {
    "paper_fig7.png": (8, (330, 268, 505, 392)),   # 0-based 쪽 번호, 자를 영역(pt)
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

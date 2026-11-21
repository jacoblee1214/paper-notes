"""논문 그림 두 장을 원문 PDF에서 잘라 report/figures/에 저장한다.

  paper_fig3.png : 그림 3 (의미 클래스 병목: 다중 모드 실패와 OOV 오염), 원문 6쪽

고른 이유: 저작자가 만든 시뮬레이션 그래프다. BeamNG 화면 캡처가 들어간 그림 1·8, 실차 사진과
영상 프레임이 들어간 그림 6·11은 뺐다(제3자 소프트웨어 화면과 실물 사진).
라이선스: arXiv abs 페이지 CC BY 4.0, 원문 PDF에 저작권·권리 표시 없음(pdftotext로 ©, copyright 검색).
잘라낸 영역은 쪽 좌표(pt)이며 캡션 글자는 뺀다. 공개본에도 넣고 출처를 적는다.

실행: python3 report/crop_figures.py   (pymupdf, Pillow)
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
JOBS = {
    "paper_fig3.png": (5, (262, 399, 507, 580)),   # 0-based 쪽 번호, 자를 영역(pt)
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

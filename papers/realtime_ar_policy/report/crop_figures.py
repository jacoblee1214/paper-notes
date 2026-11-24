"""논문 그림 일부를 원문 PDF에서 잘라 report/figures/에 저장한다 (차트와 도식만, 사진·시뮬 화면·로봇 사진 제외).

  paper_fig1_method.png : 그림 1 가운데 블록 (다중 궤적 디코딩과 제약 디코딩 도식). 양옆의 관측 사진 두 장은 자른다.
  paper_fig3.png        : 그림 3 (토큰 길이 분포와 Kinetix 차트, 원문 4쪽)
  paper_fig4.png        : 그림 4 (LIBERO 절제 차트, 원문 7쪽)
  paper_fig5.png        : 그림 5 (DROID 성공률·처리율 차트, 원문 8쪽)
  paper_fig7a.png       : 그림 7a (하드웨어별 지연 차트, 원문 9쪽). 오른쪽 표 7b의 장면 사진은 자른다.

쓰지 않은 그림: 그림 2(관측 사진 포함), 그림 6(실로봇 사진), 그림 7b(사진 열), 부록 그림 C–F (과제 사진 포함 또는 이 분석에서 쓰지 않음).
라이선스: arXiv abs 페이지 CC BY 4.0, 원문 PDF에 저작권 표시 없음(분석 문서 참조). 공개본에서 출처와 라이선스를 적는다.
실행: python3 report/crop_figures.py   (pymupdf, Pillow; paper/paper.pdf 는 arXiv에서 받아 둘 것)
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
JOBS = {
    "paper_fig1_method.png": (1, (222, 70, 420, 170)),   # 0-based 쪽 번호, 자를 영역(pt)
    "paper_fig3.png": (3, (106, 66, 504, 156)),
    "paper_fig4.png": (6, (104, 372, 512, 470)),
    "paper_fig5.png": (7, (104, 68, 508, 178)),
    "paper_fig7a.png": (8, (100, 62, 302, 145)),
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

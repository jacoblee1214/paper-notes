"""논문 그림 세 장을 원문 PDF에서 잘라 report/figures/에 저장한다.

  paper_fig3.png : 그림 3 (공간·인스턴스 이동에서 ID 대응 과제와의 성공률 비교, 원문 7쪽)
  paper_fig4.png : 그림 4 (시연 수 10/50/100 규모 확장, 원문 8쪽)
  paper_fig5.png : 그림 5 (BiMESA 팔 교체·핸드오프 방향·관절체 비교, 원문 9쪽)

고른 이유: 저작자 본인이 만든 막대·선 그래프이고 제3자 사진, 화면 캡처, 데이터셋 프레임, 실로봇 사진이 아니다.
라이선스: arXiv abs 페이지 CC BY 4.0, 원문 PDF에 저작권 표시 없음. 출처와 라이선스를 적어 쓴다.
캡션 글자는 포함하지 않는다.

실행: python3 report/crop_figures.py   (pymupdf, Pillow)
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
JOBS = {
    "paper_fig3.png": (6, (108, 68, 505, 160)),   # 0-based 쪽 번호, 자를 영역(pt)
    "paper_fig4.png": (7, (116, 68, 497, 133)),
    "paper_fig5.png": (8, (104, 68, 508, 163)),
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

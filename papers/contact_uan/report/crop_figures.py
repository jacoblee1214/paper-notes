"""논문 그림 세 장을 원문 PDF에서 잘라 report/figures/에 저장한다.

  paper_fig5.png : 그림 5 (접촉 높이 RMSE, 하드웨어 24-step 관절 위치 MSE, sim-to-sim RMSE; 원문 5쪽)
  paper_fig7.png : 그림 7 (달성 속도, 토크 감소율, 측정 힘; 원문 7쪽)
  paper_fig8.png : 그림 8 (리셋 절제 생존율, 속도 추종, 토크 사용량; 원문 8쪽)

고른 이유: 모두 저작자가 그린 막대·선 그래프(벡터)이고 사진, 사람, 시뮬레이션 화면, 상표 이미지가 없다.
쪽에 래스터 이미지가 하나도 없음을 pdfimages로 확인했다. 그림 1, 2, 4, 9, 10, 15는 사진이나 시뮬레이션 화면을
담고 있어 쓰지 않는다. 그림 3은 발 모양 삽화가 섞여 있어 쓰지 않는다.
라이선스: arXiv abs 페이지 CC BY 4.0, 원문 PDF에 저작권 표시 없음. 출처와 라이선스를 캡션에 적는다.
잘라낸 영역은 쪽 위쪽 여백 아래부터 "Figure N:" 캡션 위까지이며 캡션 글자는 포함하지 않는다.

실행: python3 report/crop_figures.py   (pymupdf, Pillow, paper/paper.pdf 필요)
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
JOBS = {
    "paper_fig5.png": (4, "Figure 5:"),   # 0-based 쪽 번호, 캡션 첫 글자
    "paper_fig7.png": (6, "Figure 7:"),
    "paper_fig8.png": (7, "Figure 8:"),
}
TOP = 50.0       # pt, 쪽 위 여백 아래
LEFT, RIGHT = 100.0, 520.0


def trim(img: Image.Image) -> Image.Image:
    bg = Image.new("RGB", img.size, (255, 255, 255))
    box = ImageChops.difference(img.convert("RGB"), bg).getbbox()
    if not box:
        return img
    pad = 8
    return img.crop((max(box[0] - pad, 0), max(box[1] - pad, 0),
                     min(box[2] + pad, img.size[0]), min(box[3] + pad, img.size[1])))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PDF)
    for name, (page_idx, cap) in JOBS.items():
        page = doc[page_idx]
        cap_y = page.search_for(cap)[0].y0
        clip = pymupdf.Rect(LEFT, TOP, RIGHT, cap_y - 3)
        pix = page.get_pixmap(dpi=300, clip=clip)
        tmp = OUT / (name + ".tmp.png")
        pix.save(tmp)
        img = trim(Image.open(tmp))
        img.save(OUT / name)
        tmp.unlink()
        print("wrote", OUT / name, img.size)


if __name__ == "__main__":
    main()

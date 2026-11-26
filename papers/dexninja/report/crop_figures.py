"""논문 그림 4의 일부(패널 b, c의 도식)만 원문 PDF에서 잘라 report/figures/paper_fig4_bc.png로 저장한다.

고른 이유: 패널 (b), (c)는 저작자가 그린 도식(다양체 위의 국소 근사)이다. 같은 그림의 패널 (a), (f)에는
실제 사진(관측 영상, 딸기)이 있어 제외한다. 그림 1·3·5와 부록 그림은 실로봇 사진, 시뮬레이션 화면, 생성 영상이
들어 있어 쓰지 않는다.
라이선스: arXiv abs 페이지 CC BY 4.0, 원문 PDF에 저작권 표시 없음. 공개본에도 넣되 출처와 라이선스를 적는다.
잘라낸 영역은 원문 6쪽을 400 dpi로 렌더링한 뒤 패널 b, c의 좌표(페이지 pt)로 자른다. 캡션 글자는 포함하지 않는다.

실행: python3 report/crop_figures.py   (pymupdf, Pillow)
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures" / "paper_fig4_bc.png"
# 원문 6쪽(0부터 5), 패널 b와 c가 들어 있는 영역(pt). 화살표와 패널 (a), (f)는 제외한다.
PAGE = 5
CLIP = pymupdf.Rect(310.5, 231.3, 426.0, 279.5)


def main():
    doc = pymupdf.open(PDF)
    pix = doc[PAGE].get_pixmap(matrix=pymupdf.Matrix(400 / 72, 400 / 72), clip=CLIP)
    pix.save(OUT)
    im = Image.open(OUT).convert("RGB")
    # 패널 (c) 오른쪽의 다음 화살표 끝이 걸리면 그 부분만 흰색으로 덮는다(도식 내용은 건드리지 않는다)
    w, h = im.size
    ImageDraw.Draw(im).rectangle([w - 40, 100, w, 160], fill="white")
    im.save(OUT)
    print("wrote", OUT, im.size)


if __name__ == "__main__":
    main()

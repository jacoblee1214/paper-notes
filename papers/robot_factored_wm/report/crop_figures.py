"""원문 PDF에서 그림 1의 오른쪽 구조도(VAE 인코더 - 패치 임베딩 - DiT - VAE 디코더)만 잘라 report/figures/paper_fig1_arch.png로 저장한다.

고른 이유: 저작자가 그린 블록 다이어그램이다. 그림 1의 왼쪽(로봇 사진, 장면 프레임, 렌더 프레임)과 오른쪽 끝의 생성 영상 프레임,
그림 2~11(로봇·시뮬레이터 프레임)은 사진·프레임이라 뺀다.
라이선스: arXiv abs 페이지 license 링크가 creativecommons.org/licenses/by/4.0/ (확인 2026-11-26). 원문 PDF에 ©·All rights reserved 표시 없음.
쪽 좌표(pt, 0-기준 쪽 번호 2 = 3쪽). 출처 표기: © 저자(Kim, Kim, Cha, Joo), CC BY 4.0, arXiv:2607.22535 그림 1 일부.
실행: python3 report/crop_figures.py   (pymupdf, Pillow)
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
JOBS = {"paper_fig1_arch.png": (2, (334, 78, 443, 226))}


def trim(img):
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
        trim(Image.open(tmp)).save(OUT / name)
        tmp.unlink()
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()

# ATOM-Bench (arXiv:2606.16826, CoRL 2026 스포트라이트) 분석 노트

*ATOM-Bench: A Real-World Benchmark for Atomic Skills and Compositional Generalization in Manipulation Policies* (arXiv v1, 2026-06-15)를 표 재계산과 통계 검정으로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.16826에서 받을 수 있습니다). 논문 그림은 차트 세 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **5개 VLA 미세조정과 실로봇 평가는 이 노트북에서 불가능합니다.** GPU와 로봇이 필요합니다.
- **지표 계산과 통계 재분석은 구현했습니다.** `analysis/recompute.py`(수 초, 표준 라이브러리).
- **원자와 조합 구조를 재는 작은 시뮬레이션은 가능합니다.** 공식 코드를 확인하지 못해 코드 대조는 하지 못했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 지표, 실험, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V12, 설계 검사 K1 |
| `analysis/04_claim_verification.md` | 주장 C1–C18, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/crop_figures.py` | 논문 그림 세 장 자르기 (원문 PDF 필요) |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 세 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install matplotlib pymupdf pillow
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 문구를 찾지 못했습니다.
- 논문 그림 세 장(차트)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2606.16826"). 사진이 든 그림은 넣지 않았습니다.
- 프로젝트 페이지: https://flageval-baai.github.io/AtomBenchPage/ . 공식 코드는 2026-11-26 기준 확인하지 못했습니다. 시연 데이터는 Hugging Face(AtomBench 조직)에 있고 받지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서, 스크립트는 이 노트의 작성자가 만들었습니다.

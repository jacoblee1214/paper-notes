# How Firm Should a Grasp Be? (arXiv:2610.11221, CoRL 2026 포스터) 분석 노트

*How Firm Should a Grasp Be?* (Matthew Beveridge, Shree K. Nayar, arXiv v1, 2026-10-08)를 표 재계산, 그림 판독, Hertz 접촉 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2610.11221에서 받을 수 있습니다). 논문 그림은 세 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 결과의 재현은 어렵습니다.** 공식 코드를 찾지 못했고, 변형체 접촉 시뮬레이션(GPU)과 UR5 + GelSight 실물이 필요합니다.
- **식 1·2·5의 최소 힘 풀이와 그림 2 민감도는 CPU로 가능합니다.** `analysis/toy_hertz_reduction.py`(numpy, scipy, 수 초).
- **합성 접촉 데이터와 작은 순환망으로 사후 추정을 축소 구현하는 일은 가능해 보입니다.** 이 노트에서는 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 세 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치 |
| `analysis/02_prior_work.md` | 선행 계보, 관련 저장소 표 |
| `analysis/03_analysis.md` | 표·그림 수치 재계산, 일관성 검사, 토이 |
| `analysis/04_claim_verification.md` | 주장별 근거 정리와 구현 가능성 |
| `analysis/recompute.py` | 표 1, 표 S1, 그림 6e 재계산 (numpy, scipy, 수 초) |
| `analysis/toy_hertz_reduction.py` | 토이: 식 1–2로 힘 감소율을 풀고 그림 2와 비교 |
| `analysis/read_fig6d.py` | 그림 6d 막대 수와 높이를 픽셀에서 읽음 (원문 PDF 필요) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 세 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib pillow
python3 analysis/recompute.py
python3 analysis/toy_hertz_reduction.py
python3 analysis/plot_figures.py
```

`analysis/read_fig6d.py`는 `paper/paper.pdf`(arXiv:2610.11221v1)와 `pdftoppm`이 있어야 돕니다.

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 세 장(그림 2, 6c, 6e)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2610.11221"). 사진, 시뮬 장면이 있는 그림은 넣지 않았습니다.
- 공식 코드는 찾지 못했습니다. 프로젝트 페이지는 논문에 링크가 있습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 데이터셋과 가중치는 받지 않았습니다.

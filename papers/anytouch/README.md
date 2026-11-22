# AnyTouch (arXiv:2502.12191, ICLR 2025) 분석 노트

*AnyTouch: Learning Unified Static-Dynamic Representation across Multiple Visuo-tactile Sensors* (arXiv v3, 2025-04-01)을 표 재계산, 일관성 검사, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2502.12191에서 받을 수 있습니다). 논문 그림은 한 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **사전학습 재현은 이 맥에서 불가능합니다.** 논문 설정은 A800 4장 기준이고, 공식 코드는 CUDA 전용입니다.
- **체크포인트 추론과 선형 탐침 재평가는 조건부입니다.** 데이터 다운로드와 코드 수정이 먼저 필요합니다.
- **재계산과 정적 대조는 완료했습니다.** `analysis/recompute.py`(numpy, scipy, 수 초)와 `analysis/plot_figures.py`로 결과와 그래프를 다시 만들 수 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 한 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, 공식 저장소 표 |
| `analysis/03_analysis.md` | 수치 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조, 구현 가능성, 라이선스 |
| `analysis/recompute.py` | 표 값 재계산과 검정 (numpy, scipy) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |
| `report/figures/` | 그래프 두 장(우리가 그림)과 논문 그림 한 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 논문 그림 한 장(그림 4, t-SNE)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2502.12191"). 개요도(그림 2)와 실로봇·데이터셋 사진이 들어 있는 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/GeWu-Lab/AnyTouch`에 있으며 MIT입니다. 이 저장소에는 코드를 넣지 않았고, 설치하거나 실행하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트와 분석 문서는 이 노트의 작성자가 만들었습니다.
- 체크포인트와 데이터셋은 받지 않았습니다.

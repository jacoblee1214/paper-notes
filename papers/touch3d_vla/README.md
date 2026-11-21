# 3DTouch (arXiv:2611.23961) 분석 노트

*3DTouch: A Tactile-aware VLA Model with 3D Geometric Relations for Dexterous Manipulation* (arXiv v1, 2026-11-18)의 표 수치를 다시 계산하고 주장을 검증한 자료입니다. 학회 표기는 없습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.23961에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- **전체 재현은 이 맥에서 불가능합니다.** 실로봇 평가, 공식 코드와 데이터 부재, GPU 전제 학습이 이유입니다.
- **관계 토큰 계산과 촉각 좌표 변환은 CPU로 가능합니다.** 이 자료에는 토이를 넣지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 코드 확인 결과 |
| `analysis/03_analysis.md` | 표 재계산, Wilson·Newcombe·Fisher·Holm 검정, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C20, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (수 초, numpy·scipy) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스(`nonexclusive-distrib/1.0`)로 공개되어 있습니다. CC BY가 아니므로 논문 그림을 넣지 않았습니다.
- 공식 코드와 프로젝트 페이지는 찾지 못했습니다. 따라서 코드 대조는 하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 검정, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터는 받지 않았습니다.

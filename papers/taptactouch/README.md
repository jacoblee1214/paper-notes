# TapTacTouch: Shape Completion from Sparse Tactile Contacts with Geometric Priors (arXiv:2611.24207) 분석 노트

*TapTacTouch* (arXiv v1, 2026-11-18)를 표 재계산과 주장 대조로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 세 장은 우리가 만든 것입니다. 학회 채택 기록과 공식 코드는 확인되지 않았습니다.

## 구현 가능성 (요약)

- **구현은 가능합니다.** 식 (3)–(14)의 GPIS 파이프라인은 신경망이 없어서 numpy와 scipy로 CPU에서 돌릴 수 있습니다. 노력은 2–4주로 추정합니다(측정 전).
- **논문 수치와 실측 재현은 어렵습니다.** 로봇 장비, 대상 GT 메쉬, 대상별 등록 변환 방법이 공개되지 않았습니다.
- **지표의 입력 조건:** 논문의 비교 지표는 GT 표면에 투영한 접촉에 0.5 mm 잡음을 더한 입력으로 계산되었습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법 (식 번호), 결과, 주장과 증거의 틈, 빈칸 |
| `analysis/02_prior_work.md` | 선행 계보, 학회 확인, 코드 탐색 결과 |
| `analysis/03_analysis.md` | 평균 재계산, 대상 짝비교와 검정, 표 사이 일관성, 기하 상수 |
| `analysis/04_claim_verification.md` | 주장 판정, 빈칸 목록, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv:2611.24207 v1. abs 페이지의 license 링크는 non-exclusive distribution 1.0입니다(2026-11-22 확인). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드와 프로젝트 페이지는 찾지 못했습니다. 저장소를 복제하거나 실행하지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮긴 것입니다. 그래프, 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.
- 데이터셋과 모델 가중치는 받지 않았습니다.

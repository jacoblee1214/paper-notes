# 3D-ViTac (arXiv:2410.24091, CoRL 2024) 분석 노트

*3D-ViTac: Learning Fine-Grained Manipulation with Visuo-Tactile Sensing* (arXiv v2, 2025-01-06)을 표 재계산, 통계 검정, 공개 저장소 정적 읽기로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2410.24091에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 모든 그래프는 우리가 그렸습니다.

## 구현 가능성 (요약)

- **부분 구현 가능, 결과 재현 불가.** 촉각 점 표현과 정책 골격은 맥에서 만들 수 있지만, 실로봇 표 재현에는 양팔 로봇, 촉각 패드 4개, 카메라 3대, GPU가 필요합니다.
- **정책 학습 코드는 공개되어 있지 않습니다.** 공개된 것은 센서 펌웨어와 판독 소프트웨어입니다. 라이선스가 세 가지로 적혀 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 검증, 공개 하드웨어 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, statsmodels, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy statsmodels matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv abs 페이지에서 CC BY 4.0으로 표시되어 있습니다. 원문 PDF에서 저작권 고지는 찾지 못했습니다.
- 학회 표기: CoRL 2024, PMLR v270 공식 proceedings에서 확인했습니다.
- 공개 저장소의 코드는 이 저장소에 넣지 않았고, 실행하거나 설치하지 않았습니다. 하드웨어 저장소의 라이선스 표기는 서로 다릅니다(`analysis/04_claim_verification.md` §1).
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

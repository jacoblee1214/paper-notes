# EgoScale (arXiv:2602.16710, CoRL 2026 구두) 분석 노트

*EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data* (arXiv v1, 2026-02-18)을 그림 라벨 재계산, 스케일링 법칙 재적합, 토이 시뮬레이션으로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다. PDF에 "© 2026 NVIDIA. All rights reserved." 표기가 있어서입니다.

## 구현 가능성 (요약)

- **이 맥에서 논문 방법의 전체 재현은 불가능합니다.** 256장 GPU 규모 사전학습, 비공개 데이터, 로봇 평가가 필요합니다.
- **부분 구현은 조건부로 가능합니다.** 손목 상대 행동, 소형 흐름 정책, 리타게팅 최적화는 가능하고, 소형 데이터 규모 곡선은 데이터 준비 뒤에 가능합니다.
- 공식 코드는 확인되지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계, 본문 불일치 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 표 |
| `analysis/03_analysis.md` | 그림 4 성공률 재계산, 스케일링 재적합, 토이 결과, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 검증표, 본문과 그림의 불일치, 구현 가능성, 저자에게 물을 질문 |
| `analysis/table_check.py` | 성공률 라벨 재계산, Wilson과 Fisher 검정, 격자 검사 (numpy, scipy) |
| `analysis/scaling_refit.py` | 그림 5 판독값 재적합, 대안 모델, 곡률 검정, 외삽 (numpy, scipy) |
| `analysis/toy_scaling_fit.py` | 점 수와 잡음에 따른 곡률 검정과 외삽 토이 (numpy, 수십 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/scaling_refit.py
python3 analysis/toy_scaling_fit.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다. 다만 PDF에는 "© 2026 NVIDIA. All rights reserved." 표기가 있어서 논문 그림은 넣지 않았습니다. 원문은 arXiv:2602.16710에서 받을 수 있습니다.
- 그림 라벨과 판독 마커 위치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 데이터셋과 가중치는 받지 않았습니다.

# OmniH2O (arXiv:2406.08858, CoRL 2024) 분석 노트

*OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning* (arXiv v1, 2024-06-13)을 표 재계산, 통계 검정, 공식 코드의 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2406.08858에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- **논문 수치 재현은 이 맥에서 불가능합니다.** 시뮬 학습에는 Isaac Gym과 CUDA가, 실로봇에는 Unitree H1이 필요합니다.
- **표 재계산은 완료했습니다.** 시뮬 개선과 DAgger 효과는 확실하고, 실로봇 표본(20 시퀀스)과 선속도 제거의 이점은 판단하기 어렵습니다.
- **공식 코드와 논문 설명이 다릅니다.** 최대 발 높이 보상의 방향이 반대이고, 추적 보상 계수도 다릅니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그래프 세 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 학회본과의 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소와 페이지 표 |
| `analysis/03_analysis.md` | 표 재계산, 검정, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/figures/` | 그래프 세 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

`plot_figures.py`는 `report/figures/`에 그림을 쓰므로, 이 폴더에서는 `report/figures/`를 만들어 두고 실행합니다.

## 출처와 라이선스

- 논문은 arXiv에 비배타적 배포 허락(nonexclusive-distrib 1.0)으로 공개되어 있습니다. CC BY가 아니므로 논문 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/LeCAR-Lab/human2humanoid`에 있습니다. README는 CC BY-NC 4.0이라고 적습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

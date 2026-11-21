# TMA (arXiv:2611.18976) 분석 노트

*TMA: Sim-to-Real Forceful Humanoid Manipulation via Tactile-Driven Motor Adaptation* (arXiv v1, 2026-11-13)을 표 재계산과 통계 검정으로 읽은 자료입니다. 학회 표기는 없습니다(arXiv). 공식 코드는 찾지 못했습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.18976에서 받을 수 있습니다). 그림은 우리가 그린 그래프 두 장뿐입니다.

## 구현 가능성 (요약)

- **논문 결과 재현은 이 맥에서 불가능합니다.** 시뮬레이션 학습에 NVIDIA GPU가 필요하고, 실로봇 시험에는 휴머노이드와 촉각 손이 필요합니다.
- **알고리즘 골격은 구현할 수 있습니다.** 촉각 힘의 원뿔 좌표 변환과 이력 기반 잠재 회귀는 작은 규모로 검사할 수 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그래프 두 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 원고 내부 점검 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, 공식 코드 상태 표 |
| `analysis/03_analysis.md` | 실로봇과 시뮬 수치 재계산, 검정, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C12, 구현 가능성 판정 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv non-exclusive distribution license로 공개되어 있습니다(arXiv abs 페이지 확인). CC BY 4.0이 아니므로 논문 그림은 싣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 프로젝트 페이지 저장소(github.com/humanoid-tma/humanoid-tma.github.io)에는 웹사이트 자산만 있고 라이선스가 없습니다. 이 저장소에는 그 내용을 넣지 않았습니다.

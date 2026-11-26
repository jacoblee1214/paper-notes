# Cloth Simulator-in-the-Loop (arXiv:2606.24552, CoRL 2026 스포트라이트) 분석 노트

*Enabling Robust Cloth Manipulation via Inference-Time Simulator-in-the-Loop Refinement* (arXiv v2, 2026-09-24)를 표 재계산, 통계 검정, 작은 numpy 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.24552에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 그래프는 우리가 논문 표 값으로 다시 그린 것입니다.

## 구현 가능성 (요약)

- **논문 시스템의 전체 재현은 이 맥에서 불가능합니다.** 시뮬레이터 FLASH의 코드가 공개되지 않았고(2026-11-26에 확인하지 못함), 실로봇(AIRBOT Play, ZED 2i)과 RTX급 GPU가 필요합니다.
- **표 재계산, 통계 검정, 정제 크기 토이는 CPU로 끝냈습니다.** `analysis/recompute.py`와 `analysis/toy_mppi_refine.py`(각 수 초 이내, numpy·scipy).
- **소형 재구현(MuJoCo 천 + 사전 정책 증류 + 사전 안내 MPPI)은 가능하며 3–5주(추정, 측정 전)입니다.** 이번 분석에서는 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 판본 차이 |
| `analysis/02_prior_work.md` | 선행 계보, GitHub 표 |
| `analysis/03_analysis.md` | 재계산 V1–V13, 일관성 검사, 토이 T1 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C23, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy) |
| `analysis/toy_mppi_refine.py` | 토이: MPPI 한 스텝 보정 크기 (numpy) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_mppi_refine.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 비독점 배포 라이선스(arxiv.org/licenses/nonexclusive-distrib/1.0)로 올라 있어 논문 그림은 넣지 않았습니다.
- 공식 코드 저장소는 2026-11-26에 확인하지 못했습니다. 이 저장소에는 외부 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

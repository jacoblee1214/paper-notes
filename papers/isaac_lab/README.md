# Isaac Lab (arXiv:2511.04831) 분석 노트

*Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning* (NVIDIA, arXiv v1, 2025-11-06, 기술 보고서)을 처리량 그림 판독, 재계산, 공식 코드(v2.3.0) 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2511.04831에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **이 맥(Apple Silicon, CUDA 없음)에서 Isaac Lab은 실행할 수 없습니다.** 설치 요건은 Ubuntu 22.04 또는 Windows 11과 NVIDIA GPU입니다.
- **처리량 재현은 Linux GPU 서버가 필요합니다.** 설치와 벤치 실행은 수 일 규모로 추정합니다(측정 전).
- **판독과 재계산, 코드 대조는 가능합니다.** `analysis/recompute.py`(numpy, 수 초)와 `analysis/plot_figures.py`로 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 설계, 정량 주장 Q1–Q15, 한계 |
| `analysis/02_prior_work.md` | 계보, 비교 시뮬레이터, GitHub 표, 라이선스 |
| `analysis/03_analysis.md` | 그림 판독 기반 재계산(V1–V9), 일관성 검사(C1–C10) |
| `analysis/04_code_verification.md` | 공식 코드 대조(D1–D12), 일치 항목, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 그림 판독값과 재계산 결과 |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 표기되어 있습니다. 그러나 원문 PDF에 "© 2025 NVIDIA. All rights reserved."가 있어 두 표시가 충돌합니다. 그래서 논문 그림은 넣지 않았습니다.
- 수치와 표현은 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 코드는 `github.com/isaac-sim/IsaacLab`에 있습니다(LICENSE는 BSD-3-Clause). 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다. 코드는 정적으로만 읽었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

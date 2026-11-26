# SlipSense (arXiv:2609.15910, CoRL 2026 포스터) 분석 노트

*SlipSense: Multimodal Tactile Learning for Low-Latency and Generalized Slip Detection* (arXiv v1, 2026-09-14)을 표 재계산, 일관성 점검, 작은 CPU 토이로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2609.15910에서 받을 수 있습니다). 공식 코드와 프로젝트 페이지는 찾지 못했습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 어렵습니다.** TacV5 센서, 140만 프레임 데이터, 시험대·그리퍼·로봇 손, 코드, 체크포인트가 공개되어 있지 않습니다.
- **구조 재구현과 합성 데이터 실험은 노트북에서 가능합니다.** 가정한 구조(약 5.9M 매개변수)의 한 번 순전파는 Apple M4 Pro의 GPU(MPS)에서 3.0–3.2 ms, CPU에서 16–26 ms였습니다(`analysis/toy_latency_budget.py`, 논문 구조의 시간이 아님).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 표 |
| `analysis/03_analysis.md` | 재계산 V1–V9, 일관성 점검 K1–K14, 토이 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C19, 권리, 구현 가능성 |
| `analysis/recompute.py` | 표 수치 재계산과 가능성 검사 (numpy, scipy, 약 80초) |
| `analysis/toy_lse_invariance.py` | 토이: 로그합지수 방향 불변성 (numpy, 1초) |
| `analysis/toy_latency_budget.py` | 토이: 추론 시간 예산 (torch, 1분 안팎, 가정한 구조) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_lse_invariance.py
python3 analysis/toy_latency_budget.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문의 arXiv 라이선스는 non-exclusive distribution(CC 아님)이라 논문 그림은 넣지 않았습니다. 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다.
- 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 코드, 가중치, 데이터셋은 받지 않았고 실행하지 않았습니다.

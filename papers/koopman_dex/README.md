# Going with the Flow (arXiv:2602.07413, CoRL 2026 포스터) 분석 노트

*Going with the Flow: Koopman Behavioral Models as Pseudo Planners for Visuo-Motor Dexterity* (arXiv v4, 2026-11-20)를 표 재계산, 공식 코드 정적 대조, 롤아웃 시간 토이로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 비독점 배포 라이선스). 원문은 arXiv:2602.07413에서 받을 수 있습니다.

## 구현 가능성 (요약)

- **실로봇과 시뮬 성공률 재현은 이 맥(Apple M4 Pro)에서 어렵습니다.** Franka + LEAP 로봇, SAPIEN 2.2.1, 레거시 mujoco_py가 필요합니다. 기준선 코드는 저장소에 없습니다.
- **공식 데이터로 K-UBM만 학습하는 것은 조건부입니다.** 모델이 작지만(약 29–43만 파라미터) 데이터를 받지 않았고 학습 시간은 측정하지 않았습니다.
- **롤아웃 시간 토이는 완료했습니다.** `analysis/toy_rollout_timing.py`(수 초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 판본 차이, 방법, 실험 수치, 근거의 간격 |
| `analysis/02_prior_work.md` | 계보, 구성 요소 출처, GitHub 표 |
| `analysis/03_analysis.md` | 재계산 V1–V14, 토이, 주장 판정 C1–C10 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D16, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_rollout_timing.py` | 토이: z ← Kz 한 스텝 시간 (numpy, torch, 수 초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_rollout_timing.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 비독점 배포 라이선스(arxiv.org/licenses/nonexclusive-distrib/1.0)로 올라와 있습니다. 그래서 논문 그림은 넣지 않았고, 그래프는 표 값과 그림 판독 값(오차 약 1%p, 문서에 표시)으로 다시 그렸습니다.
- 공식 코드는 `github.com/GT-STAR-Lab/K-UBM`에 있고 LICENSE 파일이 없습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

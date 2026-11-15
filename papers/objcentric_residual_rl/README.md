# Object-Centric Residual RL (arXiv:2606.18953, CoRL 2026) 분석 노트

*Object-Centric Residual RL for Zero-Shot Sim-to-Real VLA Enhancement* (arXiv v1, 2026-06-17 제출)을 표 재계산, 통계 검정, 작은 numpy 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다. 공식 코드는 찾지 못했습니다.

## 구현 가능성 (요약)

- **실로봇 평가와 자기 개선 SFT는 이 맥에서 불가입니다.** FR3, RGB-D, GPU 학습이 필요합니다.
- **시뮬레이션 잔차 골격은 구현할 수 있습니다(조건부).** MuJoCo 5과제와 TD3 2층 MLP는 가능합니다. GR00T-N1.5 기반 행동의 MPS 추론은 확인하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 실험의 틈, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 표 |
| `analysis/03_analysis.md` | 실로봇 재계산, 절제 검정, 토이 C, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 C1–C19 판정, 빠진 정보, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_pose_residual.py` | 토이 C: 포즈 오차 축별 성공률과 편향 절벽 (numpy, 약 2.5분) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_pose_residual.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다. 논문 그림과 원문 PDF는 이 저장소에 넣지 않았습니다. 원문은 arXiv:2606.18953에서 받을 수 있습니다.
- 공식 코드는 찾지 못했습니다. 기반 VLA(GR00T-N1.5, π0.5)는 각각 NVIDIA와 Physical Intelligence의 공개 저장소에 있습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 데이터셋과 체크포인트는 받지 않았습니다.

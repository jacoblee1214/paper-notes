# 움직이는 좁은 틈 쿼드로터 (arXiv:2610.37557, CoRL 2026 포스터) 분석 노트

*Learning Visuomotor Quadrotor Flight Through Dynamic Narrow Gaps via Differentiable Simulation* (arXiv v1, 2026-10-27)을 수치 재계산, 공식 코드 정적 읽기, numpy 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **이 맥(CUDA 없음)에서 논문 결과 재현은 불가합니다.** 공식 학습은 JAX GPU 가속(Linux)과 CUDA GPU를 전제합니다. 하이-피델리티 시뮬(81%)과 실물(80%)은 공개되지 않았습니다.
- **상태 기반 틈 과제의 PyTorch 재구현은 가능합니다**(추정 2–3주). 시각 특징과 PPO 비교까지 5–7주(추정)입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 결과, 한계, 본문 불일치, 질문 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 표 |
| `analysis/03_analysis.md` | 수치 재계산(Wilson, Fisher, Table 3 산술과 시간 정의), 토이 결과 |
| `analysis/04_code_verification.md` | 논문 대 공식 코드 대조(D1–D31), 주장 검증, 구현 가능성 |
| `analysis/recompute.py` | 논문 수치 재계산과 공개 로그 대조 (scipy, 수 초) |
| `analysis/toy_gap_pathwise.py` | 토이 G: 지평별 분산, 온도, 지시함수 대조군 (numpy, 약 80초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py      # 공개 로그(repo/runs)가 없으면 로그 부분은 건너뜀
python3 analysis/toy_gap_pathwise.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포(nonexclusive-distrib)입니다. CC 라이선스가 아니므로 논문 그림과 원문 PDF는 넣지 않았습니다. 원문은 arXiv:2610.37557에서 받을 수 있습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 노트의 작성자가 만들었습니다.
- 공식 코드(uzh-rpg/agile_narrow_gap_diffsim, GPL-3.0)는 정적으로만 읽었고 재배포하지 않았습니다. 가중치와 데이터는 받지 않았습니다.
- 토이 G는 논문 코드가 아니라 같은 구조를 옮긴 모형입니다. 결과를 쿼드로터 실험으로 일반화하지 않습니다.

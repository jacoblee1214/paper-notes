# Flow Policy Gradients for Robot Control (arXiv:2602.02481, CoRL 2026) 분석 노트

*Flow Policy Gradients for Robot Control* (CoRL 2026 구두 발표, arXiv v2 2026-10-13)을 보고 수치 재계산, 내부 일관성 검사, 공식 저장소 정적 대조, 1차원 토이 두 개로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **알고리즘:** PyTorch로 350–500줄, 3–5일 정도(추정)로 구현할 수 있습니다. 샘플별 비율, ASPO, 클램프, 제로 샘플링이 대상입니다.
- **소형 과제 비교:** MuJoCo 같은 소형 과제에서 FPO++와 가우시안 PPO를 비교하는 것은 패키지 설치 후 CPU에서 가능합니다(조건부).
- **논문 규모 재현:** 보행(IsaacLab, CUDA GPU), 모션 트래킹과 sim-to-real(데이터, 실물 로봇), 조작 미세조정(베이스 정책, GPU 수 시간)은 이 맥에서 불가능합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 부록 세부, v1과 v2 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, 공식 저장소 표 |
| `analysis/03_analysis.md` | 보고 수치 재계산, 일관성 검사, 토이 결과 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 차이(D1–D21), 주장 검증(C1–C19), 구현 가능성, 라이선스 |
| `analysis/table_check.py` | 보고 수치 재계산과 이항 구간 (scipy, 수 초) |
| `analysis/toy_euler_steps.py` | 적분 단계별 오차 토이 (numpy, 수 초) |
| `analysis/toy_ratio_vs_likelihood.py` | 대리 비율과 가능도 비율 토이 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 -I analysis/table_check.py
python3 -I analysis/toy_euler_steps.py
python3 -I analysis/toy_ratio_vs_likelihood.py
python3 analysis/plot_figures.py
```

공식 저장소는 받아서 실행하지 않았습니다(정적 읽기만). 보려면 `git clone --depth 1 https://github.com/amazon-far/fpo-control.git`로 받습니다(분석 기준 커밋 `8f460e4`).

## 출처와 라이선스

- 논문은 arXiv에서 CC BY 4.0입니다(abs 페이지 확인, 2026-11-15). 논문 그림과 원문 PDF는 이 저장소에 넣지 않았습니다. 원문은 arXiv:2602.02481에서 받을 수 있습니다.
- 논문의 표 수치는 인용과 재계산을 위해 옮겼습니다. 문장과 그림은 옮기지 않았습니다.
- 공식 저장소에는 LICENSE 파일이 없습니다. 코드는 복사하지 않았고, 읽은 내용만 요약했습니다.
- 그래프와 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다. 토이의 파라미터는 설계 가정이며 논문 값에 맞추지 않았습니다.

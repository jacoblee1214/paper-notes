# REGRIND: Minimalist Retargeting-Guided RL for Dexterous Manipulation (arXiv:2607.11874, CoRL 2026) 분석 노트

*A Minimalist Retargeting-Guided Reinforcement Learning Recipe for Dexterous Manipulation* (arXiv v1, 2026-07-13)을 표 재계산, 공식 코드 정적 대조, 작은 numpy 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **공식 재현은 이 맥에서 불가합니다.** 학습은 Isaac Sim 5.1과 CUDA GPU가 필요하고, 실기 재현에는 두 종류의 로봇 손과 모션 캡처가 필요합니다.
- **리타게팅 목적과 잔차 구조는 구현할 수 있습니다(조건부).** scipy와 솔버로 리타게팅 목적을 옮길 수 있고, 손 접촉이 있는 MuJoCo 축소판은 처리량을 재야 판단할 수 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계, 주장과 실험의 틈 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 표 |
| `analysis/03_analysis.md` | 표 재계산(Wilson, Fisher, Welch), 일관성 검사, 토이 결과 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 불일치 목록, 구현 가능성 |
| `analysis/recompute.py` | 표 1–3 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_retarget_prior.py` | 토이: 기준 + 잔차 RL 대 처음부터 RL, 리타게팅 오차 민감도 (numpy, 단일 스레드, 약 100초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_retarget_prior.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다. 논문 그림과 원문 PDF는 이 저장소에 넣지 않았습니다. 원문은 arXiv:2607.11874에서 받을 수 있습니다.
- 공식 코드(https://github.com/yunhaif/regrind)는 MIT 라이선스입니다. 이 저장소에서는 코드를 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 데이터셋과 체크포인트는 받지 않았습니다.

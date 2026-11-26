# Grounding Generative Policies in Physics: Optimization-Guided Diffusion for Robot Control (arXiv:2606.24208, CoRL 2026) 분석 노트

*Grounding Generative Policies in Physics* (arXiv v1, 2026-06-23)를 표 재계산, 주장 검증, 작은 토이로 분석한 자료입니다. 공식 코드와 프로젝트 페이지는 2026-11-26 기준 확인하지 못했습니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 파지 평가(cuRobo, 파지 시뮬레이터)와 시각운동(DiT 학습, 시뮬레이션)이 CUDA와 시뮬레이터를 요구하고 실로봇이 필요합니다.
- **핵심 아이디어(식 3-4)의 소형 재구현은 가능합니다.** 2차원 토이는 끝냈고, 저차원 학습형 확산 모델, 대리망, 제어기 롤아웃은 각 3-5일 추정입니다(측정 아님).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 논문에 없는 값, 버전 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(V1-V14), 검정, 2차원 토이 |
| `analysis/04_claim_verification.md` | 주장 검증 C1-C18, 구현 가능성, 저자 질문 |
| `analysis/table_check.py` | 표 일관성 검사와 검정 (numpy, scipy, 수 초) |
| `analysis/toy_guidance.py` | 2차원 혼합 토이 (torch, CPU, 약 1분) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/table_check.py
python3 analysis/toy_guidance.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-26). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그림 6의 값은 정수 라벨을 읽은 것(그림 판독 값)입니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

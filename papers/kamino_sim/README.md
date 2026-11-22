# Kamino (arXiv:2603.16536, CoRL 2026 Spotlight) 분석 노트

*Kamino: GPU-based Simulation of Rigid Multi-Body Physics with Kinematic Loops* (arXiv v2, 2026-10-20)를 표 재계산, 공식 코드 정적 대조, 작은 CPU 토이로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2603.16536에서 받을 수 있고, 논문은 arXiv non-exclusive 라이선스입니다).

## 구현 가능성 (요약)

- **논문 수치(처리량, RL 학습, 실물)의 재현은 이 맥에서 불가입니다.** GPU와 Isaac Lab, 비공개 로봇 모델과 학습 코드가 필요합니다.
- **알고리즘의 축소판은 가능합니다.** `analysis/toy_padmm_stack.py`(numpy, 약 7초)로 PGS와 PADMM의 질량비 의존성을 확인했습니다.
- **공식 코드의 CPU 실행은 조건부입니다.** Newton과 Warp 설치가 필요하고 Warp CPU에서 동작하는지는 확인하지 못했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그래프는 모두 우리가 그린 것) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 결과 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 수치 재계산 (numpy, scipy, 수 초) |
| `analysis/toy_padmm_stack.py` | 토이: 쌓인 상자 LCP에서 PGS 대 PADMM (numpy, 약 7초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_padmm_stack.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv:2603.16536 (v2). arXiv abs 페이지의 license 링크는 non-exclusive-distrib/1.0입니다. 그래서 논문 그림은 쓰지 않았습니다.
- 공식 구현은 NVIDIA Newton 저장소(`github.com/newton-physics/newton`, Apache-2.0)의 kamino 솔버에 있고, 이 노트는 커밋 f39d819(2026-11-21)를 정적으로만 읽었습니다. 코드를 이 저장소에 넣지 않았고 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

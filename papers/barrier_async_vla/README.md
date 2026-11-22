# FreeBar (arXiv:2610.14997, CoRL 2026 포스터) 분석 노트

*Barrier-Constrained Guidance For Real-time Asynchronous VLA Execution* (arXiv v1, 2026-10-10)을 공식 코드의 정적 읽기, 표 수치 재계산, 통계 검사로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 비독점 라이선스). 공식 코드는 복제하거나 재배포하지 않았습니다.

## 구현 가능성 (요약)

- **알고리즘은 구현할 수 있습니다.** 핵심 계산은 공식 코드 기준 약 140줄이고 역전파가 없습니다.
- **RoboCasa와 실로봇 수치는 이 맥에서 재현할 수 없습니다.** CUDA, MuJoCo EGL, 체크포인트, 실로봇 코드가 필요합니다.
- **Kinetix는 조건부입니다.** JAX의 CUDA 고정 의존성 변경과 BC 정책 학습이 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그림은 우리가 그린 세 장) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전, 라이선스 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 수치 재계산, 일관성 검사, 통계 판단, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드와 논문의 차이, 0 패딩 문제, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/formula_check.py` | 가중치, 에너지, τ 일정, 0 패딩 비율 계산 (numpy, 1초 안팎) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/formula_check.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv non-exclusive distribution license 1.0입니다(abs 페이지 "view license" 링크 확인). CC BY가 아니므로 논문 그림은 싣지 않았습니다.
- 공식 코드는 `github.com/NamelessCrew/FreeBar`에 있고 Apache-2.0입니다. 이 자료는 코드를 포함하지 않습니다.
- 논문 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

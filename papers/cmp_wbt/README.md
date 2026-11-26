# CMP (arXiv:2604.07457, CoRL 2026 스포트라이트) 분석 노트

*CMP: Robust Whole-Body Tracking for Loco-Manipulation via Competence Manifold Projection* (arXiv v2, 2026-10-12)을 표 재계산, 통계 검정, 두 개의 작은 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2604.07457에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- **시뮬레이션(Isaac Gym)과 실로봇 재현은 이 맥에서 불가입니다.** NVIDIA GPU와 Unitree Go2 + 팔이 필요하고, 공식 저장소에는 아직 코드가 없습니다.
- **식 3, 5–7 구조의 저차원 구현과 논리 검사는 가능합니다.** `analysis/toy_lambda_bound.py`와 `analysis/toy_shell_geometry.py`(각 수 초, numpy)로 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 결과 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C15, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_lambda_bound.py` | 토이: 부록 C 하한 증명의 수치 확인 (numpy, 1초 미만) |
| `analysis/toy_shell_geometry.py` | 토이: 차원별 셸 두께 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_lambda_bound.py
python3 analysis/toy_shell_geometry.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv abs 페이지의 라이선스 링크가 arXiv 비독점 배포 라이선스(CC BY 아님)를 가리킵니다. 그래서 논문 그림은 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 저장소(`github.com/Shepherd1226/CMP`)는 2026-11-26 기준 README와 MIT 라이선스 파일만 있었습니다. 코드, 가중치, 데이터는 받지 않았고 실행하지 않았습니다.

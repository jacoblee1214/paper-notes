# Beyond the End-Effector (arXiv:2610.17802, CoRL 2026 포스터) 분석 노트

*Beyond the End-Effector: Learning Whole-Arm Contact Manipulation from Human Demonstrations* (arXiv v1, 2026-10-13)을 표 재계산, 통계 검정, 작은 삼각측량 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2610.17802에서 받을 수 있습니다). 논문 그림은 두 장(그림 6, 8)만 넣었고 출처를 적었습니다. 공식 코드는 공개되지 않아 코드 대조 대신 주장 검증을 했습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, MPS, CUDA 없음)에서 불가능합니다.** 코드가 없고 실로봇, GPT-5.5 호출, cuRobo(CUDA), SAM3와 손 추적기가 필요합니다.
- **핵심 알고리즘의 축소판은 새로 구현할 수 있습니다.** 표면 접촉 IK, 링크·앵커 빔 탐색, 구속 게이트를 시뮬레이션에서 짜는 것이며 노력은 3–4주(추정, 측정 전)입니다.
- **재계산과 토이는 끝냈습니다.** `analysis/recompute.py`와 `analysis/toy_triangulation.py`(둘 다 1초 미만).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성요소 출처, GitHub 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 결과 |
| `analysis/04_claim_verification.md` | 주장 C1–C20 판정, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy) |
| `analysis/toy_triangulation.py` | 토이: 다중 뷰 삼각측량 오차 (numpy, 가정값) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 두 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_triangulation.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지의 license 링크 확인). 원문 PDF의 저작권 고지는 찾지 못했습니다.
- 논문 그림 두 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2610.17802"). 장면·실로봇 사진이 든 그림은 넣지 않았습니다.
- 공식 코드는 공개되지 않았습니다(프로젝트 페이지 https://wacl-robot.github.io, 2026-11-23 확인).
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

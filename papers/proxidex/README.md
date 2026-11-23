# ProxiDex (arXiv:2609.16586, CoRL 2026 포스터) 분석 노트

*ProxiDex: Learning Dynamics-Guided Proximity Policy for Dexterous Manipulation* (arXiv v1, 2026-09-15)을 표 재계산, 통계 검정, 주장 검증, 근접도 토이로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2609.16586에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **논문의 시뮬·실로봇 결과 재현은 불가입니다.** 공식 코드(프로젝트 페이지 "Code: Coming Soon"), 시연 데이터, 자체 시뮬 과제, 체크포인트가 공개되지 않았고 실로봇이 필요합니다.
- **근접도 계산과 구조의 소형 재구현은 가능합니다.** `analysis/toy_proximity_noise.py`(약 15초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프 다섯 장) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 검정, 토이 결과 |
| `analysis/04_claim_verification.md` | 주장 검증, 미기재 항목, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_proximity_noise.py` | 토이: 자세 오차와 근접도 (numpy, scipy) |
| `analysis/plot_figures.py` | 그래프 다섯 장 생성 |
| `report/figures/` | 우리가 그린 그래프 다섯 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_proximity_noise.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스로 공개되어 있습니다(CC BY가 아님, arXiv abs 페이지의 license 링크 확인). 원문 PDF의 저작권 표시는 찾지 못했습니다. 논문 그림은 넣지 않았습니다.
- 공식 코드는 공개되지 않았습니다. 프로젝트 페이지는 `https://proxidex.github.io/`입니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

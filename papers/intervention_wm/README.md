# How to Learn from What a Human Would Avoid? (WHIRL, arXiv:2609.06009, CoRL 2026) 분석 노트

*WHIRL* (arXiv v1, 2026-09-05)를 표 재계산, 그림 판독, 작은 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 네 장은 우리가 만든 것입니다. 공식 코드는 아직 공개되지 않아(프로젝트 페이지 "Code coming soon") 논문의 주장을 표, 그림, 식에서 직접 검증했습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** Franka FR3, 16-DoF LEAP Hand, Manus 장갑, 페달, 시연 데이터와 RTX 4090이 필요하고 코드가 없습니다.
- **알고리즘 핵심(세계모델 네 머리, 게이트, 잔차 SAC, actor 보조 손실)의 소형 시뮬 구현은 가능합니다.** 추정 2–4일이고 소형 시뮬 비교까지 1–2주입니다(측정 전).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 이력(v1 하나) |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 1과 부록 D 재계산(V1–V6), 그림 5·6 판독(V7–V12), 표 3(V13–V14), 토이 1·2, 식 (10) 검증 |
| `analysis/04_claim_verification.md` | 주장 C1–C24 판정, 빠진 정보 K1–K10, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표·그림 값 재계산과 일관성 검사 (numpy, scipy, statsmodels, 수 초) |
| `analysis/digitize_fig.py` | 논문 그림 5·6의 마커를 읽는 스크립트 (논문 PDF 경로를 인자로 받음, poppler 필요) |
| `analysis/fig_digitized.json` | 읽은 값 (그림 판독 값, 오차 약 ±0.005) |
| `analysis/toy_single_seed.py` | 토이 1: 시드 하나 체크포인트에 Fisher 검정을 쓸 때 (numpy, 수 초) |
| `analysis/toy_hil_corridor.py` | 토이 2: 개입 확률 항을 actor에만 쓰는 설계 (numpy, 약 30초) |
| `analysis/check_retarget_gn.py` | 부록 A 식 (10)의 수치 검증 (numpy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy statsmodels matplotlib pillow
python3 analysis/recompute.py
python3 analysis/toy_single_seed.py
python3 analysis/toy_hil_corridor.py
python3 analysis/check_retarget_gn.py
python3 analysis/plot_figures.py
# 그림 판독 값을 다시 만들려면 논문 PDF를 받아서 (poppler 필요)
python3 analysis/digitize_fig.py paper.pdf
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-24). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드는 없습니다. 프로젝트 페이지(https://whirl-dexterous.github.io/)에는 "Code coming soon"만 있습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그림 판독 값은 논문 그림에서 읽은 것이며 원자료가 아닙니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

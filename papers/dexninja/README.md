# DexNinja (arXiv:2610.17208, CoRL 2026 포스터) 분석 노트

*DexNinja: Learning Robust Dexterous Cutting Policy with a Real-to-Sim-to-Real Data Engine* (arXiv v1, 2026-10-12)을 표 재계산, 통계 검정, 시행 잡음 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2610.17208에서 받을 수 있습니다). 논문 그림은 그림 4의 패널 (b), (c)만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **실로봇 실험은 이 맥에서 불가입니다.** 양팔 DR3/DR5, LinkerHand L10, 촉각, 데이터 수집 장갑, 소모되는 음식이 필요합니다.
- **공식 파이프라인은 코드가 없어 불가입니다.** 시뮬레이션은 에피소드 하나에 H200에서 GPU 메모리 약 20 GB, 약 10분입니다.
- **표 재계산과 토이는 가능하고 끝냈습니다.** `analysis/recompute.py`, `analysis/toy_ranking_mc.py`(각 수 초, numpy).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림은 그림 4 일부만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 빠진 정보 |
| `analysis/02_prior_work.md` | 선행 계보, 코드·자원 표 |
| `analysis/03_analysis.md` | 재계산 V1–V13, 일관성 K1–K14, 시행 잡음 토이 |
| `analysis/04_claim_verification.md` | 주장 C1–C19 판정, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, statsmodels, 수 초) |
| `analysis/toy_ranking_mc.py` | 토이: 표 1·2 결과의 이항 잡음 견고성 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/crop_figures.py` | 논문 그림 4의 패널 (b), (c) 자르기 (원문 PDF 필요) |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 일부 한 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy statsmodels matplotlib
python3 analysis/recompute.py
python3 analysis/toy_ranking_mc.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림은 그림 4의 (b), (c)만 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2610.17208"). 실로봇 사진, 시뮬레이션 화면, 생성 영상이 있는 그림은 넣지 않았습니다.
- 공식 코드는 확인되지 않았습니다(프로젝트 페이지 "Code Coming soon"). 시뮬레이션 에셋은 받지 않았고 크기는 목록 메타데이터에서 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

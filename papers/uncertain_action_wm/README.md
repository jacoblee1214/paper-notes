# ACTUS (arXiv:2611.23620) 분석 노트

*How Uncertain Is My Action? Self-Correcting Visual Planning with Frozen World Models* (arXiv v1, 2026-11-17)를 표 재계산, 통계 검정, CPU 토이로 분석한 자료입니다. 학회 표기는 없고, 공식 코드와 프로젝트 페이지도 찾지 못했습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.23620에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **재현은 이 맥에서 불가입니다.** 실로봇(SO-101)은 장비가 없고, 세계모델 학습은 CUDA GPU가 필요합니다.
- **구현은 조건부로 가능합니다.** 점수와 가우스-뉴턴 추정은 공개된 LeWM(MIT, PyTorch) 추론 위에 얹을 수 있습니다. 설치와 체크포인트 다운로드가 먼저 필요하고, MPS의 JVP 동작은 확인하지 못했습니다.
- **CPU로 확인한 것:** 검출 법칙의 정확 값과 식 (8) 근사 비교, 비백색 잡음에서 κ 보정 토이, 표 재계산과 일관성 검사.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 코드와 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산, 일관성 검사, 검출 법칙 확인, 토이 |
| `analysis/04_claim_verification.md` | 주장 대조표 (공식 코드 없음), 구현 가능성 판정 |
| `analysis/recompute.py` | 표 산술, 계수 구간, 정확 AUROC, 정리 2 분해 (수 초, numpy·scipy) |
| `analysis/auroc_mc_check.py` | 토이: AR(1) 잡음에서 κ 보정 후 검출 AUROC (수십 초, numpy) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/auroc_mc_check.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 라이선스(nonexclusive-distrib 1.0)로 배포되며 CC BY가 아닙니다. 그래서 논문 그림은 넣지 않았습니다.
- PDF에서 저작권 표기는 찾지 못했습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 코드는 찾지 못했습니다. LeWM(lucas-maes/le-wm, MIT)은 기반 세계모델이며 이 저장소에 복제하지 않았습니다.

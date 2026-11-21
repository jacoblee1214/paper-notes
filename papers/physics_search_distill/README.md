# Efficient World-Model Planning via Physics-Guided Search Distillation (arXiv:2611.24171) 분석 노트

PGSD (arXiv v2, 2026-11-19)를 표 재계산과 구성 요소 정리로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 두 장은 논문 수치로 우리가 그린 것입니다. 학회 표기는 찾지 못했고, 공식 코드와 프로젝트 페이지도 찾지 못했습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** NAVSIM 데이터, 플래너 체크포인트, 추론 GPU가 필요하고 공식 코드가 없습니다.
- **핵심 알고리즘의 축소판은 가능합니다.** 2D 도로, 자전거 모델, 추종 제어기, 여유 계산, 실행 선택, 한 번 검증, 탐색 교사를 numpy로 짜는 것입니다 (노력 3–5일 추정, 미측정).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계, 버전 차이, 라이선스 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 표 |
| `analysis/03_analysis.md` | 수치 재계산: 정수 계수, 목표 선택 구간, 보유율 범위, 파라미터, PDMS 상한 |
| `analysis/04_claim_verification.md` | 주장 검증 (C1–C30), 구현 가능성 판정, 저자 질문 |
| `analysis/recompute.py` | 재계산과 일관성 검사 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다 (abs 페이지 확인, 2026-11-21). 원문과 논문 그림은 이 폴더에 넣지 않았습니다.
- 원문 PDF는 arXiv 2611.24171에서 받을 수 있습니다: https://arxiv.org/abs/2611.24171
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았고, 공식 코드도 없어 실행하지 않았습니다.

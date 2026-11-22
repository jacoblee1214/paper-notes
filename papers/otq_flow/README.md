# OTQL (arXiv:2607.06262, CoRL 2026 포스터) 분석 노트

*Optimal Transport Q-Learning for Flow Policy Steering and Acceleration* (arXiv v1, 2026-07-07)을 표·그림 수치 재계산, 통계 검정, 2D 장난감으로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2607.06262에서 받을 수 있습니다). 공식 코드는 공개되지 않았습니다.

## 구현 가능성 (요약)

- **실로봇(단일 과제, SmolVLA 적응)과 MuJoCo 두 과제 재현은 이 맥에서 불가입니다.** 로봇, GPU, 비공개 환경과 시연이 필요합니다.
- **OGBench 오프라인 전체는 사실상 불가입니다.** 한두 과제의 축소 학습은 패키지 설치와 데이터 다운로드 뒤 가능합니다.
- **알고리즘 구조 구현과 장난감은 가능합니다.** `analysis/toy_wcot.py`(약 3분, torch CPU)로 핵심 부분을 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 판본·권리, 방법, 실험 구성, 보고 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, 공식·관련 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 장난감 결과 |
| `analysis/04_claim_verification.md` | 주장 검증, 구현 가능성 |
| `analysis/recompute.py` | 표·곡선 수치 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/extract_curves.py` | PDF 벡터 그래프에서 점 값 추출 (PyMuPDF) |
| `analysis/toy_wcot.py` | 장난감: 가중 질량 OT 결합 대 가중 손실 FM (torch CPU, 약 3분) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib torch pymupdf
python3 analysis/extract_curves.py paper.pdf   # arXiv:2607.06262 PDF를 paper.pdf로 받은 뒤
python3 analysis/recompute.py
python3 analysis/toy_wcot.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스(CC 아님)로 공개되어 있습니다(arXiv 라이선스 링크 확인). PDF 저작권 표시는 찾지 못했습니다. 그래서 논문 그림은 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 기준선 값 대조에는 arXiv:2601.14234의 표를 썼습니다. 그래프, 장난감, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 코드가 없습니다(프로젝트 페이지 https://ansocho.github.io/otql-flow/ 의 "Code (Coming Soon)"). 가중치와 데이터는 받지 않았습니다.

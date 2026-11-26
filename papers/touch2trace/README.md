# Touch2Trace (arXiv:2609.15921, CoRL 2026 포스터) 분석 노트

*Touch2Trace: Tactile-Driven Imitation Learning for Dexterous Cable Tracing* (arXiv v1, 2026-09-14)의 표와 본문 수치를 재계산하고 주장을 검증한 자료입니다. 공식 코드가 공개되어 있지 않아 코드 대조는 하지 않았습니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2609.15921에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **실물과 논문 수치 재현은 공개 자료만으로 할 수 없습니다.** 코드, 데이터, 시뮬레이터, 센서 설계가 공개되어 있지 않고 전용 손과 센서가 필요합니다.
- **구조 재구현과 합성 데이터 절제는 일반 노트북급에서 가능해 보입니다.** 이번 노트에서는 실행하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치 |
| `analysis/02_prior_work.md` | 선행 계보, 코드와 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V12, 작은 계산 실험 K1–K2 |
| `analysis/04_claim_verification.md` | 주장 판정 C1–C16, 저자 확인 항목, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv non-exclusive 라이선스(arxiv.org/licenses/nonexclusive-distrib/1.0)이고 CC BY가 아니므로 논문 그림은 싣지 않았습니다. 이 저장소의 그래프는 작성자가 다시 그린 것입니다.
- 프로젝트 페이지는 grmmtt.github.io/corl26-touch2trace 입니다. 코드와 데이터는 공개 전입니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 가중치와 데이터셋은 받지 않았습니다.

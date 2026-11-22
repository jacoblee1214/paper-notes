# Visuo-Tactile World Models (arXiv:2602.06001) 분석 노트

*Visuo-Tactile World Models* (arXiv v1, 2026-02-05)를 수치 재계산과 일관성 검사로 분석한 자료입니다. 학회 표기는 없습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2602.06001에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **전체 구현은 이 맥에서 불가입니다.** 173M 모델 학습은 GPU가 필요하고, 실로봇 평가는 로봇 장비가 필요합니다.
- **공식 코드는 공개되어 있지 않습니다.** 프로젝트 페이지(carolinahiguera.github.io/vtml)에도 코드 링크가 없습니다.
- 보고 수치 재계산은 `analysis/recompute.py`로 할 수 있습니다(numpy, scipy, 수 초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 공개 상태 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C8, 구현 가능성 |
| `analysis/recompute.py` | 본문 수치 재계산 (numpy, scipy) |
| `report/plot_planning.py` | 개선 폭과 5회 격자 그래프 |
| `report/figures/` | 우리가 그린 그래프 한 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 report/plot_planning.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 라이선스(arXiv non-exclusive distribution 1.0)입니다. CC BY가 아니므로 논문 그림은 넣지 않았습니다.
- 표와 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

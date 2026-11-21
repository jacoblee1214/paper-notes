# TacPilot (arXiv:2611.23804) 분석 노트

*TacPilot: Does Touch Need to Be Everywhere?* (arXiv v1, 2026-11-18)의 표 재계산과 주장 검증 자료입니다. 원문 PDF는 넣지 않았습니다(arXiv:2611.23804에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **이 맥에서 재현할 수 없습니다.** 공식 코드, 체크포인트, 시뮬 모델, 데이터가 공개되지 않았습니다. 실로봇 결과는 실로봇 없이 확인할 수 없습니다.
- **가능한 것은 표 재계산과 검정입니다.** `analysis/recompute.py`(수 초)로 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 설계와 수치 |
| `analysis/03_analysis.md` | 표 재계산, 검정, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C20, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy) |
| `analysis/plot_figures.py` | 그래프 두 장 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv non-exclusive distribution 1.0 라이선스이며 CC BY가 아닙니다. 논문 그림은 넣지 않았습니다.
- 그래프와 분석은 이 저장소 작성자가 만들었습니다. 논문 표 값은 인용과 재계산을 위해 옮겼습니다.
- 공식 코드는 찾지 못했습니다. 이 저장소에는 코드를 넣지 않았고 실행하거나 설치하지 않았습니다.

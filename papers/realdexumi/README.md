# RealDexUMI (arXiv:2606.06033, CoRL 2026 포스터) 분석 노트

*RealDexUMI: A Wearable Universal Manipulation Interface for Dexterous Robot Learning* (arXiv v2, 2026-06-06)을 표 재계산, 통계 검정, 상대 행동 식 검사로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.06033에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 그래프는 모두 이 분석에서 새로 그린 것입니다.

## 구현 가능성 (요약)

- **이 맥에서 전체 재현은 불가능합니다.** 손 모듈, 장갑, 촉각 센서의 설계와 데이터가 공개되지 않았고, 교차 로봇과 실로봇 시험이 필요합니다.
- **공식 코드는 찾지 못했습니다.** 이름이 같은 저장소는 익명 제출 보충 영상 페이지입니다. 그래서 코드 대조 없이 주장 대 본문, 부록, 재계산으로 검증했습니다.
- **재계산과 식 검사는 가능하고 완료했습니다.** 재계산은 수 초, 식 검사는 1–2초입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, v1에서 v2로의 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사, 식 검사 |
| `analysis/04_claim_verification.md` | 주장 검증과 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, statsmodels) |
| `analysis/toy_relative_action.py` | 상대 행동 식 검사 (numpy) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy statsmodels matplotlib
python3 analysis/recompute.py
python3 analysis/toy_relative_action.py
python3 analysis/plot_figures.py
```

`recompute.py`의 입력은 논문 v2의 표 값입니다. 원시 시도 기록은 공개되지 않았습니다.

## 출처와 라이선스

- 논문 라이선스는 arXiv 비독점 배포 라이선스(nonexclusive-distrib/1.0)이고, CC 라이선스가 아닙니다. 그래서 논문 그림을 싣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 식 검사, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 저장소, 데이터, 가중치는 받지 않았습니다. 실행하거나 설치하지 않았습니다.

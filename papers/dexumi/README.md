# DexUMI (arXiv:2505.21864, CoRL 2025) 분석 노트

*DexUMI: Using Human Hand as the Universal Manipulation Interface for Dexterous Manipulation* (arXiv v3, 2025-10-02)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2505.21864에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 그래프는 모두 이 분석에서 새로 그린 것입니다.

## 구현 가능성 (요약)

- **이 맥에서 전체 재현은 불가능합니다.** 외골격 제작, 두 손, 실로봇 평가가 필요합니다. 식 (1) 최적화의 CPU 재실행과 엔코더–모터 회귀 재적합은 조건부로 가능합니다.
- **공식 코드는 `real-stanford/DexUMI` (MIT)입니다.** 이 저장소는 정적으로만 읽었습니다. 실행, 설치, 가중치 다운로드는 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | venue 판정, 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 차이와 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (scipy) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 그래프 두 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

`recompute.py`의 입력은 논문 v3의 표 1 값입니다. 원시 시도 기록은 공개되지 않았습니다.

## 출처와 라이선스

- 논문 라이선스는 arXiv 비독점 배포 라이선스(nonexclusive-distrib/1.0)이고, CC 라이선스가 아닙니다. 그래서 논문 그림을 싣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 코드(MIT)는 저장소에 포함하지 않았습니다.

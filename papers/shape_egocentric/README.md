# SHAPE (arXiv:2611.24585) 분석 노트

*SHAPE: Discrete Supervision and Parallel Continuous Streams for Learning Dexterous Manipulation from Egocentric Human Data* (arXiv v1, 2026-11-18)을 표 재계산과 통계 검정으로 분석한 자료입니다. 학회 표기는 없습니다(arXiv 단독). 공식 코드는 찾지 못했습니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv non-exclusive 라이선스). 원문은 arXiv:2611.24585에서 받을 수 있습니다.

## 구현 가능성 (요약)

- **논문 전체 재현은 일반 개인 워크스테이션에서 불가능합니다.** 4B 백본 학습, 비공개 사람 데이터, 실로봇, 공개되지 않은 체크포인트가 필요합니다.
- **재계산과 검정은 완료했습니다.** 학습과 실행은 하지 않았습니다.
- **축소 토이는 계획만 있습니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 검색 결과 |
| `analysis/03_analysis.md` | 재계산, 셀 단위 일관성 검사, 과제 단위 검정 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C17, 구현 가능성 판정 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/robocasa_task_level.py` | 시뮬 결과의 과제 단위 쌍체 검정 (1초 미만) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/robocasa_task_level.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv:2611.24585 v1. 라이선스는 arXiv non-exclusive 배포 라이선스이며 CC BY가 아닙니다. 그래서 논문 그림은 넣지 않았습니다.
- 그래프와 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다. 표 값은 인용과 재계산을 위해 옮겼습니다.
- 가중치와 데이터셋은 받지 않았습니다.

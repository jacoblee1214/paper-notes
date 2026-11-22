# TLA (arXiv:2503.08548) 분석 노트

*TLA: Tactile-Language-Action Model for Contact-Rich Manipulation* (arXiv v1, 2025-03-11)을 표 재계산, 통계 검정, 주장 대조로 분석한 자료입니다. 저널 게재본(*Robot Learning* 2026, DOI 10.55092/rl20260001)과의 차이는 `analysis/01_paper_analysis.md` §0에 있습니다.

## 요약

- **공식 코드와 데이터는 찾지 못했습니다.** 프로젝트 페이지에는 영상만 있고, 게재본은 데이터를 "문의 시 제공"이라고 씁니다. 그래서 코드 대조 대신 주장 검증(`analysis/04_claim_verification.md`)을 했습니다.
- **이 맥에서 논문의 재현은 불가능합니다.** 7B 모델 학습과 CUDA 기반 시뮬레이션이 필요합니다. 라벨 식과 기준선 L1 계산은 numpy로 가능합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 탐색 결과 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 검증, 라이선스, 구현 가능성 판정 |
| `analysis/recompute.py` | 표 재계산과 검정 (표준 라이브러리, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문 arXiv 페이지의 라이선스 링크는 CC BY-NC-SA 4.0입니다. 논문 그림은 넣지 않았습니다.
- 그래프와 표 재계산은 이 노트의 작성자가 논문 표 값을 옮겨 다시 그리고 계산한 것입니다.
- 가중치와 데이터는 받지 않았습니다.

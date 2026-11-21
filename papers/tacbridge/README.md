# TacBridge (arXiv:2611.20939) 분석 노트

*TacBridge: Learning Unified Tactile Representations through Cross-Sensor Correspondence* (arXiv v1, 2026-11-16)을 표 재계산과 일관성 검사로 분석한 자료입니다. 학회 표기는 찾지 못했습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.20939에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **논문 결과 재현은 이 맥에서 불가능합니다.** 공식 코드, 체크포인트, 데이터가 공개되지 않았고, 실로봇 평가가 필요합니다.
- **방법의 소형 재구현은 조건부입니다.** 데이터 내려받기와 설치 승인이 먼저 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 학회 확인 결과, 선행 계보, 코드 탐색 결과 |
| `analysis/03_analysis.md` | 표 재계산, 일관성 검사, OOD 검정력 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C14, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 식 (11) 검산 (표준 라이브러리, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스(nonexclusive-distrib 1.0)로 공개되어 있습니다. CC BY가 아니므로 논문 그림은 쓰지 않았습니다.
- 그래프는 논문 표의 값으로 이 저장소 작성자가 그렸습니다.
- 공식 코드와 데이터는 찾지 못했습니다. 아무것도 받거나 설치하거나 실행하지 않았습니다.

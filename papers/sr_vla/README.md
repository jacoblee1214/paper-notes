# SR-VLA (arXiv:2611.22356) 분석 노트

*SR-VLA: Stall-Reduced VLA Execution via Predicted-Input Lookahead and Selective KV Cache Repair* (arXiv v1, 2026-11-17)을 표 재계산, 통계 검정, 주장 검증으로 분석한 자료입니다. 원문 PDF는 넣지 않았습니다(arXiv:2611.22356에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 결론 요약

- **구조는 구현할 수 있고, 수치 재현은 이 맥에서 불가능합니다.** 지연은 A100 40 GB에서 잰 값이고 공식 코드가 없습니다.
- **질의 속도 2.67×의 산술은 맞지만**, 원문은 시행 수와 분산을 보고하지 않습니다.
- **성공률 차이(+0.5 pp)는 0을 포함합니다.** 시행 500을 가정한 검정에서 p = 0.38입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산, 성공률 검정, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C12, 라이선스, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 표 재계산과 검정 (표준 라이브러리, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
python3 analysis/recompute.py
python3 analysis/plot_figures.py   # matplotlib 필요
```

## 출처와 라이선스

- 논문 라이선스는 arXiv의 nonexclusive-distrib 1.0입니다(확인 2026-11-22). CC BY가 아니므로 논문 그림은 넣지 않았습니다.
- 공식 코드와 프로젝트 페이지는 찾지 못했습니다(2026-11-22).
- 표 수치는 인용과 재계산을 위해 옮겼습니다. 그래프와 분석은 이 저장소 작성자가 만들었습니다.

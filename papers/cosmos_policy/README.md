# Cosmos Policy (arXiv:2601.16163, ICLR 2026) 분석 노트

*Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning* (v1, 2026-01-22)을 표 재계산, 일관성 검사, 가치 집계 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2601.16163에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 그래프는 논문 수치로 우리가 다시 그렸습니다.

## 구현 가능성 (요약)

- **학습은 이 맥에서 불가능합니다.** 저자는 8~64 H100으로 48시간을 씁니다.
- **추론은 조건부입니다.** 공식 코드가 CUDA 호출을 여러 곳에서 쓰므로 포팅이 필요합니다. 추정 1–2주입니다(측정 없음).
- **실로봇 재현은 불가능합니다.** ALOHA 계획 학습용 rollout 648개는 공개되지 않았습니다.
- **재계산과 토이는 가능하고 완료했습니다.** `analysis/recompute.py`(수 초), `analysis/toy_majority_mean.py`(수 초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 결과, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (표준 라이브러리) |
| `analysis/toy_majority_mean.py` | 토이: 가치 집계 규칙 비교 (numpy) |
| `report/plot_figures.py` | 그래프 한 장 생성 (matplotlib) |
| `report/figures/own_ci.png` | 우리가 그린 그래프 |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_majority_mean.py
python3 report/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs의 license 링크 확인). PDF 메타데이터의 `/License`와 `dc:rights`도 CC BY 4.0입니다. PDF 본문에 회사 저작권 표시는 없었습니다.
- 공식 코드는 `github.com/NVlabs/cosmos-policy`에 있고 Apache-2.0입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프와 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다. 라이선스도 확인하지 않았습니다.

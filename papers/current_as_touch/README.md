# Current as Touch (arXiv:2607.03529, CoRL 2026 포스터) 분석 노트

*Current as Touch: Proprioceptive Contact Feedback for Compliant Dexterous Manipulation* (arXiv v1, 2026-07-03)을 표의 원시 횟수로 다시 계산하고 본문과 대조한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2607.03529에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **공식 코드는 아직 공개되지 않았습니다.** 저장소에는 README 한 장("Pending release.")만 있고 라이선스가 없습니다. 그래서 논문 코드와의 대조 대신 주장과 표의 대조를 했습니다.
- **구조 재구현은 가능합니다.** 작은 Transformer 예측기와 실행 평활은 PyTorch로 만들 수 있습니다(추정 3–5일).
- **실험 재현은 이 맥에서 불가능합니다.** 실물 손과 로봇, 사람 조작 데이터가 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 원시 횟수 재계산, 검정, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장별 판정, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (scipy, 수 초) |
| `analysis/recompute_results.json` | 재계산 결과 |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문 라이선스 링크는 arXiv 비독점 배포 1.0입니다(CC BY 아님). 그래서 논문 그림을 싣지 않았습니다. PDF에 별도 저작권 표시는 없습니다.
- 학회 표기는 CoRL 2026 공식 채택 목록의 포스터입니다.
- 공식 저장소 `github.com/cublicma/current-as-touch`는 복제만 했고 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

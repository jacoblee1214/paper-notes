# AdvDex (arXiv:2608.14028, CoRL 2026 포스터) 분석 노트

*AdvDex: Learning Dexterous Manipulation from Human Demonstrations via Joint-Aligned Actions and Adversarial Learning* 을 표 재계산과 주장 검증으로 분석한 자료입니다. 공식 코드와 데이터가 공개되지 않아 재현은 하지 않았습니다.

## 구현 가능성 (요약)

- **전체 시스템은 구현 불가입니다.** 공식 코드, 체크포인트, 사람 데이터셋, 실로봇이 없습니다.
- **CPU/MPS에서 가능한 것:** 표 재계산 (완료), 슬롯 마스킹 손실과 적대 학습 신호의 작은 토이 (미실시).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 (없음) |
| `analysis/03_analysis.md` | 표 재계산 (Fisher, Newcombe, Wilson, 검정력), 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 검증, 구현 가능성 판정 |
| `analysis/recompute.py`, `recompute_results.json` | 표 재계산 (scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 비독점 라이선스(non-exclusive distribution 1.0)로 공개되어 있습니다. CC BY가 아니므로 논문 그림과 원문 PDF는 이 저장소에 넣지 않았습니다. 원문은 arXiv:2608.14028에서 받을 수 있습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 코드는 찾지 못했습니다. 실행하거나 설치하지 않았습니다.

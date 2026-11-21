# Right Checkpoint (arXiv:2611.23676) 분석 노트

*Finetuning VLAs for Dexterous Manipulation: The Right Checkpoint Is All You Need* (arXiv v1, 2026-11-18)를 표 재계산과 통계 검사로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.23676에서 받을 수 있습니다). 논문 그림은 쓰지 않았고 그래프는 모두 이 저장소 작성자가 그렸습니다.

## 구현 가능성 (요약)

- **원 실험 재현은 이 맥에서 불가입니다.** π0.5 SFT(2만 스텝)와 DSRL(레시피·체크포인트·과제·시드 120회, 실행당 50만 환경 스텝)은 GPU가 필요합니다.
- **공식 코드는 찾지 못했습니다.** 주장 검증은 표 수치와 본문 대조로 했습니다(`analysis/04_claim_verification.md`).
- **CPU로 한 것:** 표 재계산, 분모 검사, Wilson 구간, 상관과 강건성 검사 (`analysis/recompute.py`, 수 초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 코드 탐색 기록 |
| `analysis/03_analysis.md` | 재계산, 분모 검사, 상관, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 검증표 C1–C14, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs의 라이선스 링크 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 표 값(Table 5 등)은 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

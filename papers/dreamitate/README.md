# Dreamitate (arXiv:2406.16862, CoRL 2024) 분석 노트

Dreamitate를 표 재계산, 공식 코드 정적 대조로 읽은 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서는 논문 실험을 재현하지 않습니다.** 학습은 다중 GPU가 필요하고, 추론도 저장소 README 기준 약 70 GB VRAM이 필요합니다. 로봇 실험 장비도 필요합니다.
- 맥에서 가능한 것은 보고 수치 재계산(CPU, 몇 초)입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, GitHub와 자료 표, 라이선스 |
| `analysis/03_analysis.md` | 성공률 재계산 (Wilson CI, Fisher 정확 검정), 일관성 점검 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 정적 대조 (D1–D9) |
| `analysis/recompute.py`, `recompute_results.json` | 재계산 (표준 라이브러리만 사용) |
| `analysis/plot_figures.py` | 그래프 생성 (matplotlib) |
| `report/figures/own_ci_forest.png` | 우리가 그린 그래프 (Table 2 신뢰구간) |

## 재실행

```bash
python3 analysis/recompute.py > analysis/recompute_results.json
python3 analysis/plot_figures.py
```

## 원문 PDF

원문 PDF는 크기(31 MB) 때문에 이 저장소에 넣지 않았습니다. 필요하면 아래 명령으로 받습니다.

```bash
curl -L -o paper/paper_v1.pdf https://arxiv.org/pdf/2406.16862
```

## 출처와 라이선스

- 논문: arXiv:2406.16862 (v1). abs 페이지 라이선스는 CC BY 4.0입니다 (2026-11-21 확인). PDF에는 저작권 표기가 없습니다.
- 공식 코드(https://github.com/cvlab-columbia/dreamitate): 최상위 LICENSE.txt는 CC BY-NC 4.0입니다 (GitHub API는 NOASSERTION으로 표시). 이 분석은 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

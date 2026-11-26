# FACT (arXiv:2608.10232, CoRL 2026 포스터) 분석 노트

*FACT: Failure-Aware Causal Training for World-Action Models* (arXiv v2, 2026-10-13)를 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2608.10232에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **논문 수치의 재현은 GPU와 로봇이 필요합니다.** 공식 설정 학습(WAN2.2-5B, 8 GPU), RoboTwin 평가(CUDA 시뮬레이터), 실로봇이 해당합니다.
- **표 재계산과 정적 대조는 CPU에서 수 초에 끝납니다.** `analysis/recompute.py`가 해당합니다.
- **방법 핵심(마스크, 실패 행동 손실 마스킹, 가치 목표)의 소형 구현은 가능합니다.** 노력은 며칠 규모로 추정했고 측정하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

`analysis/recompute.py`와 `analysis/plot_figures.py`는 논문 텍스트(표 6)를 읽습니다. arXiv에서 PDF를 받아 `pdftotext -layout`으로 `paper/paper.txt`를 만든 뒤 실행하세요.

```bash
pip install numpy scipy matplotlib
mkdir -p paper
curl -L -o paper/paper.pdf https://arxiv.org/pdf/2608.10232v2
pdftotext -layout paper/paper.pdf paper/paper.txt
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY-NC-ND 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 그래서 논문 그림은 싣지 않았습니다.
- 공식 코드는 `github.com/Bariona/FACT`(Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 정적으로 읽기만 했습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

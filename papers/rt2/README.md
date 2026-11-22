# RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control (arXiv:2307.15818, CoRL 2023) 분석 노트

*RT-2* (arXiv v1, 2023-07-28; CoRL 2023 PMLR 229)의 보고 수치를 재계산하고, 본문 주장과 대조한 자료입니다. 재현(학습, 추론 실행)은 하지 않았습니다. 논문 원문과 논문 그림은 넣지 않았습니다(arXiv 라이선스 nonexclusive-distrib 1.0, 원문 저작권 고지). 이 폴더의 그래프 세 장은 보고된 수치로 우리가 그린 것입니다.

## 구현 가능성 (요약)

- **RT-2 자체는 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 재현할 수 없습니다.** 가중치와 코드가 공개되지 않았고, 55B는 TPU 클라우드를 전제로 합니다.
- **방법(행동 토큰화와 공동 미세조정)은 소형 공개 VLM으로 구현할 수 있습니다.** 토큰화와 평가 설계는 CPU로 가능하고, 공동 미세조정 학습은 GPU가 필요합니다. 노력은 수 주로 추정합니다(측정하지 않음).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계, arXiv와 CoRL 판본 비교 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(평균, 정수 검사, 검정, 셀 크기 가정), 그래프 |
| `analysis/04_claim_verification.md` | 주장 대조표 (C1–C22), 불일치 목록, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 재계산과 일관성 검사 (scipy, 1초 미만) |
| `analysis/recompute_results.json` | 재계산 결과 |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv:2307.15818 (abs 페이지의 라이선스는 nonexclusive-distrib 1.0, 2026-11-22 확인). CoRL 2023 판본은 PMLR 229:2165–2183.
- 공식 코드는 없습니다. 프로젝트 페이지 https://robotics-transformer2.github.io/ 에는 데모 영상이 있습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

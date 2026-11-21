# SemPhys: Semantic-Physical Co-Learning for Humanoid Motion Generation via Differentiable World Models (arXiv:2611.24074) 분석 노트

*SemPhys* (arXiv v1, 2026-11-18)를 표 재계산, 본문 일관성 검사, 주장 검증으로 읽은 자료입니다. 학회 표기는 없습니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 CC BY-NC-ND 4.0). 이 폴더의 그래프 세 장은 우리가 만든 것입니다. 공식 코드와 프로젝트 페이지는 찾지 못했습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 공식 코드가 없고, 생성기 사전학습과 적응 학습, 추종 제어기가 GPU 전제이며, 데이터와 가중치를 받지 않았습니다.
- **핵심 아이디어는 작은 모델에서 구현해 볼 수 있습니다**(노력 1–2주, 추정). 이 폴더는 재현 결과가 아니라 논문 수치의 재계산과 검증입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 주장과 근거의 간격, 본문 내 불일치 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 실패 사건 수 검정, 절제 Welch 검정, 재생 검사 구간 |
| `analysis/04_claim_verification.md` | 주장 검증, 구현 가능성 판정, 저자에게 물을 질문 |
| `analysis/recompute.py` | 재계산과 검정 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY-NC-ND 4.0으로 공개되어 있습니다(arXiv abs 페이지 확인, 2026-11-21). 원문과 논문 그림은 이 폴더에 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 폴더 작성자가 만들었습니다.
- 공식 코드와 가중치, 데이터셋은 받지 않았습니다(없거나 받지 않음).

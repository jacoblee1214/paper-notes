# Genie (arXiv:2402.15391, ICML 2024) 분석 노트

*Genie: Generative Interactive Environments* (arXiv v1, 2024-02-23)을 표 재계산과 산술 검사, 주장 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2402.15391에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- **전체 재현은 불가능합니다.** 공식 코드, 가중치, 데이터가 공개되지 않았습니다. 256 TPU급 학습이 필요합니다.
- **CoinRun 축소판은 조건부입니다.** 부록 F의 설정은 구체적이지만 해상도와 총 학습 시간이 없습니다.
- **합성 토이 LAM은 CPU에서 가능합니다.** 이 저장소에서 아직 만들지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 버전 기록 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 코드 유무, 커뮤니티 구현 표 |
| `analysis/03_analysis.md` | 표 재계산, 산술 검사, 유의성 검정 불가 근거 |
| `analysis/04_claim_verification.md` | 주장 대조 14개, 라이선스, 축소판 설정 검토, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 산술 검사 (표준 라이브러리, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |
| `report/figures/` | 그래프 두 장 (우리가 그림) |

## 재실행

```bash
pip install matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 원문 PDF

```bash
curl -L -o paper/paper.pdf https://arxiv.org/pdf/2402.15391v1
```

## 출처와 라이선스

- 논문 라이선스는 arXiv의 non-exclusive distribution 1.0입니다(CC BY가 아닙니다). 논문 PDF에는 "© 2024 Google DeepMind. All rights reserved" 표기가 있습니다. 그래서 논문 그림은 넣지 않았습니다.
- 표의 점 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 코드는 없습니다. 커뮤니티 구현 4개는 메타데이터만 확인했고 클론, 설치, 실행하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다.

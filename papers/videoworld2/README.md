# VideoWorld 2 (arXiv:2602.10102, CVPR 2026) 분석 노트

*VideoWorld 2: Learning Transferable Knowledge from Real-world Videos* (arXiv v1, 2026-02-10)을 표 재계산, 통계 검정, 공식 저장소 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2602.10102에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 그래프 두 장은 표 값으로 우리가 그렸습니다.

## 구현 가능성 (요약)

- **학습 재현은 이 맥에서 불가능합니다.** dLDM 학습은 CUDA GPU 여러 장이 필요하고, 자기회귀 단계 학습 코드는 공식 저장소에 없습니다.
- **공식 체크포인트 추론 한 사례는 조건부입니다.** 기반 Cosmos 2B 가중치는 NVIDIA 게이트 라이선스이고, 의존성을 MPS용으로 바꿔야 합니다. 미검증입니다.
- **재계산과 정적 대조는 완료했습니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전과 학회 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv abs 페이지의 license 링크는 `creativecommons.org/licenses/by/4.0/`(CC BY 4.0)입니다. PDF 본문과 첫·마지막 쪽에서 저작권 고지를 찾지 못했습니다.
- 공식 코드: 저장소 루트 Apache-2.0. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 기반 모델 Cosmos-Predict2-2B는 NVIDIA 라이선스(게이트)입니다. 가중치는 받지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.

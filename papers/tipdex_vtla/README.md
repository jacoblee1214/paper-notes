# TipDex-VTLA (arXiv:2611.25683) 분석 노트

*TipDex-VTLA: Learning When to Refine Actions with Touch for Dexterous Manipulation* (arXiv v1, 2026-11-19)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 학회 표기는 없습니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 비독점 라이선스). 그래프는 우리가 논문 표 값으로 그렸습니다.

## 구현 가능성 (요약)

- **공식 추론은 조건부입니다.** 저장소는 CUDA 12를 전제합니다. CPU 포팅 가능성은 측정하지 않았습니다.
- **학습 재현은 불가입니다.** 학습 루프가 없고 A100 8장 규모입니다.
- **실로봇과 시뮬 평가는 불가입니다.** 하드웨어와 평가 코드가 공개되지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행연구 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산, 검정, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조 (D1–D13), 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스(nonexclusive-distrib/1.0)입니다. CC BY가 아니므로 논문 그림은 싣지 않았습니다.
- 공식 추론 코드는 `github.com/tiplab/TipDex-VTLA`(Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 모델 패키지는 Gemma 조건이 붙습니다. 데이터셋과 가중치는 받지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.

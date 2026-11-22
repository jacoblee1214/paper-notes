# Being-H0 (arXiv:2507.15597, ICML 2026 포스터) 분석 노트

*Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos* (arXiv v1, 2025-07-21)을 본문 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 라이선스가 CC BY가 아닙니다). 원문은 arXiv:2507.15597에서 받을 수 있습니다.

## 구현 가능성 (요약)

- **학습과 실로봇 재현은 이 맥에서 불가능합니다.** 공식 저장소에 학습 코드가 없고, 실로봇 장비도 없습니다.
- **추론은 조건부입니다.** 1B와 8B는 CPU 경로만 있고 MPS 경로는 없습니다. 14B는 24 GiB에 들어가지 않습니다.
- **재계산과 검정은 완료했습니다.** `analysis/recompute.py`(numpy, scipy, 수 초)와 `analysis/plot_figures.py`로 확인할 수 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (공개용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전과 학회 확인 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소와 모델 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 값 재계산과 검정 (numpy, scipy) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv:2507.15597 v1입니다. arXiv 라이선스는 nonexclusive-distrib 1.0입니다(CC BY가 아님). 논문 그림은 싣지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트와 분석 문서는 이 저장소 작성자가 만들었습니다.
- 공식 코드(MIT)는 정적으로만 읽었습니다. 설치하거나 실행하지 않았습니다. 가중치와 데이터는 받지 않았습니다(설정 JSON 몇 개만 받음).
- MANO 파일과 UniHand 원천 데이터의 라이선스는 확인하지 않았습니다.

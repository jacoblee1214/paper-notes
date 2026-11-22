# RDT-1B (arXiv:2410.07864, ICLR 2025) 분석 노트

*RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation* (arXiv v2, 2025-03-01)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 비독점 배포 라이선스). 그림은 우리가 그린 그래프 두 장입니다.

## 구현 가능성 (요약)

- **사전학습과 전체 파인튜닝은 이 맥에서 불가입니다.** 48 H100과 CUDA를 전제합니다.
- **추론 경로는 조건부입니다.** 2.46 GB 체크포인트와 언어 임베딩 사전계산이 필요하고, 측정 전입니다.
- **실로봇과 공식 평가 재현은 불가입니다.** 하드웨어와 자체 데이터가 공개되지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그림은 우리가 그린 것만) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사, 주장 판정 |
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

- 논문은 arXiv에 비독점 배포 라이선스(nonexclusive-distrib 1.0)로 공개되어 있습니다. CC BY가 아니므로 논문 그림을 쓰지 않았습니다.
- 공식 코드는 `github.com/thu-ml/RoboticsDiffusionTransformer`에 MIT 라이선스로 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

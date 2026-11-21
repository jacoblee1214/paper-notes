# PhysVAM (arXiv:2611.26105, 학회 표기 없음) 분석 노트

*PhysVAM: Physics-oriented Future Value Action Model for Dynamic Object Manipulation* (arXiv v1, 2026-11-19)을 표 재계산, 통계 검정, 식 검사로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.26105에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- **논문 재현은 이 맥(Apple M4 Pro, CUDA 없음)에서 불가능합니다.** 공식 코드가 공개되지 않았고, 학습은 RTX 5090 4장 48 GPU 시간을 씁니다. 실로봇 장비도 필요합니다.
- **축소 구현은 조건부로 가능합니다.** 작은 동적 도달 토이, 노력 1–2주(추정)입니다. 논문 수치와 비교할 수 없습니다.
- **이 자료에서 확인한 것**: 표 재계산, 표 간 일치, 가치 목표 식의 성질(`analysis/toy_value_target.py`).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 기록 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표, 학회와 라이선스 확인 |
| `analysis/03_analysis.md` | 재계산, 검정, 표 간 일치 검사, 가치 목표 식 검사 |
| `analysis/04_claim_verification.md` | 주장 목록과 판정, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_value_target.py` | 식 (2)–(4) 가치 목표의 성질 검사 (0.2초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_value_target.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 비배타적 배포 라이선스(arXiv non-exclusive distribution license 1.0)로 공개되어 있습니다. CC BY 4.0이 아닙니다. 그래서 논문 그림은 게시하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 검사 스크립트, 분석 문서는 이 자료의 작성자가 만들었습니다.
- 공식 코드는 찾지 못했습니다. 프로젝트 페이지 저장소(MetaRobo/physvam.github.io)는 웹사이트 소스이며 코드가 아닙니다. 이 자료는 코드를 받거나 실행하지 않았습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

# Physics-Guided Residual Dynamics (arXiv:2607.13451, CoRL 2026 Spotlight) 분석 노트

*Learning Physics-Guided Residual Dynamics for Deformable Object Simulation* (v1, 2026-07-15)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 검토한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2607.13451에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- 공식 학습과 재현은 CUDA 전제라 이 맥에서 어렵습니다. 공식 설치에는 하위 모듈 하나가 빠져 있습니다.
- 스프링-질량 백본과 속도 잔차 적분의 CPU 축소 재구현은 가능하다고 봅니다(추정 1–2주, 측정 전).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 빠진 정보 |
| `analysis/02_prior_work.md` | 권리 상태, 공식 저장소 표, 선행 계보 |
| `analysis/03_analysis.md` | 표 재계산, 검정, 본문 주장 대조, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조 (차이 D1–D22), 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문 라이선스는 arXiv non-exclusive(nonexclusive-distrib/1.0)입니다. CC BY가 아니므로 논문 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/shivanshpatel35/pgrd`에 있습니다. LICENSE 파일이 없습니다. 이 저장소에는 코드를 넣지 않았고 실행하거나 설치하지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

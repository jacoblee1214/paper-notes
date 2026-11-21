# MoRE (arXiv:2611.25336, CoRL 2026 Spotlight) 분석 노트

*MoRE: Mixture-of-Experts for Multi-Physics Robotic World Models* (arXiv v1, 2026-11-19)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.25336에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 우리가 논문 표 값으로 그린 그래프 세 장만 넣었습니다.

## 구현 가능성 (요약)

- **이 맥(M4 Pro, MPS, CUDA 없음)에서 논문 규모 재현은 불가능합니다.** 공식 코드는 CUDA가 하드코딩되어 있고, 데이터는 약 10.6 GB이며, 실로봇과 시뮬 MPC가 필요합니다.
- **공식 코드의 라우팅은 테스트에서도 데이터 물성 라벨을 씁니다.** 학습 가능한 게이트는 손실에 들어가지 않아 학습되지 않습니다. 그래서 논문 표 5의 96.7%는 이 코드로 재현되지 않습니다(`analysis/04_code_verification.md` D1–D3).
- **라우팅 설계를 축소판으로 검증하는 것은 가능합니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전과 학회, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표, 그림과 라이선스 판단 |
| `analysis/03_analysis.md` | 표 재계산, 통계 검정, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D17, 라이선스, 구현 가능성 판정 |
| `analysis/recompute.py`, `recompute_results.json` | 표 수치 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 표시는 찾지 못했습니다.
- 논문 그림은 넣지 않았습니다. 그림 가운데 실로봇 사진, DROID 장면, 시뮬 렌더링이 섞여 있기 때문입니다.
- 공식 코드는 `github.com/AlwaySleepy/MoRE-WM`에 있습니다. 기본 브랜치에 LICENSE 파일이 없습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

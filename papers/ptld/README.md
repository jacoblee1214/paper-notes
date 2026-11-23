# PTLD: Sim-to-Real Privileged Tactile Latent Distillation for Dexterous Manipulation (arXiv:2603.04531, CoRL 2026 포스터) 분석 노트

*PTLD* (arXiv v3, 2026-06-18)를 보고 수치 재계산과 일관성 검사, 주장 검증으로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2603.04531에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **논문 시스템 전체는 Apple M4 Pro(CUDA 없음) 맥에서 구현하거나 재현할 수 없습니다.** IsaacGym 학습은 NVIDIA GPU가 필요하고, 증류 데이터에는 Franka, Allegro, Xela 촉각 스킨, 카메라 4대로 된 계측 셀이 필요합니다.
- **공식 코드는 확인되지 않습니다.** 프로젝트 페이지(akashsharma02.github.io/ptld-website)의 코드 링크(github.com/akashsharma02/ptld)가 2026-11-23에 404였습니다.
- 보고 수치 재계산은 `analysis/recompute.py`로 할 수 있습니다(numpy, scipy, 수 초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공개 상태, 학회 확인 |
| `analysis/03_analysis.md` | 재계산 V1-V9, 일관성 검사 K1-K14 |
| `analysis/04_claim_verification.md` | 주장 검증 C1-C14, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 표와 그림 판독 값 재계산 |
| `analysis/fig6_readout.json` | 그림 6 막대 높이의 판독 값(픽셀 측정) |
| `analysis/plot_figures.py` | 자체 그래프 두 장 |
| `report/figures/` | 우리가 그린 그래프 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 라이선스(arXiv non-exclusive distribution 1.0)입니다. CC BY가 아니므로 논문 그림은 넣지 않았습니다.
- 표와 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그림 6의 값은 막대 높이를 읽은 값입니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

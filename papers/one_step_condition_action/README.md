# UCA-Flow (arXiv:2608.16153) 분석 노트

*Unified Condition-Action Modeling for Accurate One-Step Action Generation* (arXiv v5, 2026-10-23)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 라이선스가 CC BY가 아님). 원문은 arXiv:2608.16153에서 받을 수 있습니다.

## 구현 가능성 (요약)

- **핵심 모듈(통합 토큰 열, 이중 손실, 한 스텝 추론)은 PyTorch로 옮길 수 있습니다.** 노력은 약 1주로 추정합니다(측정 전).
- **공식 코드 설치는 이 맥에서 어렵습니다.** Python 3.8, torch 2.2.1 cu118, mujoco-py 2.1과 CUDA 전제입니다.
- **전체 수치 재현과 실로봇 재현은 불가입니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 버전 차이 (v1에서 v5 정정) |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소와 관련 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조 (D1–D21), 라이선스, 구현 가능성 |
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

- 논문은 arXiv에 non-exclusive 라이선스(`arxiv.org/licenses/nonexclusive-distrib/1.0/`)로 공개되어 있습니다. CC BY가 아니므로 논문 그림을 싣지 않았습니다.
- 공식 코드는 `github.com/NTUMARS/UCA-Policy`(MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

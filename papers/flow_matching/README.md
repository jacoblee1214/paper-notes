# Flow Matching for Generative Modeling (arXiv:2210.02747) 분석 노트

*Flow Matching for Generative Modeling* (Lipman 외, arXiv v2 2023-02-08, ICLR 2023)을 표 재계산, 수식 검사, 주장 검증으로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 라이선스가 CC BY가 아님). 원문은 arXiv:2210.02747에서 받을 수 있습니다.

## 구현 가능성 (요약)

- **CFM 목표와 OT 경로, 샘플링 ODE는 PyTorch로 옮길 수 있습니다(반나절 추정).** 수식(식 15, 19, 21, 정리 2)은 `analysis/toy_check.py`로 CPU에서 검사했습니다.
- **논문 표 1 규모 재현은 이 맥에서 불가능합니다.** IN-32 이상은 16–32 GPU, 157k–500k 반복이 필요합니다.
- **공식 코드는 찾지 못했습니다.** 재현을 막는 것은 σ_min 값이 논문에 없다는 점입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 버전 차이 (v1에서 v2) |
| `analysis/02_prior_work.md` | 선행 계보, 학회 확인, 관련 구현 표 |
| `analysis/03_analysis.md` | 재계산, 검정 가능 여부, 표끼리의 일관성, 수식 검사 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C17과 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy 불필요, 표준 라이브러리) |
| `analysis/toy_check.py` | 수식 수치 검사 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 한 장 생성 (matplotlib) |
| `report/figures/` | 우리가 그린 그래프 한 장 |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_check.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive 라이선스(`arxiv.org/licenses/nonexclusive-distrib/1.0/`)로 공개되어 있습니다. CC BY가 아니므로 논문 그림을 싣지 않았습니다.
- 공식 코드는 찾지 못했습니다. 제3자 구현 torchcfm(MIT)과 2024년 라이브러리 facebookresearch/flow_matching(CC BY-NC 4.0)이 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

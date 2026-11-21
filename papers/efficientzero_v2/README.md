# EfficientZero V2 (arXiv:2403.00564, ICML 2024) 분석 노트

*EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data*를 표 값으로 다시 세고, 공식 코드를 정적으로 읽은 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2403.00564에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다. 그래프는 모두 이 저장소 작성자가 그렸습니다.

## 구현 가능성 (요약)

- **공식 코드를 그대로 돌리는 것은 불가능합니다.** CUDA 호출이 많고 환경이 오래되었습니다.
- **재계산과 정적 읽기는 완료했습니다.** 학습, 설치, 실행은 하지 않았습니다.
- **재현은 GPU 자원이 필요합니다.** CPU에서는 1차원 장난감 실험만 가능하며, 아직 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보와 관련 자료 |
| `analysis/03_analysis.md` | 표 재계산, 통계 검정, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 설정 차이, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `analysis/recompute_results.json` | 재계산 결과 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv non-exclusive distribution license입니다(CC 아님). 그래서 논문 그림을 넣지 않았습니다.
- 공식 코드는 GPL-3.0입니다(`github.com/Shengjiewang-Jason/EfficientZeroV2`). 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

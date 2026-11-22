# SimplerEnv (arXiv:2405.05941, CoRL 2024) 분석 노트

*Evaluating Real-World Robot Manipulation Policies in Simulation* (arXiv v1, 2024-05-09; CoRL 2024, PMLR 270)을 표 값 재계산, 구간과 검정, 공식 코드 정적 대조로 분석한 자료입니다. 재현 실험은 하지 않았습니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2405.05941에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **이 맥(CUDA 없음)에서는 공식 평가를 재현할 수 없습니다.** 정책 추론에 NVIDIA GPU가 필요합니다.
- **표 재계산과 통계 검사는 가능하고 끝냈습니다.** `analysis/recompute.py`(numpy, scipy)가 수 초 안에 돕니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 주장과 근거, 학회 판본 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 구간, 검정, 논문 내부 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
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

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 그러나 논문 그림에는 실로봇 사진과 실제 배경이 많아 그림은 싣지 않았습니다.
- 공식 코드는 `github.com/simpler-env/SimplerEnv`(MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 설치하거나 실행하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
- 체크포인트, 데이터, 자산은 받지 않았습니다.

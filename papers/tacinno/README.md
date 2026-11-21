# TacInno (arXiv:2611.26405) 분석 노트

*TacInno: Teaching Vision-Language-Action Models the Contact Dynamics Beyond Vision and Action* (arXiv v1, 2026-11-19)을 표 재계산과 통계 검정으로 분석한 자료입니다. 학회 표기는 없습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.26405에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **공식 코드의 학습과 시뮬레이션 평가는 이 맥에서 불가입니다.** CUDA에 고정되어 있고, 시뮬레이션은 GPU와 Isaac Sim이 필요합니다.
- **재계산과 검정은 가능하고 완료했습니다.** `analysis/recompute.py`(numpy, scipy, 수 초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림은 카메라 장면 썸네일을 포함하므로 넣지 않았습니다. 그래프는 이 저장소 작성자가 표 값에서 다시 그렸습니다.
- 공식 코드는 `github.com/LisavilaLee/TacInno`(Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다.
- 가중치와 데이터셋은 받지 않았습니다.

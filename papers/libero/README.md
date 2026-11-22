# LIBERO (arXiv:2306.03310, NeurIPS 2023 Datasets and Benchmarks) 분석 노트

*LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning* (arXiv v2, 2023-10-14)을 표 재계산, 통계 검정, 코드 정적 대조, 선택 편향 몬테카를로로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2306.03310에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 그래프 두 장은 작성자가 만들었습니다.

## 구현 가능성 (요약)

- **통계 재계산과 검정, 선택 편향 몬테카를로는 CPU에서 수 초 안에 됩니다.** 원문 텍스트(`paper/paper.txt`, arXiv PDF를 `pdftotext -layout`으로 변환)가 필요합니다.
- **시뮬레이터 실행은 조건부입니다.** Python 3.8, robosuite 1.4.0, EGL 렌더링을 전제합니다. 데이터는 suite별로 6–67 GB입니다.
- **학습 재현은 GPU가 필요해 이 분석에서 하지 않았습니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 벤치마크 설계, 프로토콜, 판본 차이 |
| `analysis/02_prior_work.md` | 학회 확인, 선행연구 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 본문 주장의 검정, 선택 편향 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D17, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 파싱, 일관성 검사, Welch 검정 (numpy, scipy) |
| `analysis/toy_max_selection.py` | 최댓값 보고 편향 몬테카를로 (numpy) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py          # paper/paper.txt 필요
python3 analysis/toy_max_selection.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다. PDF 본문에서 저작권 표시를 찾지 못했습니다.
- 공식 코드(Lifelong-Robot-Learning/LIBERO)는 MIT 라이선스입니다. 이 저장소에는 코드를 넣지 않았습니다.
- 데이터 라이선스는 README(CC BY 4.0)와 Hugging Face 카드(apache-2.0)가 다르게 적혀 있습니다. 데이터는 받지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 몬테카를로, 분석 문서와 스크립트는 작성자가 만들었습니다.

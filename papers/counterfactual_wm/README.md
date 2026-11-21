# Counterfactual World Models (arXiv:2611.24778) 분석 노트

*Counterfactual World Models: JEPAs That Reason About "What If"* (arXiv v1, 2026-11-18)를 표 재계산, 비쌍대 검정, 공식 코드 정적 대조, CPU 토이로 분석한 자료입니다. 학회 표기는 없습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.24778에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 우리가 그린 그래프만 있습니다.

## 구현 가능성 (요약)

- **픽셀 세계모델 재현은 이 맥에서 불가능합니다.** 예측기 학습은 CUDA GPU가 필요하고, 전체 재현은 약 150 GPU-시간과 최대 80 GB 디스크가 필요합니다.
- **규칙과 평가 수식은 구현할 수 있습니다.** 공식 코드의 장치 기본값(`cuda`)을 바꾸면 CPU/MPS 추론도 가능할 것으로 보입니다(확인 안 함).
- **CPU 토이는 완료했습니다.** `analysis/toy_interaction.py`(약 3초, numpy).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그래프 세 장) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 이론, 실험 수치 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표, 학회 표기 |
| `analysis/03_analysis.md` | 표 재계산과 검정, 일관성 검사, 토이, 미확인 항목 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 15–17 재계산과 비율 검정 (표준 라이브러리, 수 초) |
| `analysis/toy_interaction.py` | 토이: 감쇠 진자의 상호작용 비율과 전이 길이 (numpy, 약 3초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_interaction.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지의 license 링크 확인). PDF에서 저작권 표시는 찾지 못했습니다.
- 논문 그림은 넣지 않았습니다. 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 코드는 `github.com/galilai-group/counterfactual-WM`(MIT)에 있습니다. 이 저장소에는 넣지 않았고, 실행하거나 설치하지 않았습니다(정적 읽기만).
- 가중치와 데이터셋은 받지 않았습니다. MAE 가중치는 CC-BY-NC 4.0입니다.

# Doppel (arXiv:2611.24925) 분석 노트

*Render the Robot, Generate the World: One Video World Model Across Embodiments* (arXiv v1, 2026-11-18)을 표 재계산과 공식 코드 정적 대조로 분석한 자료입니다. 학회 표기는 없습니다(arXiv 전용).

## 구현 가능성 (요약)

- 이 맥에서 재현은 불가능합니다. 학습은 H100 8–32장, 추론은 80 GB GPU, 카메라 보정은 CUDA를 전제로 합니다.
- 공식 코드는 실행하지 않았습니다. 정적 읽기로 논문과 대조했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 결과, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 코드와 가중치 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 논문과 공식 코드 대조 |
| `analysis/recompute.py` | 표 재계산 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 논문 PDF는 넣지 않았습니다(arXiv:2611.24925에서 받을 수 있습니다).
- 논문 그림은 넣지 않았습니다. 그래프는 표 값으로 우리가 그렸습니다.
- 공식 코드는 `github.com/DoppelWM/Doppel` (MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았습니다. 가중치는 CC BY-NC 4.0입니다.
- 가중치와 데이터셋은 받지 않았습니다.

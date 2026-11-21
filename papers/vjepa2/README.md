# V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (arXiv:2506.09985) 분석 노트

V-JEPA 2 (arXiv v1, 2025-06-11)를 표 재계산, 공식 코드 정적 대조로 분석한 자료입니다. 학회 표기는 없습니다(arXiv). 논문 원문과 논문 그림은 넣지 않았습니다. 이 폴더의 그래프 한 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **로봇 결과(표 2–3)는 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서도, 공개 코드만으로도 재현할 수 없습니다.** 공개 코드에는 로봇 인터페이스가 없습니다.
- 모델 구조와 계획 알고리즘은 공개 코드로 재구성할 수 있습니다. 학습에는 GPU가 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 방법, 실험, 주장과 증거의 틈 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 자료와 GitHub 표 |
| `analysis/03_analysis.md` | 표 재계산, Wilson 구간, Fisher 검정, 일관성 검사 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 차이 D1–D10 |
| `analysis/recompute_results.py` | 표 재계산 (표준 라이브러리, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 생성 (matplotlib) |

## 재실행

```bash
python3 analysis/recompute_results.py
pip install matplotlib
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(abs 페이지, 2026-11-21 확인). PDF에는 저작권 고지가 없습니다. 그래도 논문 그림은 넣지 않았습니다.
- 공식 코드(https://github.com/facebookresearch/vjepa2)는 MIT 라이선스입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값은 재계산과 인용을 위해 옮겼습니다. 그래프, 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

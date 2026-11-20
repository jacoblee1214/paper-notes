# WorldVLA: Towards Autoregressive Action World Model (arXiv:2506.21539) 분석 노트

*WorldVLA* (arXiv v1, 2025-06-26)를 표 재계산, 공식 코드(`worldvla` 브랜치) 정적 대조, 작은 numpy toy로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 CC BY-NC-SA 4.0). 이 폴더의 그래프 두 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 전체 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 7B 백본을 공식 설정(GPU 2장, 40 에포크)으로 학습하는 것이 필요하고, 학습 시간은 논문과 코드에 없습니다.
- 맥에서 가능한 것은 표 재계산, 마스크 단위 검증, 행동 이산화 오차 toy, 검출력 계산입니다. 공개 가중치 1과제 평가는 조건부입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, 원문 내부 불일치 |
| `analysis/02_prior_work.md` | 학회 표기 확인, 코드와 라이선스 표 |
| `analysis/03_analysis.md` | 표 재계산, 통계적 불확실성, 본문 주장 대조, 검출력 toy |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 차이(D1–D14) |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, scipy, 1초 미만) |
| `analysis/toy_power.py` | 평균 성공률 차이의 검출력 모의실험 (numpy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_power.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에서 CC BY-NC-SA 4.0으로 공개되어 있습니다(abs 페이지 확인). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/alibaba-damo-academy/WorldVLA, `worldvla` 브랜치)는 Apache 2.0입니다. 이 분석에서는 실행하거나 설치하지 않고 정적으로만 읽었습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, toy, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

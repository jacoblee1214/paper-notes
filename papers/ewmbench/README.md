# EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models (arXiv:2505.09694, BMVC 2025) 분석 노트

*EWMBench* (arXiv v2, 2025-05-18)를 표 재계산, 공식 코드 정적 대조, 불확실성 상한으로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다. 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **표 2 전체 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 캡션 모델(Qwen2.5-VL-7B)과 검출 경로가 CUDA로 고정되어 있고, 영상 2,100개의 캡션 생성이 필요합니다. 학습 코드와 사람 평가 자료는 공개되지 않았습니다.
- **궤적 지표(HSD, NDTW, DYN)의 CPU 재구현은 가능합니다.** 샘플 궤적 데이터(전체 약 270 MB)를 받아야 합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(합계, Logics 분모, 순위 tau, 표준오차 상한), 일관성 검사 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 불일치와 빈칸, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (표준 라이브러리, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). PDF에 저작권 고지가 없습니다. 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/AgibotTech/EWMBench)와 데이터, 가중치는 CC BY-NC-SA 4.0입니다. 저장소에 LICENSE 파일이 없습니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프와 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

# GR-1: Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation (arXiv:2312.13139, ICLR 2024) 분석 노트

GR-1 논문 (arXiv v2, 2023-12-21)을 표 재계산, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다. 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 공식 코드에는 사전학습과 파인튜닝 코드가 없고, 평가 코드는 CUDA와 Linux 렌더링을 요구합니다.
- **모델 구조 옮기기는 가능합니다.** 최신 PyTorch로 구조를 다시 짜고 체크포인트 출력과 맞추는 작업입니다(노력 2–4일, 추정).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거, 권리 검사, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, 공식 코드와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(평균 길이 검산, 격자, 유의성, 실로봇 시도 수), 작업별 표 7 검사 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 빈칸과 불일치, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산 (표준 라이브러리) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
python3 analysis/recompute.py
python3 analysis/plot_figures.py   # pip install matplotlib
```

## 출처와 라이선스

- 논문은 arXiv abs 페이지의 "view license" 링크 기준 CC BY 4.0입니다(2026-11-21 확인). PDF에는 저작권 표시가 없습니다. 논문 원문과 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/bytedance/GR-1)는 Apache-2.0입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

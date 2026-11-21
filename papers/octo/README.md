# Octo: An Open-Source Generalist Robot Policy (arXiv:2405.12213, RSS 2024) 분석 노트

*Octo* (arXiv v2, 2024-05-26)를 표 재계산, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다. 이 폴더의 그래프 두 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **추론과 구조 검산은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 가능합니다(조건부).** 공식 코드는 JAX 기반입니다.
- **미세조정, 사전학습, 실로봇 평가와 논문 수치 재현은 불가능합니다.** 논문은 A5000 한 장에서 Base 미세조정에 약 5시간, TPU v4-128에서 사전학습에 약 14시간을 썼습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표, 라이선스 |
| `analysis/03_analysis.md` | 표 I, II/VI, VII 재계산과 검정, 일관성 검사 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조(일치와 차이 D1–D14), 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 Fisher 검정, Wilson 구간 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(abs 페이지 확인, 2026-11-21). 그래도 이 자료에는 논문 그림을 넣지 않았습니다.
- 공식 코드(https://github.com/octo-models/octo)는 MIT 라이선스입니다. 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 체크포인트와 데이터셋은 받지 않았습니다.

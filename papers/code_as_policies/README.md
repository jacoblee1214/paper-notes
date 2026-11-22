# Code as Policies: Language Model Programs for Embodied Control (arXiv:2209.07753, ICRA 2023) 분석 노트

*Code as Policies* (arXiv v4, 2023-05-25)를 표 재계산, 공식 노트북의 정적 대조, 구현 가능성으로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 두 장은 우리가 표 값으로 그린 것입니다.

## 구현 가능성 (요약)

- **방법은 이 맥(Apple M4 Pro, 24 GiB, MPS, CUDA 없음)에서 구현할 수 있습니다.** 오픈 코드 LLM으로 RoboCodeGen 37문항을 다시 재는 실험은 1–2일(추정)입니다.
- **논문 수치는 재현할 수 없습니다.** 원래 모델은 OpenAI API이고, CLIPort 학습은 GPU가 필요하며, 실로봇 코드는 공개되지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 주장과 증거의 틈, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산과 일관성 검사, 과제군 평균, 검정 |
| `analysis/04_code_verification.md` | 논문과 공식 노트북의 대조, 차이 목록, 구현 가능성, 저자에게 물을 것 |
| `analysis/recompute.py` | 표 재계산과 노트북 저장 출력 대조 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(google-research 모노레포의 `code_as_policies/`, Apache-2.0)는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

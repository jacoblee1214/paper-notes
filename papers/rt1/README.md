# RT-1: Robotics Transformer for Real-World Control at Scale (arXiv:2212.06817, RSS 2023) 분석 노트

*RT-1* (arXiv v2, 2023-08-11; RSS 2023 공식 proceedings)의 보고 수치를 재계산하고, 논문과 공식 코드를 정적으로 대조한 자료입니다. 학습과 추론은 하지 않았습니다. 논문 원문과 논문 그림은 넣지 않았습니다(arXiv 라이선스 nonexclusive-distrib 1.0). 이 폴더의 그래프 세 장은 보고된 수치로 우리가 그린 것입니다.

## 구현 가능성 (요약)

- **RT-1 학습은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 재현할 수 없습니다.** GPU와 대규모 로봇 데이터, 실로봇 평가 장비가 필요합니다.
- **구조는 PyTorch로 다시 짜서 형태 검사까지 맥에서 가능합니다.** 공식 체크포인트 추론은 TensorFlow 설치 후 CPU로 가능할 것으로 추정합니다(미확인). 노력은 수 주로 추정합니다(측정하지 않음).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계, arXiv 판본 비교 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 자료 표 |
| `analysis/03_analysis.md` | 수치 재계산(분모 역산, 검정, 표 간 일관성), 그래프 |
| `analysis/04_code_verification.md` | 논문과 공식 코드 대조 (D1–D16), 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 재계산과 일관성 검사 (표준 라이브러리) |
| `analysis/recompute_results.json` | 재계산 결과 |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv:2212.06817 (abs 페이지의 라이선스는 nonexclusive-distrib 1.0, 2026-11-22 확인). RSS 2023 공식 proceedings 목록(roboticsproceedings.org/rss19/p025.html)에서 제목과 저자를 확인했습니다.
- 공식 코드: https://github.com/google-research/robotics_transformer (Apache-2.0). 정적 읽기만 했습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

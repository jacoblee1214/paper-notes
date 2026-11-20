# DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos (arXiv:2602.06949) 분석 노트

*DreamDojo* (arXiv v1, 2026-02-06)를 표 재계산, 공식 코드 정적 대조, 본문과 표의 일관성 검사로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(PDF의 저작권 표시). 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **학습과 수치 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 사전학습은 H100 256장에 140k 스텝이고, 사람 영상 사전학습 데이터는 공개되지 않았습니다.
- **2B 추론은 조건부로 가능합니다.** 저장소는 H100과 CUDA에서 시험되었습니다. MPS 포팅이 필요하고 메모리와 시간은 측정하지 않았습니다(추정 3–5일). 14B는 메모리상 어렵습니다(추정).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 주장과 증거의 틈, 학회와 버전 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 표 |
| `analysis/03_analysis.md` | 표 재계산(승률 구간, 정책 평가 상관, 탐지 한계, 속도 단위), 본문과 표의 일관성 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 차이 목록, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (표준 라이브러리) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(abs 페이지 확인, 2026-11-20). 그러나 PDF 1쪽에 저작권 표시("© 2026 NVIDIA. All rights reserved.")가 있어 논문 그림은 넣지 않았습니다.
- 공식 코드(https://github.com/NVIDIA/DreamDojo)는 Apache-2.0입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 학회는 ICML 2026입니다. 채택판은 제목을 바꾼 개정판이며, 이 분석은 arXiv v1 기준입니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

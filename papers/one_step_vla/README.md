# When One Step Is Enough (arXiv:2610.30735, CoRL 2026) 분석 노트

*When One Step Is Enough: An Empirical Study of Euler Integration in Flow-Matching Policies* (arXiv v1, 2026-10-22)을 표 재계산, 독립 근사 검정, 정확한 속도장 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **수학과 평가 프로토콜은 구현할 수 있습니다.** 짝 평가 파이프라인은 PyTorch로 300–500줄, 3–5일(추정)입니다.
- **논문 수치 재현은 이 맥에서 불가합니다.** LIBERO 평가와 VLA 학습은 CUDA GPU와 시뮬레이터가 필요합니다. 공식 코드는 확인되지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계, 본문 불일치 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 표 |
| `analysis/03_analysis.md` | 수치 재계산, 독립 근사 검정, McNemar 최소 p, 토이 결과, 불일치 목록 |
| `analysis/04_claim_verification.md` | 불일치 항목, 주장 검증표, 구현 가능성, 저자에게 물을 질문 |
| `analysis/table_check.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_one_step_gmm.py` | 정확한 속도장 토이 (numpy, 약 1분) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/toy_one_step_gmm.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다. 논문 그림과 원문 PDF는 이 저장소에 넣지 않았습니다. 원문은 arXiv:2610.30735에서 받을 수 있습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 데이터셋과 체크포인트는 받지 않았습니다.

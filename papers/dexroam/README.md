# DexRoam: Learning Mobile Bimanual Dexterous Manipulation from Egocentric Whole-Body Human Demonstrations (arXiv:2609.35761, CoRL 2026) 분석 노트

*DexRoam* (arXiv v1, 2026-09-28)을 표 재계산, 공식 코드 두 저장소 정적 대조, 정렬 지표 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 다섯 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 수집은 Quest 3와 ZED Mini, 학습은 CUDA GPU, 평가는 Astribot + XHand 로봇이 필요합니다.
- 정렬의 기하와 시간 재표집, 분포 지표의 합성 검증은 CPU로 가능합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표·그림 재계산 V1–V15, 검정, 지표 토이 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조 D1–D16, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy). V15는 공식 코드 clone을 읽음 |
| `analysis/toy_alignment_metric.py` | 토이: 정렬 지표 눈금 (numpy, scipy, 약 15초) |
| `analysis/plot_figures.py` | 그래프 다섯 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/toy_alignment_metric.py
# recompute.py의 V15는 정책 학습 저장소 clone이 있어야 한다(없으면 건너뜀):
#   git clone --depth 1 https://github.com/zhourui9813/DexRoam-Policy-Training.git repo_policy
#   git -C repo_policy checkout a82d2c98dc0dff00c17e1aa4fb2bb93d1c5655fe
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-26). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/zhourui9813/DexRoam, https://github.com/zhourui9813/DexRoam-Policy-Training, 모두 MIT)는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

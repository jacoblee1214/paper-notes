# DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation (arXiv:2210.02697, IEEE ICRA 2023) 분석 노트

*DexGraspNet* (arXiv v2, 2023-03-08)을 표 재계산, 공식 코드 정적 대조, 계산 비용 분해로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(라이선스 표시가 CC BY 4.0과 CC BY-NC 4.0으로 엇갈립니다). 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **공식 파이프라인은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 돌지 않습니다.** 생성 코드에 `cuda`가 하드코딩되어 있고, 시뮬 검증은 NVIDIA GPU와 Linux의 Isaac Gym이 필요합니다. 논문 설정으로 1.32M 파지를 만드는 데는 A100 약 950 GPU 시간이 듭니다.
- **에너지 식과 최적화는 torch로 옮길 수 있고, 검증은 MuJoCo로 바꿀 수 있습니다.** 노력은 2–4주(추정)입니다. 물리 엔진이 달라서 성공률 수치는 논문과 비교할 수 없습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, v1과 v2의 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산 (물체 수, 방향 검사, 엔트로피 단위, 유의성 최소 n, 계산 비용 분해) |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 차이 D1–D15, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문 arXiv 페이지의 라이선스 링크는 CC BY 4.0입니다(2026-11-22 확인). 프로젝트 페이지와 공식 저장소 README는 "이 작업과 데이터셋은 CC BY-NC 4.0"이라고 씁니다. 두 표시가 충돌하므로 논문 그림은 넣지 않았습니다.
- 공식 코드(https://github.com/PKU-EPIC/DexGraspNet)는 저장소에 LICENSE 파일이 없습니다. 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 데이터셋과 모델 가중치는 받지 않았습니다.

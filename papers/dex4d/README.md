# Dex4D: Task-Agnostic Point Track Policy for Zero-Shot Dexterous Manipulation (arXiv:2602.15828, CoRL 2026) 분석 노트

*Dex4D* (arXiv v2, 2026-10-13)를 표 재계산, 공식 코드 정적 대조, 작은 토이 두 개로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 네 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 학습은 Isaac Gym Preview와 NVIDIA GPU를 전제로 하고, 실로봇·평가·기준선 코드는 공개되지 않았습니다.
- 영상 → 점 궤적 부분은 조건부, 점 쌍 인코더와 증류 손실의 합성 검증은 CPU로 가능합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산 V1–V14, 검정, 토이 1과 2 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조 D1–D14, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy). V12는 공식 코드 clone을 읽음 |
| `analysis/toy_pair_encoding.py` | 토이 1: 점 인코딩 (torch CPU, 약 100초) |
| `analysis/toy_kabsch.py` | 토이 2: Kabsch 민감도 (numpy, 약 25초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/toy_kabsch.py
pip install torch                       # 토이 1에만 필요
python3 analysis/toy_pair_encoding.py
python3 analysis/plot_figures.py
# recompute.py의 V12는 공식 저장소 clone이 있어야 한다(없으면 건너뜀):
#   git clone --depth 1 https://github.com/Dex4D/Dex4D-Simulation.git repo/simulation
#   git -C repo/simulation checkout 0dfccd82797e929cda6ab665003bc27a6b8cd31a
python3 analysis/recompute.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-26). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/Dex4D/Dex4D-Simulation, Dex4D-Vision은 Apache-2.0, Dex4D-Hardware는 라이선스 없음)는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

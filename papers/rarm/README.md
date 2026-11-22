# RARM: Confidence-Gated Progress Reward Modeling for RL in Manipulation (arXiv:2606.22027, CoRL 2026) 분석 노트

*RARM* (arXiv v5, 2026-10-12)를 표 재계산, 공식 코드 정적 대조, 작은 numpy 토이와 무작위 가중치 시간 측정으로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 네 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 공개 정책 학습은 Linux, CUDA, EGL을 전제로 하고 보상 계산만 한 번의 실행에서 약 13시간(추정)입니다. 보상 모델 학습은 이 기기에서 45에폭 약 6.5–9.8일(추정)이고 학습 영상 목록이 공개되어 있지 않습니다. 실로봇, 표 2, MetaWorld, 기준선은 코드가 공개되지 않았습니다.
- **공개 체크포인트의 유사도 추론은 조건부**(새 설치, 가중치 다운로드 필요), **축소판(자작 MuJoCo 과제, 소형 비교기)은 구현할 수 있습니다**(노력 1–2주 추정).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, 버전 차이 v1–v5 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 1·2·그림 5·부록 재계산, 게이트와 보상의 성질, 토이 2개, 질의·학습 비용 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조(차이 D1–D15), 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사, 비용 추정 (numpy, scipy, 1초 미만) |
| `analysis/toy_reward_scale.py` | 토이 1: 공식 코드의 보상 배치가 만드는 크기 (numpy, 1초 미만) |
| `analysis/toy_gate.py` | 토이 2: 게이트 Δ의 민감도 (numpy, 수 초) |
| `analysis/bench_mps.py` | 무작위 가중치 ViT-B와 비교기의 질의 시간 측정 (torch, MPS가 있으면 약 1분) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_reward_scale.py
python3 analysis/toy_gate.py
pip install torch                       # 시간 측정에만 필요
python3 analysis/bench_mps.py
python3 analysis/recompute.py           # bench 결과를 읽어 비용 추정을 갱신
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-23). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/pengzhi1998/rarm)에는 라이선스 파일이 없습니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다. 체크포인트는 DINOv3 라이선스로 표기되어 있습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

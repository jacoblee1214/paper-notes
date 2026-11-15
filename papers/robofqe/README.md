# RoboFQE: Rollout-Free Evaluation of Visual Robot Policies from Offline Trajectories (arXiv:2610.18699, CoRL 2026) 분석 노트

*RoboFQE* (arXiv v2, 2026-11-05)를 표 재계산, 공식 코드 정적 대조, 작은 numpy 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** LIBERO 데이터 로더는 이미지를 RAM에 올려 Spatial 10 GB, Long 23 GB가 필요합니다. 인코더 한 스텝은 MPS에서 약 3.2초이고, 50k 스텝은 약 45시간입니다. LRM 검출기(8B VLM), 세계모델, CALVIN, 실로봇도 맥에서는 어렵거나 불가능합니다.
- **MuJoCo 축소판은 조건부로 가능합니다.** 새 설치 없이 시작할 수 있습니다(처리량 미측정, 노력 1–2주 추정).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(ARE–MSE 상한, 순위 n, Welch, 백분율 가능성, 메모리), 토이 1과 2, 맥 속도 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 불일치와 빈칸, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, scipy, 1초 미만) |
| `analysis/toy_gt_noise.py` | 토이 1: 정답 200궤적 노이즈 바닥과 순위 천장 (numpy, 1초 미만) |
| `analysis/toy_label_noise.py` | 토이 2: 종료 라벨 오류의 편향과 순위 (numpy, 1초 미만) |
| `analysis/bench_vit_step.py` | ViT-S 구조의 한 스텝 시간 측정 (torch, 랜덤 가중치, MPS가 있으면 약 30초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_gt_noise.py
python3 analysis/toy_label_noise.py
python3 analysis/plot_figures.py
pip install torch                       # 속도 측정에만 필요
python3 analysis/bench_vit_step.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-15). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/prediction-action-lab/RoboFQE)는 MIT 라이선스입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

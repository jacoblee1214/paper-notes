# RL-Co: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models (arXiv:2602.12628, CoRL 2026) 분석 노트

*Beyond Imitation: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models* (arXiv v5, 2026-10-22)을 표 재계산, 공식 코드 정적 대조, 작은 numpy 토이로 분석한 자료입니다. 논문 원문과 사진류 그림은 넣지 않았습니다. 논문 차트 세 장은 리포트에만 있고, 이 폴더에는 우리가 만든 그래프만 있습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, CUDA 없음)에서 불가능합니다.** 공식 π0.5 Stage II는 Linux와 CUDA, GPU 시뮬을 씁니다. OpenVLA 공동 학습 코드는 공개되지 않았고, 실로봇에는 하드웨어가 필요합니다.
- **두 단계 손실 결합(식 2)의 축소판은 조건부로 구현할 수 있습니다.** numpy 토이는 완료했습니다. 접촉이 있는 MuJoCo 과제로 넓히는 것은 새 설치 없이 가능하지만 처리량은 측정하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이, 한계, 주장과 실험의 틈 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 Hugging Face 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(정수 쌍, Avg 정의, Fisher, Holm), 일관성 검사, 토이 A와 토이 B |
| `analysis/04_code_verification.md` | 공식 RLinf 저장소와 논문의 대조, 불일치 목록, 구현 가능성 |
| `analysis/recompute.py` | 표와 그림 값의 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_mixing_weight.py` | 토이 A: 실데이터 앵커 가중치 β의 최적값 (numpy, 1초 미만) |
| `analysis/toy_eval_design.py` | 토이 B: 1 체크포인트 x 2 세션 평가 설계의 한계 (numpy, scipy, 약 20초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_mixing_weight.py
python3 analysis/toy_eval_design.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(abs 페이지 확인, 2026-11-15). 리포트에는 논문 차트 세 장(Fig. 4, 5c, 6)을 잘라 넣고 `© 저자, CC BY 4.0, 출처 arXiv:2602.12628`을 달았습니다. 원문 PDF는 넣지 않았습니다.
- 공식 코드(https://github.com/RLinf/RLinf)는 Apache-2.0 라이선스입니다. 이 저장소에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

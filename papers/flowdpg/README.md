# FlowDPG: Deterministic Policy Gradient on Flow Matching Policies for Real-World Manipulation (arXiv:2606.22303, CoRL 2026) 분석 노트

*FlowDPG* (arXiv v2, 2026-09-30)를 보고 수치 재계산, 갱신식의 정상점 유도, 작은 CPU 토이로 분석한 자료입니다. 공식 코드가 없어서 코드 대조 대신 주장 검증(C1–C12)을 썼습니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 결과의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 양팔 실로봇, 시연, 보상 모델이 필요하고 코드, 데이터, 가중치가 없습니다.
- **알고리즘 구현은 가능합니다.** PyTorch 300–500줄, 3–5일(추정). 작은 시뮬레이션 축소 검증은 1–2주(추정)입니다.
- 행동 머리 규모(176 M)는 MPS에서 한 스텝 0.67초(배치 64)로 느리고, 비전 백본 합동 학습은 24 GiB에서 현실적이지 않습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(V1–V5), 정상점 유도와 토이(K1–K4) |
| `analysis/04_claim_verification.md` | 주장 검증(C1–C12), 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산, k/n 가능성, Fisher·Wilson, 검정력 (numpy, scipy, 약 5초) |
| `analysis/toy_dpg.py` | 정상점, 민감도, 방법 비교, 기울기 잡음 토이 (torch, CPU 약 1분) |
| `analysis/bench_dit_step.py` | 행동 머리 한 스텝 시간 측정 (torch, 랜덤 가중치, MPS가 있으면 약 10초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
pip install torch                       # 토이와 속도 측정에만 필요
python3 analysis/toy_dpg.py
python3 analysis/bench_dit_step.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 넣지 않았습니다.
- 공식 코드는 없습니다. 제3자 구현 하나(`LikhithKumar555/continual-flowdpg-libero`)는 README만 확인했고 가져오지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

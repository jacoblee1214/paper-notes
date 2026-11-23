# TactileStep: Sole Tactile Learning for Regulating Foot-Terrain Interaction in Humanoid Locomotion (arXiv:2609.28959, CoRL 2026 스포트라이트) 분석 노트

*TactileStep* (arXiv v1, 2026-09-24)을 표 재계산, 통계 검정, 촉각 시뮬레이터 numpy 재구현으로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 학습은 Isaac Sim과 RTX 4090 급 GPU를 전제하고, 실험은 Unitree G1과 압력 인솔이 필요합니다. 공식 코드는 아직 공개되지 않았습니다(프로젝트 페이지 "Code Soon").
- **촉각 시뮬레이터(식 3-7)의 numpy 재구현은 가능하고 완료했습니다.** 논문이 주지 않은 값은 가정으로 두었습니다.
- **MuJoCo 축소판(이중 비평가, 촉각 관측 절제)은 조건부로 가능합니다.** 노력은 2-4주(추정, 측정 전)입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(Welch, Holm, 정수 가능성, 시뮬 대 실물), 토이 |
| `analysis/04_claim_verification.md` | 주장 C1-C14 판정, 본문 식과 부록 표 대조, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_tactile_sim.py` | 토이: 촉각 시뮬 식 3-7 재구현과 민감도 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_tactile_sim.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-23). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드는 공개되지 않았습니다. 프로젝트 페이지는 https://tactilestep.github.io/ 입니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그림 6의 값은 그림에 적힌 숫자를 읽은 값입니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

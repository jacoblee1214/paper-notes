# TacCoRL (arXiv:2606.11743, CoRL 2026 포스터) 분석 노트

*TacCoRL: Integrating Tactile Feedback into VLA via Simulation* (arXiv v1, 2026-06-10)을 표 재계산과 통계 검정으로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.11743에서 받을 수 있습니다). 논문은 arXiv 비독점 배포 라이선스라서 논문 그림은 넣지 않았고, 그래프는 모두 이 저장소 작성자가 그렸습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 불가능합니다.** π0.5 학습과 RLinf PPO는 GPU, 실로봇은 PiPER 두 대와 촉각 패드가 필요합니다. 공식 코드는 공개되어 있지 않습니다.
- **구성 요소의 축소판은 작은 합성 환경에서 구현할 수 있습니다.** 접촉 게이트와 촉각 토큰, SPSA 컨트롤러 식별, 앵커 있는 PPO입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 방법, 실험, 판본·학회·라이선스, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, GitHub 표 |
| `analysis/03_analysis.md` | 표 재계산과 검정 V1–V11, 일관성 검사 K1–K9 |
| `analysis/04_claim_verification.md` | 주장 C1–C14와 구현 가능성 판정 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문의 arXiv 라이선스는 비독점 배포(`http://arxiv.org/licenses/nonexclusive-distrib/1.0/`)이고 CC BY가 아닙니다. 그래서 논문 그림은 이 저장소에 없습니다. 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다.
- 프로젝트 페이지는 https://tac-corl.github.io/ 입니다. 공식 코드는 찾지 못했습니다.
- 가중치와 데이터셋은 받지 않았습니다. 분석 문서, 그래프, 스크립트는 이 저장소 작성자가 만들었습니다.

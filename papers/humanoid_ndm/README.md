# HumanoidNDM (arXiv:2610.34126, CoRL 2026 포스터) 분석 노트

*HumanoidNDM: Closing the Training-Deployment Gap of Motion Tracking via Humanoid Neural Dynamics Model* (arXiv v1, 2026-10-25)을 표 재계산, 통계 검정, 토이 두 개로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2610.34126에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **실로봇 재현(Unitree G1)은 이 환경(Apple M4 Pro, CUDA 없음)에서 불가입니다.**
- **공식 코드가 없습니다.** 프로젝트 페이지의 코드 링크가 가리키는 저장소는 2026-11-24 기준 비어 있습니다.
- **시뮬 축소판(MuJoCo CPU와 PyTorch)은 구현할 수 있고, NDM 학습과 모델 기울기 정책 갱신의 계산량은 같은 자릿수로 돕니다.** 토이 `analysis/toy_ndm_cost.py`로 확인했습니다(CPU 7.1 s, MPS 1.8 s 추정, 정책 구조 등은 가정).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 판본 |
| `analysis/02_prior_work.md` | 선행 계보, 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V15, 일관성 K1–K18, 토이 |
| `analysis/04_claim_verification.md` | 주장 점검 C1–C18, 구현 가능성 |
| `analysis/table_data.py` | 논문 표 값을 옮긴 데이터 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_fence.py` | 토이: 1자유도 링크의 탐색 방식별 흔들림과 파라미터 식별 (numpy, 수 초) |
| `analysis/toy_ndm_cost.py` | 토이: 부록 A.5 NDM 구조의 계산 시간 (PyTorch, 약 10초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_fence.py
python3 analysis/toy_ndm_cost.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스(CC 라이선스 아님)로 공개되어 있습니다(arXiv 페이지의 license 링크 확인). 그래서 논문 그림은 넣지 않았고, 표 값과 본문 수치만 인용과 재계산을 위해 옮겼습니다.
- 프로젝트 페이지는 https://pku-epic.github.io/HumanoidNDM/ 입니다. 코드 링크가 가리키는 GitHub 저장소는 비어 있었습니다.
- 그래프, 토이, 분석 문서와 스크립트는 이 자료 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았고, 외부 코드를 실행하지 않았습니다.

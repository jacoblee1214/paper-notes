# OmniTacTune (arXiv:2607.03723, CoRL 2026 포스터) 분석 노트

*OmniTacTune: Policy-Agnostic Real-World RL for Tactile Residual Adaptation of Visual Policies* (arXiv v1, 2026-07-04)를 표 재계산과 통계 검정, 본문 주장 대조로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다. 공식 코드는 찾지 못했습니다.

## 구현 가능성 (요약)

- 실로봇 실험, flow 생성기와 촉각 증강 모델 학습은 일반 노트북(Apple M4 Pro 24 GiB 기준)에서 불가입니다.
- 방법의 골격(웜스타트 critic 부트스트랩, 접촉 게이트, 크기 스케줄 잔차 SAC)을 소형 시뮬레이션 과제로 옮기는 것은 조건부로 가능합니다(추정, 측정 전). 표 재계산은 끝냈습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 실험의 틈, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, GitHub 표 |
| `analysis/03_analysis.md` | 수치 재계산 V1-V17, 불일치 K1-K8 |
| `analysis/04_claim_verification.md` | 주장 C1-C12 판정, 빠진 정보, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 비독점 배포 라이선스로 공개되어 있습니다(CC BY 아님). 논문 그림과 원문 PDF는 이 저장소에 넣지 않았습니다. 원문은 arXiv:2607.03723에서 받을 수 있습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼고, 그림 4, 6, 15-18의 값은 그림에서 읽은 값입니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 데이터셋과 체크포인트는 받지 않았습니다.

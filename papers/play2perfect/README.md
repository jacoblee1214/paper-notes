# Play2Perfect (arXiv:2606.26428, CoRL 2026) 분석 노트

*Play2Perfect: What Matters in Dexterous Play Pretraining for Precise Assembly?* (arXiv v4, 2026-10-15)를 표 재계산, 신뢰구간, 판본 대조, 공식 코드(MIT, 커밋 f67ef45) 정적 대조, 목표 허용오차 기하 toy로 분석한 자료입니다.
논문 원문 PDF는 넣지 않았습니다(arXiv:2606.26428에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다(arXiv 비독점 라이선스).

## 구현 가능성 (요약)

- **공식 재현은 불가능합니다.** Isaac Sim은 NVIDIA RTX GPU가 필요하고 사전학습은 7일입니다. 실로봇 재현에는 KUKA iiwa 14, Sharpa 손, 외부 자세 추적 fork가 필요합니다.
- **축소 재구현은 조건부 가능합니다.** 평행 턱 그리퍼나 소형 손 모델로 6D 목표 보상을 CPU·MPS에서 비교하는 것은 가능합니다.
- 기하 toy(`analysis/toy_goal_tolerance.py`)는 numpy만으로 1초 안에 돕니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 판본 차이 |
| `analysis/02_prior_work.md` | 선행 연구 계보와 대안 |
| `analysis/03_analysis.md` | 재계산, 신뢰구간, Fisher 민감도, 33배 산술, 허용각 toy 결과 |
| `analysis/04_code_verification.md` | 논문 ↔ 공식 코드 차이 D1–D25와 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 1초 안팎) |
| `analysis/toy_goal_tolerance.py` | 목표 허용오차가 허용하는 회전각 (numpy) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 직접 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_goal_tolerance.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv:2606.26428 (arXiv 비독점 배포 라이선스). CoRL 2026 구두 발표 O5 및 포스터 #126 (corl.org 목록 확인).
- 공식 코드: https://github.com/kushal2000/play2perfect (MIT). 이 저장소에는 코드를 넣지 않았고, 실행·설치·import는 하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, toy, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

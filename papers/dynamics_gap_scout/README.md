# Scouting the Dynamics Gap (arXiv:2609.36107, CoRL 2026 포스터) 분석 노트

*Scouting the Dynamics Gap: Test-Time Policy Adaptation via Action-Outcome Feedback* (arXiv v1, 2026-09-28)을 표 재계산, 근사 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2609.36107에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- **구조 재구현은 가능합니다.** 작은 규모는 CPU나 MPS로 돌릴 수 있습니다.
- **학습 재현은 이 맥에서 하지 않습니다.** CUDA GPU가 필요합니다.
- **공식 체크포인트 평가는 조건부입니다.** Linux 환경과 MuJoCo 3 포크가 필요합니다.
- **실로봇 재현은 불가능합니다.** 코드와 제어 설정이 없습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 근사 검정, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 근사 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스(`nonexclusive-distrib/1.0`)로 공개되어 있습니다. CC BY가 아니므로 논문 그림은 쓰지 않았습니다.
- 공식 코드는 `github.com/liy1shu/SCOUT`에 있습니다. 루트 LICENSE 파일이 없습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

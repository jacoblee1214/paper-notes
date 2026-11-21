# Scaling Sim-to-Real VLA RL with Generative 3D Worlds (arXiv:2603.18532) 분석 노트

CoRL 2026 포스터 논문을 표 재계산과 통계 검정으로 읽은 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv에서 받을 수 있습니다). 그래프는 논문 표의 값으로 그렸고, 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **재현은 CUDA GPU와 로봇이 필요해서 불가능합니다.** 학습은 8장 5일, 장면 생성은 장면당 약 47분(RTX 4090), 실물 평가는 로봇이 필요합니다.
- **재계산과 통계 검사는 CPU로 끝납니다.** `analysis/recompute.py`(수 초)와 `analysis/plot_figures.py`로 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그래프 세 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 관련 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 대조, 공식 코드 조사, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 고지를 찾지 못했습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
- 공식 코드는 찾지 못했습니다. 프로젝트 페이지의 GitHub 저장소(HorizonRobotics/gail)는 웹 페이지 소스뿐입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다.

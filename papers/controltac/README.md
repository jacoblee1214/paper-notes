# ControlTac (arXiv:2505.20498, CoRL 2026 포스터) 분석 노트

*ControlTac: Scaling Tactile Data with Physically Controlled Tactile Image Generation* (arXiv v3, 2026-09-08)의 표와 그림 수치를 다시 계산하고 공식 코드를 정적으로 읽은 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2505.20498에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **논문 결과의 재현은 어렵습니다.** 학습은 GPU 25.4시간과 약 82 GB 데이터, 실로봇 실험은 로봇이 필요합니다.
- **공식 체크포인트 추론은 설치와 다운로드를 승인받으면 CPU에서 가능해 보입니다.** 속도는 측정 전입니다.
- **수치 재계산은 끝냈고 `analysis/recompute.py`로 수 초에 다시 낼 수 있습니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv non-exclusive-distrib 라이선스입니다(CC 아님). 논문 그림은 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 코드는 `github.com/DongyuLUO/ControlTac`에 있고 LICENSE 파일이 없습니다. 이 저장소에는 코드를 넣지 않았고 실행하거나 설치하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다.

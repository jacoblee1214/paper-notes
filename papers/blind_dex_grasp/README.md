# Blind Dexterous Grasping via Real2Sim2Real Tactile Policy Learning (arXiv:2606.11767, CoRL 2026 포스터) 분석 노트

*Blind Dexterous Grasping via Real2Sim2Real Tactile Policy Learning* (arXiv v2, 2026-06-11)을 표 재계산, 통계 검정, 합성 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.11767에서 받을 수 있습니다). 논문 라이선스가 arXiv 비독점 배포 라이선스(CC 아님)라서 논문 그림도 넣지 않았고, 그림은 모두 이 저장소 작성자가 그렸습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥에서 불가입니다.** 공식 코드, 보정 값, 센서 설계, 실로봇이 공개되어 있지 않습니다.
- **축소판 구현은 부분적으로 가능합니다.** 접촉 모델과 보정 토이(`analysis/toy_event_calibration.py`, 약 40초, numpy)와 표 재계산(`analysis/recompute.py`)은 이 자료에서 했습니다. 인코더 사전학습과 전문가-증류 축소판은 합성 환경이나 손 모델 확보 뒤에 시작할 수 있습니다(4–6주, 추정, 측정 전).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 관련 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사(K), 토이(T) |
| `analysis/04_claim_verification.md` | 주장 판정(C), 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_event_calibration.py` | 토이: 이진 접촉 이벤트 보정 (numpy, 약 40초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_event_calibration.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv:2606.11767. abs 페이지의 license 링크가 arXiv 비독점 배포 라이선스를 가리킵니다(CC 아님).
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 코드는 공개되어 있지 않아 대조하지 않았습니다. 가중치와 데이터셋은 받지 않았습니다.

# VITaL Pretraining (arXiv:2403.11898, ICRA 2025) 분석 노트

VITaL Pretraining: Visuo-Tactile Pretraining for Tactile and Non-Tactile Manipulation Policies (arXiv v2, 2024-09-26)의 표 재계산, 통계 검정, 공식 코드 정적 대조 자료입니다. 재현은 하지 않았습니다.

- `report/report_public.pdf`: 통합 리포트 (논문 그림 없음)
- `analysis/01`–`04`: 논문 분석, 선행연구, 재계산, 코드 대조
- `analysis/recompute.py`: 그림 7 판독값의 Fisher·Holm 검정과 Wilson 구간 (scipy, 수 초)
- `analysis/plot_figures.py`: 그래프 생성
- `report/figures/own_success_ci.png`: 우리가 그린 그래프

## 재실행

    python3 analysis/recompute.py
    python3 analysis/plot_figures.py

## 출처와 라이선스

- 논문은 arXiv에서 CC BY-NC-SA 4.0으로 공개되어 있습니다. 이 저장소에는 논문 그림을 넣지 않았습니다. 원문 PDF도 넣지 않았습니다(arXiv:2403.11898 v2에서 받을 수 있습니다).
- 공식 코드(github.com/Abraham190137/TactileACT)는 MIT입니다. 코드는 넣지 않았고 실행하거나 설치하지 않았습니다.
- 그래프와 표 재계산은 이 저장소 작성자가 만들었습니다. 논문 수치는 인용과 재계산을 위해 옮겼습니다.

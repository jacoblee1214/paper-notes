# VPP (arXiv:2412.14803, ICML 2025 Spotlight) 분석 노트

*Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations* (arXiv v2, 2025-05-04)을 표 재계산, 통계 검정, 작은 토이로 읽은 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv non-exclusive 라이선스). 원문은 arXiv:2412.14803에서 받을 수 있습니다.

## 구현 가능성 (요약)

- **학습과 CALVIN 평가는 이 맥에서 불가능합니다.** GPU 여러 장과 Linux 환경, 약 500 GB 데이터가 필요합니다.
- **특징 추출 한 번의 시간 측정은 조건부로 가능합니다.** 설치와 가중치 다운로드, 코드 수정이 필요합니다. 아직 측정하지 않았습니다.
- **재계산과 토이는 완료했습니다.** `analysis/recompute.py`(수 초)와 `analysis/toy_future_error.py`(수 초, numpy)로 확인할 수 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy) |
| `analysis/toy_future_error.py` | 토이: 예측 오차와 역동역학 이득 (numpy) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_future_error.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv:2412.14803. arXiv abs 페이지의 "view license"는 arXiv non-exclusive distribution license 1.0을 가리킵니다. CC 라이선스가 아니므로 논문 그림은 싣지 않았습니다.
- 공식 코드: github.com/roboterax/video-prediction-policy, MIT. 저장소 코드는 싣지 않았고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

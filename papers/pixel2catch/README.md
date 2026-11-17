# Pixel2Catch (arXiv:2602.22733, RA-L 2026) 분석 노트

*Pixel2Catch: Multi-Agent Sim-to-Real Transfer for Agile Manipulation with a Single RGB Camera* (arXiv v1, 2026-02-26)를 수치 재계산, 통계 검정, numpy 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2602.22733에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다(CC BY-NC-ND 4.0).

## 구현 가능성 (요약)

- **이 맥에서 논문 실험은 재현할 수 없습니다.** 공식 학습은 NVIDIA GPU와 Isaac Sim을 전제하고, 체크포인트, 평가 스크립트, 실로봇 파이프라인이 공개되어 있지 않습니다.
- **축소 재구현은 조건부로 가능합니다.** 팔 한 개와 구 물체로 줄인 MAPPO는 1–2주 정도로 추정합니다(측정 전).
- **정보량 토이는 구현했습니다.** `analysis/toy_pixel_features.py`(numpy, 수 초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 결과, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_pixel_features.py` | 토이: 픽셀 특징의 접근 시각과 깊이 정보량 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_pixel_features.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY-NC-ND 4.0으로 공개되어 있습니다(arXiv 초록 페이지 확인). 이 자료는 논문 그림을 싣지 않았고, 논문 문장을 길게 옮기지 않았습니다.
- 공식 코드는 GitHub `seongdrgn/pixel2catch-github`에 있습니다. 라이선스 파일이 없습니다. 이 저장소에는 코드를 넣지 않았고, 실행, 설치, import를 하지 않았습니다. 파일 경로와 설정 이름만 짧게 인용했습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 자료의 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

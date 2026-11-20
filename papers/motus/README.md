# Motus (arXiv:2512.13030, CVPR 2026) 분석 노트

*Motus: A Unified Latent Action World Model* (arXiv v2, 2025-12-25)을 표 재계산, 과제 단위 통계 검정, 공식 코드 정적 대조, 2차원 CPU 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2512.13030에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 그래프는 모두 표 수치로 다시 그렸습니다.

## 구현 가능성 (요약)

- **공식 결과 재현은 이 환경에서 불가입니다.** 8B 모델 학습 (GPU 약 1.8만 시간), RoboTwin 평가 (GPU, 시뮬레이터), 실로봇 재현이 필요합니다.
- **축소 재구현은 가능합니다.** 설계 요소 (Tri-model 주의, 독립 타임스텝, 모드 샘플링)를 소형 합성 데이터로 옮기는 것은 CPU나 MPS에서 가능합니다 (추정 1–2주).
- **공식 코드에 없는 부분:** 광학 흐름 기반 잠재 행동 VAE, 다섯 모드 추론 API.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계, v1과 v2 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, 공식 저장소와 체크포인트 표, 학회 확인 |
| `analysis/03_analysis.md` | 표 재계산, 쌍체 검정, 실세계 부분 점수 역산, 토이 결과 |
| `analysis/04_code_verification.md` | 공식 코드 (커밋 88f6fe3) 정적 대조 D1–D10, 구현 가능성 판정 |
| `analysis/table14_robotwin.csv` | 논문 Table 14 과제별 값 (pdftotext에서 옮긴 뒤 검수) |
| `analysis/recompute.py` | 표 재계산과 검정. 입력은 위 CSV와 스크립트 안의 실세계 표 값 |
| `analysis/toy_unidiffuser.py` | 토이: 독립 타임스텝 대 공유 타임스텝 (2차원 가우시안, torch CPU) |
| `analysis/plot_figures.py` | 그래프 세 장 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_unidiffuser.py          # CPU, 약 13초
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv abs 페이지에서 확인한 라이선스가 arXiv non-exclusive distribution license 1.0입니다 (CC가 아닙니다). 따라서 논문 그림은 넣지 않았습니다.
- 공식 코드 `github.com/thu-ml/Motus`는 Apache-2.0입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 체크포인트 (Hugging Face `motus-robotics/*`)는 받지 않았고 라이선스를 확인하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 학회: CVPR 2026 포스터. 근거는 CVPR 공식 virtual 포스터 페이지와 CVF Open Access 논문 페이지입니다.

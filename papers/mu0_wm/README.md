# μ0 (arXiv:2606.13769, CoRL 2026 포스터) 분석 노트

*μ0: A Scalable 3D Interaction-Trace World Model* (arXiv v3, 2026-10-20)을 표 재계산, 통계 검정, 공개 코드 정적 대조, B-spline 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.13769에서 받을 수 있습니다). 논문 그림은 그림 6(막대그래프) 한 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **전체 재현은 개인 장비로는 어렵습니다.** 사전학습(2.59B), TraceExtract(CUDA 확장, 약 1.2K GPU 시간), 정책 학습(4 × L40S 가정), 실로봇이 필요합니다.
- **공개 체크포인트로 궤적 예측 추론은 조건부입니다.** 가중치 5.62 GB와 평가 세트 4.51 GB, 새 환경, 코드의 기본 장치(`cuda`) 수정이 필요합니다.
- **표 재계산, 통계, 코드 상수 산술, B-spline 토이는 CPU에서 수 초에 돕니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림은 그림 6 한 장) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V20, 일관성 검사, 토이, 주장 판정 C1–C14 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D19, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/code_arithmetic.py` | 코드 상수로 파라미터 수와 프레임률 산술 (표준 라이브러리) |
| `analysis/toy_bspline.py` | 토이: B-spline 표현 오차 (numpy, 약 1초) |
| `analysis/plot_figures.py` | 그래프 다섯 장 생성 |
| `report/figures/` | 그래프 다섯 장(우리가 그림)과 논문 그림 6(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/code_arithmetic.py
python3 analysis/toy_bspline.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에 저작권 고지는 없습니다.
- 논문 그림 6은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2606.13769"). 영상 프레임과 사진이 들어 있는 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/Yoonkyo/mu0`(커밋 963d5f13a94df31921d22887ff8611763b48178d)와 `github.com/Yoonkyo/TraceExtract`(커밋 c02eb8efefd988dda7f0046e38efb00762a84e08)에 있고 둘 다 Apache-2.0입니다. 이 저장소에는 코드를 넣지 않았고, 정적으로 읽기만 했습니다(실행, 설치 안 함).
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

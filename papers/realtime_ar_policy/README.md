# Real-Time Execution with Autoregressive Policies (arXiv:2606.13355, CoRL 2026 포스터) 분석 노트

*Real-Time Execution with Autoregressive Policies* (arXiv v2, 2026-11-05)를 표·그림 수치 재계산과 주장 검증으로 분석한 자료입니다. 방법 코드가 공개되어 있지 않아(프로젝트 페이지 "Coming soon") 코드 대조는 하지 않았습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.13355에서 받을 수 있습니다). 논문 그림은 차트·도식 다섯 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 CPU 노트북에서 불가능합니다.** LIBERO는 π0-FAST 체크포인트, JAX CUDA, 6k 스텝 학습(A100 4장)과 공개되지 않은 RTC 기준선 구현이 필요하고, DROID는 Franka 로봇과 H200 8장 학습이 필요합니다.
- **DP 제약 디코딩은 합성 어휘에서 독립 구현해 검증했습니다.** `analysis/toy_constrained_dp.py`(약 6초, numpy). 부록 A.1의 표는 완전탐색과 300/300 일치합니다. 합성 어휘라 논문의 성공률에 대한 증거는 아닙니다.
- 실제 FAST+ 어휘 확인과 소형 AR 정책 + 행동 큐 구현은 가능하지만 이 저장소에는 포함하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 차트·도식 다섯 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, v1 대 v2 차이 |
| `analysis/02_prior_work.md` | 선행 계보, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 재계산 V1–V14, 지연 산술, DROID 과제 단위 검정, 토이 K1–K3 |
| `analysis/04_claim_verification.md` | 주장 C1–C26, 재구현에 필요한 정보, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표·그림 수치 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/extract_figures.py` | 원문 PDF의 벡터 그림에서 막대·점 값을 읽음 (pymupdf, 원문 PDF 필요) |
| `analysis/figure_values.json` | 위 스크립트의 결과(그림에서 읽은 값) |
| `analysis/toy_constrained_dp.py` | 토이: DP 제약 디코딩 (numpy, 약 6초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |
| `report/crop_figures.py` | 논문 그림 다섯 장 자르기 (pymupdf, Pillow, 원문 PDF 필요) |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 다섯 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib pymupdf pillow
curl -L https://arxiv.org/pdf/2606.13355v2 -o paper/paper.pdf     # extract_figures.py, crop_figures.py에만 필요
python3 analysis/extract_figures.py
python3 analysis/recompute.py
python3 analysis/toy_constrained_dp.py
python3 analysis/plot_figures.py
python3 report/crop_figures.py
```

`analysis/recompute.py`는 `analysis/figure_values.json`이 있으면 원문 PDF 없이 돌아갑니다.

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지의 라이선스 링크 확인). 원문 PDF에서 저작권 고지는 찾지 못했습니다.
- 논문 그림은 차트와 도식 다섯 장(그림 1의 가운데 블록, 그림 3, 4, 5, 7a)만 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2606.13355"). 관측 사진, 실로봇 사진, 장면 사진이 있는 그림은 넣지 않았습니다.
- 프로젝트 페이지는 `oddqueue.github.io/realfast`에 있고, 저자 계정의 `oddqueue/realfast`는 openpi 포크입니다(방법 코드 없음). 이 저장소에는 외부 코드를 넣지 않았고 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

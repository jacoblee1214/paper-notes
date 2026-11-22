# LeaP (arXiv:2606.17408, CoRL 2026 포스터) 분석 노트

*Where Should Action Generation Begin? A Learnable Source Prior for Generative Robot Policies* (arXiv v1, 2026-06-16)를 표 재계산, 통계 검정, 공식 코드 정적 대조, 작은 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.17408에서 받을 수 있습니다). 논문 그림은 세 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 표(RoboTwin 15과제, 실로봇)의 재현은 이 맥에서 불가입니다.** 시뮬레이터가 Linux, NVIDIA GPU, Vulkan, CUDA 툴킷을 요구하고 기준선과 절제 코드는 공개되지 않았습니다.
- **재계산, 정적 코드 대조, 소스 변형 토이는 가능하고 끝냈습니다.** 토이는 `analysis/toy_source_prior.py`(약 1.8분, torch CPU)입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 세 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 기록 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1-V12, 토이, 주장 판정 C1-C16 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1-D16, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, statsmodels, 수 초) |
| `analysis/param_count.py` | 코드 레이어 모양에서 파라미터 수 계산 |
| `analysis/toy_source_prior.py` | 토이: 소스 변형 비교 (torch, 약 1.8분) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 세 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy statsmodels matplotlib torch
python3 analysis/recompute.py
python3 analysis/param_count.py
python3 analysis/toy_source_prior.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 세 장(그림 3, 5, 6)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2606.17408"). 입력 영상이 있는 그림, 실로봇 사진, 시뮬레이터 장면은 넣지 않았습니다.
- 공식 코드는 `github.com/daimeipo/LeaP`(MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았고 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

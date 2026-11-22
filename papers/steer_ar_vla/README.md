# Token Steering (arXiv:2606.15021, CoRL 2026 포스터) 분석 노트

*Steering Autoregressive Vision-Language-Action Policies via Action Token Intervention* (arXiv v1, 2026-06-12)을 표 재계산, 정합성 검사, 작은 CPU 토이, 저자 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.15021에서 받을 수 있습니다). 논문 그림은 저자가 그린 그래프 패널 네 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 실험(DROID 실로봇, 사용자 7명)의 재현은 이 맥(Apple M4 Pro, MPS, CUDA 없음)에서 불가능합니다.** 실로봇, GPU, 사용자 모집이 필요합니다.
- **방법의 구조 재구현은 쉽습니다.** 디코딩 루프에서 앞쪽 토큰 몇 개를 강제하는 수십 줄이며 소형 자기회귀 모델 토이로 검증할 수 있습니다.
- **재계산과 토이는 완료했습니다.** `analysis/recompute.py`(수 초)와 `analysis/toy_steering_prefix.py`(1초 미만).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그래프 패널 네 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 빠진 정보 |
| `analysis/02_prior_work.md` | 선행 계보, 코드와 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V8, 일관성 K1–K13, 토이, 주장 판정 C1–C9 |
| `analysis/04_code_verification.md` | 저자 코드 대조 D1–D15, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_steering_prefix.py` | 토이: 조향 청크의 DCT 배치와 Panda 기구학 prefix 정보 (numpy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그래프 패널 네 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_steering_prefix.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림은 저자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2606.15021"). 실로봇 사진과 카메라 화면이 깔린 패널, 관측 썸네일이 있는 그림은 넣지 않았습니다.
- 코드는 저자 계정의 openpi 포크(`github.com/jasontchan/openpi-fast-steering`, Apache-2.0)입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다. 논문과 프로젝트 페이지(`jasontchan.github.io/token-steering`)는 이 저장소를 링크하지 않아 저자 코드로 추정했습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

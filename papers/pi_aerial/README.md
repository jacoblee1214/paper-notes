# π, But Make It Fly (AirVLA, arXiv:2603.25038, CoRL 2026 포스터) 분석 노트

*π, But Make It Fly: Physics-Guided Transfer of VLA Models to Aerial Manipulation* (arXiv v1, 2026-03-26)을 표 재계산, 통계 검정, 토이 두 개로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2603.25038에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 그래프는 모두 이 노트 작성자가 그렸습니다.

## 구현 가능성 (요약)

- **논문 결과의 재현은 불가입니다.** 실기체(Starling 2 Max), 모션 캡처, 비공개 데이터(교시 270개), π0 미세조정(CUDA)이 필요하고 공식 코드는 아직 공개되지 않았습니다.
- **가이던스 샘플러의 소형 구현은 가능합니다.** 소형 흐름 정책에서 PyTorch로 1–2일(추정). 이 노트에서는 해석적 1차원 정책 토이(`analysis/toy_payload_guidance.py`, 1초 미만)로 세기 민감도를 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프 세 장) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법(식 1–17), 실험 표, 비판적 읽기 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, GitHub와 코드 표 |
| `analysis/03_analysis.md` | 표 재계산(V1–V11)과 토이 A, B |
| `analysis/04_claim_verification.md` | 공식 코드가 없어서 주장 검증(C1–C18), 구현 가능성 판정 |
| `analysis/recompute.py` | 표 정수 복원, Wilson 구간, Fisher 검정, 표본 크기 (numpy, scipy, 수 초) |
| `analysis/toy_payload_guidance.py` | 토이: 가중 α 표와 1차원 가우시안 정책의 가이던스 (numpy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_payload_guidance.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지의 license 링크 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림은 실기체와 실험실 사진, 그 스플랫 재구성이라 넣지 않았습니다.
- 프로젝트 페이지는 `airvla.github.io`이고 공식 코드는 아직 없습니다. 코드와 데이터는 "Coming Soon"입니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 노트 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았고 어떤 코드도 실행하지 않았습니다.

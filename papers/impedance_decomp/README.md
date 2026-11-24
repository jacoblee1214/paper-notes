# Constraint-Decomposed Impedance Control (arXiv:2610.16316, CoRL 2026 Spotlight) 분석 노트

*Constraint-Decomposed Impedance Control from Generative Policy for Articulated Object Manipulation* (arXiv v1, 2026-10-12, 단일 버전)을 표 재계산, 공식 코드의 정적 대조, 공개 예시 앙상블 위의 직접 구현으로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2610.16316에서 받을 수 있습니다). 논문 그림은 세 장(그림 1 위쪽 세 패널, 그림 4, 그림 5)만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **방법(앙상블 → PCA → 게인)은 CPU에서 구현·실행했습니다.** `analysis/toy_ensemble_estimator.py`(numpy, 약 3초).
- **Isaac Lab 시뮬레이션 평가와 정책 학습은 일반 노트북에서 어렵습니다.** Isaac Sim은 Linux/Windows와 NVIDIA GPU가 필요하고, 실로봇(FR3 + 힘토크 센서)도 필요합니다.
- **MuJoCo와 소형 정책으로 만든 축소판은 가능해 보입니다**(2–3주, 추정, 측정 전).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 세 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산, 일관성 검사, 이론 검산, toy 결과, 주장 판정 C1–C15 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D17, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 수치 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/check_theory.py` | 부록 B, C의 대수 주장 수치 검산 (numpy, 1초 미만) |
| `analysis/toy_ensemble_estimator.py` | Algorithm 1 직접 구현을 공식 예시 앙상블 4개에 적용 (numpy, 약 3초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 세 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/check_theory.py
# 공식 저장소를 따로 받아 examples/data 폴더 경로를 넘긴다 (코드를 실행하지는 않고 npz 배열만 읽음)
python3 analysis/toy_ensemble_estimator.py <공식 저장소>/examples/data
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 논문 그림 세 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2610.16316"). 실로봇 사진, 시뮬레이션 렌더 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/erickun0125/constraint-decomposed-impedance-control`(MIT, 시뮬레이션 전용)에 있습니다. 이 저장소에는 코드를 넣지 않았고 실행하거나 설치하지 않았습니다. 예시 앙상블(npz)은 공식 저장소에서 받아 배열만 읽습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 직접 구현, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

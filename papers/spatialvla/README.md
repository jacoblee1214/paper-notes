# SpatialVLA (arXiv:2501.15830, RSS 2025) 분석 노트

*SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model* (arXiv v5, 2025-05-19)을 표 재계산, 공식 코드 정적 대조, CPU 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다. arXiv:2501.15830에서 받을 수 있습니다.

## 요약

- **시뮬 평균은 재계산과 맞습니다.** Google Robot 시각 매칭 71.9, WidowX 34.4 / 42.7, LIBERO 78.05.
- **실로봇 제로샷 우위는 방향이 확실하고 크기는 불확실합니다.** 77회 합산 72.7% 대 51.9% (Fisher p = 0.012), 과제당 11회입니다.
- **Franka 실험의 우위는 유의하지 않습니다.** 세 설정 모두 p ≥ 0.08입니다.
- **적응형 행동 격자는 재구성 오차로 보면 균일 격자보다 나쁩니다.** 같은 빈 수에서 MSE 2.9배 (CPU 토이).
- **공식 코드의 기본값이 논문과 다른 것이 여럿 있습니다.** 패치 점 수, 청크 길이, LoRA epoch, 카메라 파라미터 가정.

## 구현 가능성

- 사전학습과 미세조정은 이 맥(Apple M4 Pro, MPS, CUDA 없음)에서 불가능합니다.
- 실로봇과 시뮬 평가 재현은 불가능합니다.
- 추론 재현은 체크포인트(약 8.1 GB) 받기와 설치 뒤에 측정해야 하며 조건부입니다.
- 격자와 토크나이저 검사, 재계산은 CPU로 완료했습니다.

## 파일

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소와 체크포인트 표 |
| `analysis/03_analysis.md` | 표 재계산, 일관성 검사, 토이 결과, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_adaptive_grid.py` | 토이: 적응 격자와 균일 격자의 재구성 오차 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_adaptive_grid.py   # 공식 저장소 복제본이 spatialvla/repo 또는 SPATIALVLA_REPO에 있어야 함
python3 analysis/plot_figures.py
```

공식 저장소는 `git clone --depth 1 https://github.com/SpatialVLA/SpatialVLA repo` 로 받을 수 있습니다. 이 패키지는 그 코드를 포함하지 않습니다.

## 출처와 라이선스

- 논문은 arXiv 비독점 라이선스(nonexclusive-distrib 1.0)로 공개되어 있습니다. CC BY가 아니므로 논문 그림은 넣지 않았습니다.
- 공식 코드는 라이선스 파일이 없습니다. 이 패키지에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 체크포인트는 Hugging Face에서 MIT로 표시되어 있습니다. 받지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

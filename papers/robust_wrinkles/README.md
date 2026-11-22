# Robustness without Wrinkles / CORD-SLS (arXiv:2606.14188, CoRL 2026 스포트라이트) 분석 노트

*Robustness without Wrinkles: Parallel Simulation and Robust MPC for Certified Deformable Manipulation* (arXiv v1, 2026-06-12)을 표 재계산, 공식 코드 정적 대조, 작은 numpy 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.14188에서 받을 수 있습니다). 논문 그림은 세 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 규모의 재현과 하드웨어 재현은 이 맥(Apple M4 Pro, CUDA 없음)에서 불가입니다.** KUKA 로봇, 카메라, GPU 학습이 필요하고 지각과 제어 코드가 저장소에 없습니다.
- **공식 코드의 소규모 실행은 조건부입니다.** JAX CPU 설치와 실행이 필요하고, 저장소의 벤치마크 설정은 SLS 단계를 끄고 있습니다(정적 읽기).
- **수치 재계산과 토이는 가능하며 끝냈습니다.** `analysis/recompute.py`, `analysis/toy_cp_coverage.py`, `analysis/toy_scan_riccati.py`(각 1초 미만).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 세 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 기록 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 결과, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 1·2·3 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_cp_coverage.py` | 토이: 정합 예측 교정의 궤적 수준 보장 (numpy, 약 1 s) |
| `analysis/toy_scan_riccati.py` | 토이: 병렬 스캔 결합 규칙 검사 (numpy, 1 s 미만) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 세 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_cp_coverage.py
python3 analysis/toy_scan_riccati.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 세 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2606.14188"). 실로봇 사진이 있는 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/trustworthyrobotics/CORD-SLS`(커밋 77e82b4)에 있습니다. 최상위 LICENSE가 없고 `planners/gpu_sls`만 MIT입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

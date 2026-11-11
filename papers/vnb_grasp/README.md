# VNB: 불확실성 하의 파지 (arXiv:2604.25897v2, IROS 2026 채택) 분석 노트

*Variational Neural Belief Parameterizations for Robust Dexterous Grasping under Multimodal Uncertainty* (Enwerem, Kalyanaraman, Baras, Belta)를
표 재계산, 유의성 검정, 유도, numpy 토이, 공식 저장소 정적 대조로 분석한 자료입니다. 논문 원문과 그림은 넣지 않았습니다(아래 출처 참조).

## 구현할 수 있는지

- **공식 시뮬레이션 재현: 조건부 가능, 중간 노력.** MuJoCo, PyTorch, NumPy가 필요합니다. 샘플 1 에피소드는 약 6분, 480 에피소드는 4–8시간으로 추정됩니다(저장소 README). CUDA 필요 여부는 확인하지 못했습니다.
- **지금 상태로 논문 표를 재현하는 것은 어렵습니다.** 마찰 혼합 규칙, 위험 가중치, 접촉 강성, 성공 기준이 논문과 코드에서 다릅니다(`analysis/04_code_verification.md` §2).
- **실물 재현: 불가.** 로봇 팔, 손, 카메라 두 대, ICP 파이프라인이 필요하고 코드가 공개되지 않았습니다.

## 파일

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, v1 → v2 차이 |
| `analysis/02_prior_work.md` | 선행연구 계보, 관련 저장소 표, GitHub 표 |
| `analysis/03_analysis.md` | Table III 재계산, 검정, 보정 방향, 유도, toy 결과 |
| `analysis/04_code_verification.md` | 논문과 공식 저장소 대조(D1–D25), 구현 가능성, 권리 |
| `analysis/table_check.py` | Table III 재계산과 검정(1초 미만) |
| `analysis/cvar_toy.py` | 이중 봉우리 믿음의 CVaR 선택, µ_f 민감도, softplus 편향, 샘플링 확인 |
| `analysis/extra_eq8_tau.py` | 추가 검증: 식 (8) 샘플의 간격 질량과 온도 τ 어닐링 (numpy, 1스레드, 약 1초) |
| `analysis/plot_figures.py` | 그래프 세 장 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/cvar_toy.py
python3 analysis/plot_figures.py
```

모두 CPU 단일 스레드에서 수 초 안에 끝납니다. toy의 파지 비용, 실패 벌점, 믿음 모양은 단순화한 가정입니다.

## 출처와 라이선스

- arXiv abs 페이지의 라이선스는 CC BY 4.0이지만, 논문 PDF 1쪽에 IEEE 권리 고지가 있습니다. 두 고지가 충돌하므로 **논문 그림과 원문 PDF는 이 저장소에 넣지 않았습니다.**
- 공식 저장소는 `coenwerem/vnb-grasp`(Apache-2.0, 커밋 `23702bd`)입니다. 클론하지 않았고, 대조 결과만 `04_code_verification.md`에 있습니다.
- 그래프는 논문 Table III의 수치와 직접 만든 토이로 그렸습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

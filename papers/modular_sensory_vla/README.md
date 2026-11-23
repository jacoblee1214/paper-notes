# MoSS (arXiv:2604.23272, CoRL 2026 포스터) 분석 노트

*Modular Sensory Stream for Integrating Physical Feedback in Vision-Language-Action Models* (arXiv v2, 2026-09-11)를 표 재계산, 통계 검정, 주장별 검증으로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2604.23272에서 받을 수 있습니다). 논문은 arXiv 비독점 배포 라이선스라서 논문 그림도 넣지 않았고, 그래프는 모두 이 저장소 작성자가 그렸습니다.

## 구현 가능성 (요약)

- **논문 결과의 재현은 이 맥(Apple M4 Pro, MPS, CUDA 없음)에서 불가입니다.** 실로봇(Franka, AnySkin, OpenArm 손)과 CUDA 기반 VLA 학습이 필요합니다. 공식 코드는 공개되지 않았습니다(프로젝트 페이지 "Coming Soon").
- **구조의 축소판 재구현은 가능합니다.** 합성 접촉 과제에서 스트림 설계와 학습 순서를 비교하는 작은 torch 모델입니다(노력 추정 1–2일, 측정 전).
- **수치 재계산은 CPU에서 수 초에 끝납니다.** `analysis/recompute.py`.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공개 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V15, 일관성 K1–K20, 구조 시뮬레이션 T1 |
| `analysis/04_claim_verification.md` | 주장 판정 C1–C18, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 표 재계산과 검정 (numpy, scipy, statsmodels, 수 초) |
| `analysis/toy_joint_attention.py`, `toy_joint_attention_results.json` | 공동 어텐션에서 동결 스트림의 영향 (numpy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy statsmodels matplotlib
python3 analysis/recompute.py
python3 analysis/toy_joint_attention.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문의 arXiv 라이선스는 비독점 배포(`nonexclusive-distrib/1.0`)입니다. CC BY가 아니므로 논문 그림은 싣지 않았습니다.
- 프로젝트 페이지는 `jiminlx.github.io/MoSS`입니다. 코드는 공개되어 있지 않습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

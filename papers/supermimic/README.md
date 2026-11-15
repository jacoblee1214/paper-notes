# SuperMimic (arXiv:2610.28480, CoRL 2026 구두) 분석 노트

*SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model* (arXiv v2, 2026-11-09)을 수치 재계산, 공식 코드 정적 읽기, numpy 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **이 맥(CUDA 없음)에서 논문 결과 재현은 불가합니다.** 공식 저장소는 NVIDIA GPU와 MuJoCo Warp를 전제하고, 2-GPU 코드와 평가 스크립트가 없습니다.
- **알고리즘 재구현(PyTorch)은 가능합니다**(추정 1–2주). 소형 G1 MuJoCo CPU 환경은 조건부입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 결과, 한계, 본문 불일치, 질문 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 표 |
| `analysis/03_analysis.md` | 수치 재계산, 실패 차이의 검정, 토이 결과 |
| `analysis/04_code_verification.md` | 논문 대 공식 코드 대조(D1–D28), 구현 가능성 |
| `analysis/recompute.py` | 본문 수치 재계산과 일관성 검사 (scipy, 수 초) |
| `analysis/toy_bptt_vs_pg.py` | 토이: 학습된 모델 기울기 대 정책 기울기 (numpy, 약 2분) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_bptt_vs_pg.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포(nonexclusive-distrib)입니다. CC 라이선스가 아니므로 논문 그림과 원문 PDF는 넣지 않았습니다. 원문은 arXiv:2610.28480에서 받을 수 있습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 코드(HybridRobotics/supermimic, Apache-2.0)는 정적으로만 읽었고 재배포하지 않았습니다. 데이터(BONES-SEED)와 체크포인트는 받지 않았습니다.

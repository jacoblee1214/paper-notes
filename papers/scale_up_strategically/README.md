# Scale Up Strategically (arXiv:2607.21582, CoRL 2026 포스터) 분석 노트

*Scale Up Strategically: Learning Compositional Generalization via Bias-Aware Evaluation and Data Collection for Robotic Manipulation* (arXiv v1, 2026-07-23)를 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 공식 코드는 넣지 않았습니다(arXiv:2607.21582, 공식 코드 저장소는 `analysis/02_prior_work.md`).

## 구현 가능성 (요약)

- **논문의 학습과 실로봇 실험은 이 노트북(Apple M4 Pro, MPS, CUDA 없음)에서 재현할 수 없습니다.** GPU 여러 장과 로봇이 필요하고 공식 코드는 Linux 모션 플래너를 전제합니다.
- **표 재계산과 통계는 구현했습니다.** `analysis/recompute.py`(수 초, 표준 라이브러리).
- **훈련 칸 비율을 분리해 보는 합성 과제를 구현했습니다.** `analysis/toy_seen_fraction.py`(torch CPU, 수십 초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 평가(FDR/FDH), 수집 전략, 실험 |
| `analysis/02_prior_work.md` | 선행 계보, 코드와 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V9, 설계 검사 K1–K2, 주장 판정 C1–C13 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조 D1–D13, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 |
| `analysis/toy_seen_fraction.py` | 합성 과제: 전체 격자 대 미훈련 칸 성공률 |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install matplotlib numpy torch
python3 analysis/recompute.py
python3 analysis/toy_seen_fraction.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive-distrib 라이선스로 공개되어 있습니다(arXiv 라이선스 링크 확인). 논문 그림은 넣지 않았고, 그림은 모두 이 노트의 작성자가 표 수치와 공개 코드 값으로 그린 것입니다.
- 프로젝트 페이지: https://yqi19.github.io/scale-up-strategically/ . 공식 코드: https://github.com/yqi19/CoRL2026-Scale-Up-Strategically (분석 커밋 `b5b32418bd070b2ce55f3fcde21d742306331ce0`, 정적으로만 읽음).
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 분석 문서와 스크립트는 이 노트의 작성자가 만들었습니다.

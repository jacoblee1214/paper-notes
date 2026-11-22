# BeyondMimic (arXiv:2508.08241) 분석 자료

BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion (arXiv v4, 2025-11-13). 학회 표기 없음 (arXiv).

- 논문: https://arxiv.org/abs/2508.08241
- 프로젝트 페이지: https://beyondmimic.github.io/
- 공식 코드: https://github.com/HybridRobotics/whole_body_tracking (MIT, 아카이브됨)

## 그림 게시

- 그림 게시: 불가 (비독점 배포 라이선스 nonexclusive-distrib 1.0, CC BY 아님. 공개 산출물에는 우리가 그린 그래프만 씀)

## 파일

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 |
| `analysis/02_prior_work.md` | 선행연구, GitHub 표 |
| `analysis/03_analysis.md` | 사용자 연구 재계산, 일관성 검사 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조와 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 재계산 (CPU, 수 초) |
| `analysis/plot_figures.py` | 그래프 (우리가 그린 그림) |
| `report/report_public.pdf` | 공개 리포트 (논문 그림 없음) |

## 검증 방식

- 보고 수치 재계산. 재학습과 재현은 하지 않음.
- 공식 코드 정적 대조 (아카이브 커밋 9db915838b17). 실행하지 않음.

## 재실행

```
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

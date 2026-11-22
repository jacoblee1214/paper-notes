# SPIDER (arXiv:2511.09484) 분석 노트

*SPIDER: Scalable Physics-Informed Dexterous Retargeting* (arXiv v3, 2026-09-26)을 표 재계산, 통계 검정, 공식 저장소 정적 대조로 분석한 자료입니다. 학회 표기는 없습니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2511.09484에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **논문 수치 재현은 이 맥에서 불가능합니다.** 시뮬레이터, 정책 학습, 실로봇 실험이 GPU나 로봇을 전제합니다.
- **공식 CPU 워크플로로 장면 하나 재생은 조건부입니다.** MuJoCo 3.17 이상 설치와 장면 다운로드가 필요합니다.
- **알고리즘 재구현은 가능합니다.** 식 2, 3과 Algorithm 1은 명확합니다. β1, β2, 강건화 범위, 접촉 필터 임계값은 논문과 코드에 없습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소와 자료 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 통계 검정, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 표 재그림 세 장 (matplotlib) |
| `report/figures/` | 표 재그림 세 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv abs 페이지 "view license"는 CC BY 4.0입니다. 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 논문 그림은 넣지 않았습니다. 그림 중 일부에 사람 영상 프레임과 실로봇 사진이 있기 때문입니다.
- 공식 코드 `github.com/facebookresearch/spider`는 `LICENSE` 본문이 CC-BY-NC 4.0입니다. 이 자료에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 자료의 작성자가 만들었습니다.
- 원자료(궤적별 롤아웃)와 데이터셋은 받지 않았습니다.

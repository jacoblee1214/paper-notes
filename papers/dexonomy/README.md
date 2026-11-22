# Dexonomy (arXiv:2504.18829, RSS 2025) 분석 노트

*Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy* (arXiv v2, 2025-09-03)을 표 재계산, 일관성 검사, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2504.18829에서 받을 수 있습니다). 그림은 우리가 논문 표의 숫자로 다시 그린 그래프 세 장입니다.

## 구현 가능성 (요약)

- **데이터 생성과 학습 재현은 이 맥에서 불가입니다.** 논문은 GPU 8장에서 3일 걸리는 데이터 생성과 CUDA 학습을 씁니다.
- **MuJoCo 지역 정제와 여섯 방향 시험의 소규모 재현은 조건부입니다.** 설치 승인 후 CPU로 시작할 수 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그래프 세 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소의 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 통계 민감도, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 Wilson 구간 (표준 라이브러리만, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/figures/` | 그래프 세 장 (우리가 그림) |

## 재실행

```bash
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다 (arXiv 라이선스 페이지 확인). 원문 PDF에 저작권 표시는 찾지 못했습니다.
- 공식 저장소 README는 "이 작업과 데이터셋은 CC BY-NC 4.0"이라고 씁니다. LICENSE 파일은 없습니다. 두 표시가 충돌하므로 논문 그림은 넣지 않았습니다.
- 공식 코드와 데이터는 이 저장소에 넣지 않았고, 실행하거나 설치하지 않았습니다.

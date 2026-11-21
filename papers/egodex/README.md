# EgoDex (arXiv:2505.11709, ICLR 2026 포스터) 분석 노트

*EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video* (arXiv v3, 2026-03-09)의 보고 수치를 다시 계산하고, 공식 저장소의 지표 코드를 정적으로 대조한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다. arXiv 라이선스가 CC BY가 아니기 때문입니다. 데이터셋은 받지 않았습니다.

## 구현 가능성 (요약)

- **이 맥에서 논문 학습 재현은 불가능합니다.** 14개 모델은 모델당 A100 8장으로 약 72시간이 걸리고, 데이터는 1.7 TB입니다.
- **지표와 입력 파이프라인은 CPU에서 할 수 있습니다.** 행동 빌더(6D 회전, 손가락 끝, 카메라 프레임, 청크 상대화)는 공식 코드에 없어서 직접 만들어야 합니다.
- 공식 저장소(apple/ml-egodex)는 지표와 데이터 로더만 담고 있습니다. 학습 코드는 없습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 버전 차이(v1–v3), 권리 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 퍼센트 재계산, 표 사이 일관성, best-of-K 개선, 불확실성 민감도, 데이터 통계 |
| `analysis/04_code_verification.md` | 공식 코드 대조 (확인 항목 K, 차이 D), 저자에게 물을 질문, 구현 가능성 |
| `analysis/recompute.py` | 표 2–6 재계산 스크립트 (순수 Python) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv:2505.11709 (abs 페이지 라이선스는 비독점 배포 1.0, CC BY 아님).
- 공식 코드: https://github.com/apple/ml-egodex (Apple 자체 라이선스). 이 분석은 저장소를 얕게 복제해 정적으로만 읽었고, 실행하지 않았습니다.
- 데이터: 저장소 README는 CC-by-NC-ND라고 씁니다. 논문 v3 본문에는 이 문장이 없습니다.
- 그래프, 재계산 스크립트, 분석 문서는 이 분석이 만든 것입니다.

# Phantom (arXiv:2503.00779, CoRL 2025) 분석 노트

*Phantom: Training Robots Without Robots Using Only Human Videos* (arXiv v2, 2026-05-28)를 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2503.00779에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 우리가 그린 그래프 두 장만 있습니다.

## 구현 가능성 (요약)

- **논문 결과 재현은 이 맥에서 불가입니다.** 실로봇 평가, 사람 영상 수집, CUDA 전제 정책 학습이 필요합니다.
- **공식 코드에 없는 것:** Hand Mask 확산 모델과 증강 코드.
- **가능한 것:** 행동 변환 기하 검사(CPU), 재계산과 검정 (수 초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 비교 기준, 구성 요소 출처, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정력, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 구현 가능성, 라이선스 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다. PDF 본문에서 저작권 표시를 찾지 못했습니다. 논문 그림은 실로봇과 장면 사진을 포함하므로 넣지 않았습니다.
- 공식 코드는 `github.com/MarionLepert/phantom`(MIT, LICENSE 파일 확인)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 데이터와 체크포인트는 받지 않았습니다.

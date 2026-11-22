# DROID (arXiv:2403.12945, RSS 2024) 분석 노트

*DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset* (arXiv v2, 2025-04-22)을 정책 실험 수치 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2403.12945에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 우리가 그린 그래프 두 장만 넣었습니다.

## 구현 가능성 (요약)

- **데이터 수집과 실로봇 평가는 이 맥에서 불가능합니다.** Franka 팔, 카메라, 텔레오퍼레이션 장비가 필요합니다.
- **정책 학습 코드는 조건부로 가능합니다.** 공식 코드의 언어 모델이 `cuda`로 고정되어 있어 수정이 필요합니다. 논문 규모의 학습은 GPU가 필요합니다.
- **재계산과 검정은 완료했습니다.** `analysis/recompute.py`는 numpy와 scipy만 쓰고 수 초가 걸립니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 수집 체계, 정책 실험, 버전 차이 |
| `analysis/02_prior_work.md` | 데이터셋 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 정책 실험 재계산, 검정, 일관성 검사 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 그림 8, 10 재계산과 검정 (numpy, scipy) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 표시는 찾지 못했습니다.
- 공식 코드는 `github.com/droid-dataset/droid_policy_learning`(MIT)과 `github.com/droid-dataset/droid`(라이선스 파일 없음)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 그래프와 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다. 그래프 값은 논문 그림 8과 10의 막대 위 숫자를 옮긴 것입니다.
- 데이터와 가중치는 받지 않았습니다.

# World Model for Robot Learning: A Comprehensive Survey (arXiv:2605.00080) 분석 노트

이 서베이의 분류, 표 수치, 본문 주장을 점검한 분석입니다. 논문 원문과 논문 그림은 넣지 않았습니다(라이선스는 CC BY 4.0이지만 게시본에서는 우리가 만든 그래프만 씁니다). 학회 표기는 없습니다(arXiv 전용).

## 구현 가능성 (요약)

- **서베이의 주장 검증은 맥에서 가능하고 완료했습니다.** 재계산과 문헌 대조만 필요합니다.
- **서베이가 인용한 방법의 재현은 맥에서 불가능합니다.** 영상 세계 모델의 학습과 평가에 CUDA GPU가 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 주장과 근거, 라이선스 |
| `analysis/02_prior_work.md` | 계보, 서베이 인용과 자료 위치 |
| `analysis/03_analysis.md` | 인용 규모, 표 재계산, 일관성 검사, 초록 대조 |
| `analysis/04_claim_verification.md` | 서베이 주장 판정과 구현 가능성 |
| `analysis/survey_taxonomy.py` | 참고문헌 파싱과 인용 집계 (표준 라이브러리) |
| `analysis/recompute_survey.py` | 표 5 평균 검산, 500회 정수 검사, 검정 (표준 라이브러리) |
| `analysis/plot_figures.py` | 그래프 세 장 (matplotlib) |

## 재실행

```bash
python3 -I analysis/survey_taxonomy.py
python3 -I analysis/recompute_survey.py
pip install matplotlib
python3 -I analysis/plot_figures.py
```

## 출처와 가정

- 논문: arXiv:2605.00080 (v1). abs 페이지의 라이선스 CC BY 4.0. 원문 PDF는 이 저장소에 넣지 않았습니다.
- 표 5·6의 시행 횟수는 논문에 없습니다. LIBERO 500회 가정을 쓴 검정과 정수 검사는 가정에 의존합니다.
- 서베이 저장소(NTUMARS/Awesome-World-Model-for-Robotics-Policy)는 큐레이션 목록이며 라이선스 파일이 없습니다. 이 저장소에는 복사하지 않았습니다.

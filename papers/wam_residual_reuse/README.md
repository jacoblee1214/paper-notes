# Cache or Denoise Less? 분석 (arXiv:2611.19709)

논문 수치를 쌍체 불일치 개수로 다시 계산하고, 공식 코드를 정적으로 대조했습니다. 모델은 실행하지 않았습니다.

- `analysis/01_paper_analysis.md`: 논문 분석
- `analysis/02_prior_work.md`: 선행연구, 학회, 권리 판단
- `analysis/03_analysis.md`, `03_recompute.py`, `03_recompute_results.json`: 재계산
- `analysis/04_code_verification.md`: 공식 코드 정적 대조
- `analysis/plot_own.py`: 그래프 생성 (논문 표 수치에서 직접 그림)
- `report/report_public.pdf`: 공개 리포트

재계산은 공식 저장소의 `results/budget_shift/summary.json`을 읽습니다. 이 파일은 저장소 복제본에 있으며 이 패키지에는 들어 있지 않습니다. 논문 표 값은 스크립트 안에 직접 적혀 있습니다.

논문 그림은 넣지 않았습니다. 그래프는 우리가 그렸습니다.

라이선스: 논문은 CC BY 4.0입니다. 공식 코드는 라이선스 파일이 없어 코드를 포함하지 않습니다.

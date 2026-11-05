# CHORD (arXiv:2607.00033) 분석 노트

*Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration*
(Zhu, Liu, Jain 외, NVIDIA, CoRL 2026)을 공식 코드 대조(`nvidia-isaac/video_to_data` @ `9cad120`), 표·그림 재계산,
공식 보상 함수의 numpy toy, FlashDexRetarget(arXiv:2610.01849) 기준선 결과와의 대조로 분석한 자료입니다. 학습 재현은 하지 않았습니다(NVIDIA GPU 필요).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations, v1 ↔ v2 차이 |
| `analysis/02_prior_work.md` | 선행연구 계보, FlashDexRetarget과의 정합, 접촉 보상 3종 비교, GitHub 정리 |
| `analysis/03_analysis.md` | 지지 함수의 성질, Eq. 3 vs 코드 보상, force closure 표본 문제, 학습 예산 산술 |
| `analysis/04_code_verification.md` | 논문 ↔ 코드 대조(D1–D12), 주장별 재계산(C1–C14), 구현 가능성 |
| `analysis/table_check.py` | 표·그림·예산 재계산 |
| `analysis/wrench_toy.py` | 공식 보상 함수 numpy 이식과 toy(W1–W9) |
| `analysis/plot_figures.py` | 리포트 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/wrench_toy.py && python3 analysis/plot_figures.py
```

모두 CPU에서 1초 안팎에 끝납니다. 논문 PDF나 공식 코드는 필요하지 않습니다(수치와 설정값은 출처와 함께 스크립트에 적었습니다).

## 출처와 라이선스

- 논문은 arXiv abs 페이지 기준 **CC BY 4.0**이지만, PDF 첫 쪽에 "© 2026 NVIDIA. All rights reserved." 문구가 함께 있습니다. 권리 표기가 충돌하므로 **논문 그림은 이 저장소에 넣지 않았습니다.** 그림은 [프로젝트 페이지](https://nvidia-isaac.github.io/video_to_data/chord/)에서 볼 수 있습니다.
- `reward_curves.png`는 공식 코드의 보상 함수를 옮긴 toy로, `table1_bars.png`와 `budget.png`는 논문 표·공식 설정값·FlashDexRetarget Table I의 수치로 직접 그렸습니다.
- 공식 코드는 Apache-2.0(문서는 CC BY 4.0)입니다. `wrench_toy.py`는 그 함수들을 numpy로 다시 쓴 것입니다.
- 원문 PDF는 이 저장소에 넣지 않았습니다. 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

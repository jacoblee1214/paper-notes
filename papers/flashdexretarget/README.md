# FlashDexRetarget (arXiv:2610.01849) 분석 노트

*FlashDexRetarget: Accelerating Dexterous Manipulation Data Generation through Multi-Motion Retargeting*
(Lee, Kim, Hwang, Oh et al., KAIST AI · Holiday Robotics, 2026)을 논문 표 재계산, 보상 식 toy 검증,
[공식 코드](https://github.com/Holiday-Robot/FlashDexRetarget)(`6f95e13`) 대조로 분석한 자료입니다. 학습 재현(GPU)은 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보와 구성 요소별 출처 |
| `analysis/03_analysis.md` | 점군 정합 보상의 기하, 회전축 사각지대, 거리 보상의 여유, 계산량 해부 |
| `analysis/04_code_verification.md` | 논문 ↔ 코드 대조(D1–D13), 수치 재계산(C1–C13, V1–V2), 구현 가능성 |
| `analysis/table_check.py` | Table I과 본문·프로젝트 페이지 수치 재계산 |
| `analysis/reward_toy.py` | 공식 코드의 보상 식을 numpy로 옮긴 toy 검증 |
| `analysis/plot_headline.py` | 리포트 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/reward_toy.py
cd analysis && python3 plot_headline.py
```

모두 CPU에서 몇 초 안에 끝납니다. 논문 PDF나 공식 코드는 필요하지 않습니다(수치는 스크립트에 출처와 함께 적었습니다).

## 출처와 라이선스

- 논문은 **CC BY 4.0**입니다(arXiv abs 페이지 확인). `report/figures/paper_fig*.png`는 논문에서 잘라낸 그림이며 © 저자, CC BY 4.0, 출처 arXiv:2610.01849입니다.
- `headline_table1.png`는 논문 Table I의 수치로, `reward_axis.png`는 공식 코드(MIT)의 보상 식으로 직접 그렸습니다.
- 공식 코드와 원문 PDF는 이 저장소에 넣지 않았습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

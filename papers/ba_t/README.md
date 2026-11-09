# BA-T (arXiv:2606.03287) 분석 노트

*BA-T: An Iterative Transformer for Two-View Bundle Adjustment*
(Zhang, Chen, Cremers, Wang, TU Munich · MCML · Ulm University · ETH Zurich, NeurIPS 2026)을 논문 표 재계산,
파라미터·연산량 셈, numpy 두 시점 BA toy, [공식 코드](https://github.com/zhangganlin/BA-T)(`406d791`) 대조로 분석한 자료입니다.
모델 실행(가중치 다운로드)과 학습 재현은 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations, v1 ↔ v2 차이 |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소별 출처, 관련 GitHub 저장소 |
| `analysis/03_analysis.md` | Schur 보완의 의미, 파라미터 산술, 갱신 순서 toy, 효율 해부 |
| `analysis/04_code_verification.md` | 논문 ↔ 코드 대조(D1–D13), 수치 재계산(C1–C16, V1–V3), 구현 가능성 |
| `analysis/table_check.py` | 표 1–6과 식 11 가중치 재계산 |
| `analysis/param_check.py` | 공식 코드 모듈 정의로 파라미터 수·연산량 셈 (torch 불필요) |
| `analysis/ba_toy.py` | 두 시점 BA toy: Schur = 전체 해, 갱신 순서 비교, 대응점 수 의존성 |
| `analysis/plot_figures.py` | 리포트 그래프 세 장 |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/table_check.py && python3 analysis/param_check.py && python3 analysis/ba_toy.py
python3 analysis/plot_figures.py
```

모두 CPU에서 몇 초 안에 끝납니다. 논문 PDF, 공식 코드, 가중치는 필요하지 않습니다(표 수치는 스크립트에 출처와 함께 적었습니다).

## 출처와 라이선스

- 논문은 **CC BY 4.0**입니다(arXiv abs 페이지 확인). `report/figures/paper_fig*.png`는 논문에서 잘라낸 그림이며 © 저자, CC BY 4.0, 출처 arXiv:2606.03287입니다.
- `iter_curves.png`, `ablation_ladder.png`는 논문 표 2·3의 수치로, `toy_order.png`는 `ba_toy.py` 결과로 직접 그렸습니다.
- 공식 코드(Apache-2.0 + CC BY-NC-SA 4.0), 가중치(CC BY-NC-SA 4.0), 원문 PDF는 이 저장소에 넣지 않았습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

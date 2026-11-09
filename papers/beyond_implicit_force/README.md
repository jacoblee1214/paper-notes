# Beyond Implicit Force (arXiv:2607.14578) 분석 노트

*Beyond Implicit Force: Evaluating Explicit Force-Torque Proxies in Action Chunking with Transformers*
(Wong, Liu, Dayoub, Adelaide University, **IROS 2026**)를 Table IV 통계 재계산, 기반 공개 코드(ACT·ALOHA) 대조,
1-DoF 물리 toy(위치 서보의 접촉력, 강성 판별, 전류 토크의 검출 한계, 불일치–힘 관계)로 분석한 자료입니다.
공식 코드는 공개되지 않아(2026-11-09 확인) 논문 코드 대조는 하지 않았습니다.

**구현할 수 있는지:** 공식 재현은 불가능합니다(코드·데이터 미공개, ALOHA Solo 하드웨어 필요).
아이디어(proprio에 effort 연결, 목표를 미래 qpos로 교체)는 ACT·LeRobot에서 몇 줄로 다시 만들 수 있고,
장벽은 GPU보다 하드웨어와 평가 설계입니다(`analysis/04_claim_verification.md` §4).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 관련 논문과의 관계, 관련 GitHub 저장소 |
| `analysis/03_analysis.md` | P 서보 정상상태 해석, 물리 toy T1–T4, Table IV 통계(2×2 상호작용 포함), Fig. 4 판독, 하이퍼파라미터 |
| `analysis/04_claim_verification.md` | 기반 코드 대조(D1–D8), 주장 검증(C1–C11), 논문에 없는 재현 정보, 구현 가능성 |
| `analysis/table_check.py` | Table IV 통계(Fisher, Holm, Wilson, Mantel–Haenszel, Youden J) |
| `analysis/force_proxy_toy.py` | 1-DoF 물리 toy T1–T4 |
| `analysis/plot_figures.py` | 리포트 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/force_proxy_toy.py && python3 analysis/plot_figures.py
```

모두 CPU에서 몇 초 안에 끝납니다. 논문 수치는 스크립트에 출처(표·절 번호)와 함께 적었고, toy의 강성·질량·마찰 값은 가정입니다.

## 출처와 라이선스

- 논문은 **CC BY 4.0**입니다(arXiv abs 페이지 확인, PDF 안에 다른 권리 표기 없음). `report/figures/paper_*.png`는 논문에서 잘라낸 그림이며 © 저자, CC BY 4.0, 출처 arXiv:2607.14578입니다.
- 코드 대조는 `tonyzhaozh/act` @ `742c753`, `tonyzhaozh/aloha` @ `06369f0`(둘 다 MIT)의 공개 코드 기준입니다. 이 저장소에는 그 코드를 넣지 않았습니다.
- 나머지 그래프는 논문 수치와 toy 시뮬레이션으로 직접 그렸습니다. 원문 PDF는 이 저장소에 넣지 않았습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

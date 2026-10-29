# AW-0: Agentic World Action Models (arXiv:2610.30136) 분석 노트

*Agentic World Action Models* (Liu, Niu, Zhang, Shao, …, Fan, Darrell; UC Berkeley, NVIDIA, UPenn, 2026)를
논문 표·그림 재계산, [RoboTwin 2.0 공식 리더보드](https://robotwin-platform.github.io/leaderboard) 대조,
[공식 코드](https://github.com/ZhuoyangLiu2005/agentic-wam)(커밋 `275818e`) 정적 대조, noise 재표본화 numpy toy로 분석한 자료입니다.
학습·평가 재현은 GPU가 필요해서 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소별 출처, 가까운 대안 |
| `analysis/03_analysis.md` | 수치 정합성, 리더보드 대조, 효과 분해, 통계, noise 분석 |
| `analysis/04_code_verification.md` | 논문 ↔ 코드 대조(D1–D11), 주장별 판정(C1–C13), 구현 가능성 |
| `analysis/table_check.py` | Table 11 → Table 1, 본문 수치, Table 12 정합, 루브릭 격자, 검정, 리더보드 대조 |
| `analysis/noise_toy.py` | video noise 분포, 중요도 가중치, 추론 단계별 σ |
| `analysis/plot_figures.py` | 리포트 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
mkdir -p paper/site
curl -L -o paper/paper.pdf https://arxiv.org/pdf/2610.30136 && pdftotext -layout paper/paper.pdf paper/paper.txt
curl -L -o paper/site/robotwin_leaderboard.json https://robotwin-platform.github.io/data/robotwin_leaderboard.json
python3 analysis/table_check.py && python3 analysis/noise_toy.py && python3 analysis/plot_figures.py
```

모두 CPU에서 수 초 안에 끝납니다. 리더보드는 계속 갱신되므로 값과 순위는 달라질 수 있습니다(2026-10-25 갱신본 기준).

## 출처와 라이선스

- 논문은 **CC BY 4.0**입니다(arXiv abs 페이지 확인). `report/figures/paper_*.png`는 논문에서 잘라낸 그림이며 © 저자, CC BY 4.0, 출처 arXiv:2610.30136입니다.
- `robotwin_headline.png`, `robotwin_pertask.png`는 논문 Table 1·11과 RoboTwin 공식 리더보드 데이터로, `ablation_ladder.png`는 논문 Fig. 5·7과 Table 3·4로, `noise_sigma.png`는 논문 Eq. 3–6과 부록 A의 설정으로 직접 그렸습니다.
- 공식 코드(BSD-2-Clause)는 이 저장소에 넣지 않았고, 대조 위치만 파일·함수 이름으로 적었습니다.
- 원문 PDF와 리더보드 데이터 파일은 넣지 않았습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

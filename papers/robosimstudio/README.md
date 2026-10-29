# RoboSimStudio (arXiv:2610.18835) 분석 노트

*RoboSimStudio: An Interactive Toolkit for Human and Agent Rapid Real2Sim of Robot Manipulation Tasks*
(Chen, Xie et al., UC Berkeley · NVIDIA, 2026)를 [공식 저장소](https://github.com/robosimstudio/robosimstudio)(`81f57c9`)와
대조하고, 표·데모 수치를 재계산하고, 영상 기반 물성 식별 toy로 분석한 자료입니다. 도구 설치·실행과 재현은 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | Abstract → Method → 사례 연구 → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소 출처, GitHub 표 |
| `analysis/03_analysis.md` | Preston 마모 검산, 영상 vs 힘 식별성 toy, 강성과 잡기, GaP 곡선 분모, Table 1 구조 |
| `analysis/04_code_verification.md` | 논문 ↔ 코드 대조(D1–D11), 수치 재계산(C1–C7, V1), 구현 가능성 |
| `analysis/table_check.py` | 표·데모 수치 재계산 |
| `analysis/identifiability_toy.py` | 영상·힘 관측의 강성 식별성 toy (numpy) |
| `analysis/plot_figures.py` | 리포트 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/identifiability_toy.py
cd analysis && python3 plot_figures.py
```

모두 CPU에서 1초 안에 끝납니다. 데모 수치는 공식 저장소 `examples/08_demos/wipe_the_bar/`(커밋 `81f57c9`)에서 옮겨 스크립트에 출처와 함께 적었습니다.

## 출처와 라이선스

- 논문은 **CC BY 4.0**입니다(arXiv abs 페이지 확인). `report/figures/paper_fig*.png`는 논문에서 잘라낸 그림이며 © 저자, CC BY 4.0, 출처 arXiv:2610.18835입니다.
- `latch_effect.png`는 공식 저장소(Apache-2.0) README의 측정값으로, `identifiability.png`와 `table1_tally.png`는 직접 계산해 그렸습니다.
- 공식 코드와 원문 PDF는 이 저장소에 넣지 않았습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

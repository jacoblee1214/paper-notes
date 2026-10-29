# DexForce (arXiv:2501.10356) 분석 노트

*DexForce: Extracting Force-informed Actions from Kinesthetic Demonstrations for Dexterous Manipulation*
(Chen, Yu, Choi, Cutkosky, Bohg, Stanford, RA-L 2025)을 그림 수치 재계산, Fig. 9 픽셀 판독, 식 1–3의 toy 검증,
arXiv v1 ↔ v2 대조로 분석한 자료입니다. 공식 코드와 데이터가 공개되지 않아 코드 대조와 재현은 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소별 출처, GitHub 정리 |
| `analysis/03_analysis.md` | 힘 배율 G = k_p·k_f, Fig. 9를 힘 허용 폭으로 읽기, 관측 위치 기준선이 실패하는 조건, 통계 |
| `analysis/04_claim_verification.md` | 수치 재계산(C1–C9), v1 ↔ v2 대조, 재현 정보, 구현 가능성 |
| `analysis/table_check.py` | Fig. 7·8·9와 v1 Fig. 9 재계산 |
| `analysis/fig_digitize.py` | Fig. 9 픽셀 판독 (원문 PDF가 `paper/paper.pdf`에 있어야 함) |
| `analysis/impedance_toy.py` | 임피던스 + force-informed 목표 toy (힘 배율, slide cube, 허용 창) |
| `analysis/plot_figures.py` | 리포트 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib pillow
mkdir -p paper && curl -L -o paper/paper.pdf https://arxiv.org/pdf/2501.10356v2   # fig_digitize.py에만 필요
python3 analysis/fig_digitize.py && python3 analysis/table_check.py && python3 analysis/impedance_toy.py
cd analysis && python3 plot_figures.py
```

모두 CPU에서 몇 초 안에 끝납니다(pdftoppm 필요: poppler).

## 출처와 라이선스

- arXiv v2는 **CC BY 4.0**입니다(arXiv abs 페이지 확인). arXiv PDF에는 충돌하는 권리 표기가 없습니다. 출판된 RA-L 판(IEEE)의 그림은 쓰지 않았습니다.
- `report/figures/paper_fig*.png`는 arXiv v2에서 잘라낸 그림이며 © 저자, CC BY 4.0, 출처 arXiv:2501.10356입니다.
- `kf_gain.png`, `obs_ablation_ci.png`는 논문 그림의 수치로, `toy_impedance.png`는 논문 식 1–3의 toy로 직접 그렸습니다.
- 원문 PDF는 이 저장소에 넣지 않았습니다. 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

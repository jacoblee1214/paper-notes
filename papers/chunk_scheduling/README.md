# Action Chunk Scheduling (arXiv:2608.00337, CoRL 2026 포스터) 분석 노트

*Action Chunk Scheduling for Batched Robot Policy Serving* (Armory, arXiv v1, 2026-07-31)을 표 재계산, 정수 복원 검정, 공식 코드의 정적 대조, 독립 토이 시뮬레이터로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2608.00337에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **논문 수치의 재현은 CPU만으로는 어렵습니다.** 실로봇 10대, L40S 서빙, LIBERO 다중 클라이언트가 필요하고 실로봇 코드는 공개 `main`에 없습니다.
- **스케줄링 로직의 독립 재구현은 가능합니다.** `analysis/toy_mirror_sim.py`(약 15초, numpy)가 표 1의 지연으로 RR, EDF, Lookahead를 구현해 표 4의 기아율을 평균 1–2 %p 오차로 재현합니다.
- **공식 mock 모드(CPU)는 새 설치와 코드 실행이 필요합니다.** 시도하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장 지도 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, GitHub 표 |
| `analysis/03_analysis.md` | 수치 재계산 V1–V12, 토이 시뮬레이터 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D12, 구현 가능성 판정 |
| `analysis/recompute.py`, `paper_tables.py` | 표 재계산과 검정 (numpy, scipy, 수 초). `paper_tables.py`는 논문 표를 옮긴 것 |
| `analysis/extract_fig4.py` | 원문 그림 4의 벡터 값을 읽는 스크립트 (pymupdf, 논문 PDF 경로를 인자로 받음). 결과는 `fig4_extracted.json` |
| `analysis/toy_mirror_sim.py` | 토이 시뮬레이터 (numpy, 약 15초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
cd analysis
python3 recompute.py
python3 toy_mirror_sim.py
cd ..
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문의 arXiv 라이선스는 비독점 배포 1.0(CC BY 아님)입니다. 그래서 논문 그림은 넣지 않았고, 모든 그래프는 우리가 논문의 표를 다시 계산해 그렸습니다.
- 공식 코드는 `github.com/GaTech-RL2/armory`(Apache-2.0)에 있습니다. 이 폴더에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

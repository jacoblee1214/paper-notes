# DexFLEX (arXiv:2610.14372, CoRL 2026 포스터) 분석 노트

*DexFLEX: Contact-Aware Foundation Controller for Command-Guided Dexterity*를 표·그림 수치 재계산, 통계 검정, 주장 검증으로 읽은 자료입니다. 공식 코드는 공개되지 않았습니다(2026-11-22 확인). 논문 원문 PDF는 넣지 않았습니다(arXiv:2610.14372에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **이 맥(Apple M4 Pro, MPS, CUDA 없음)에서 논문 수치를 재현할 수 없습니다.** Isaac Gym 학습 데이터, 촉각 시뮬레이션, 실로봇이 필요하고, 논문이 가리키는 부록이 PDF에 없습니다.
- **핵심 선택 루프의 MuJoCo 축소판은 조건부로 구현할 수 있습니다**(약 2–4주 추정, 측정 전). 결과는 논문과 비교할 수 없습니다.
- **재계산과 toy는 `analysis/recompute.py`로 수 초에 다시 할 수 있습니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, GitHub 표 |
| `analysis/03_analysis.md` | 수치 재계산 V1–V10, 계산·toy K1–K2 |
| `analysis/04_claim_verification.md` | 주장 C1–C12, 구현 가능성 |
| `analysis/recompute.py`, `analysis/recompute_results.json` | 재계산, 검정, 검정력, 선택 toy |
| `analysis/fig3_readout.py`, `analysis/fig3_readout.json` | 그림 3 벡터 좌표 판독 |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장과 논문 그림 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib pymupdf
python3 analysis/fig3_readout.py paper.pdf analysis/fig3_readout.json   # 논문 PDF를 arXiv에서 받아 둘 것
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에서 CC BY 4.0(creativecommons.org/licenses/by/4.0)으로 공개되어 있습니다. 그림 2, 3, 5(막대)를 `report/figures/paper_fig*.png`로 옮겼고 각각 © 저자(Wan 외), CC BY 4.0, 출처 arXiv:2610.14372입니다. 실로봇 사진이 든 그림은 넣지 않았습니다.
- 프로젝트 페이지는 dex-flex.github.io에 있습니다. 공식 코드는 없습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 재계산 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.

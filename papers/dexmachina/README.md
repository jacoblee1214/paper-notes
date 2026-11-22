# DexMachina (arXiv:2505.24853, ICML 2026 포스터) 분석 노트

*DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation* (arXiv v1, 2025-05-30)을 공식 집계 파일 재계산, 기준선 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2505.24853에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **이 맥에서 전체 재현은 어렵습니다.** 학습은 GPU(환경 12,000개)를 전제하고, 공식 코드는 전처리 단계까지 CUDA를 직접 호출합니다.
- **재계산과 검정은 가능하고 완료했습니다.** `analysis/recompute.py`(numpy, scipy)가 수 초 안에 끝납니다. 집계 파일 `repo/dexmachina/eval/stats/dexmachina_main_stats_AUC.json`이 필요하며, 공식 저장소를 얕은 clone하면 됩니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계, 논문 안의 불일치 |
| `analysis/02_prior_work.md` | 계보, 구성 요소별 출처, 공식 저장소와 관련 저장소 |
| `analysis/03_analysis.md` | Fig. 3 재계산, 손 비교, 기준선 검정, 일관성 검사 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 재계산과 결과 |
| `analysis/plot_figures.py` | 그래프 두 장 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py      # repo/ 가 필요
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 초록 페이지에서 CC BY 4.0으로 표시되어 있습니다. PDF에서 저작권 고지는 찾지 못했습니다. 그래도 논문 그림은 패널별 권리를 확인하지 않았으므로 게시하지 않았습니다.
- 공식 코드는 `github.com/MandiZhao/dexmachina`(MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 설치하거나 실행하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 재계산, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 기준선 값은 논문 그림 3의 막대 라벨을 눈으로 읽은 값입니다(0.1 단위).

# EigenDEXplore (arXiv:2610.07681) 분석 노트

*EigenDEXplore: Structured Exploration for Dexterous Manipulation with Human Priors* (Gupta, Lum et al., Stanford, 2026)를
[공식 코드](https://github.com/hgupt3/eigendexplore)(`5d57180`)와 대조하고, 공개 PCA 자산과 벤치마크 설정을 직접 계산하며 분석한 자료입니다. 학습 재현(GPU)은 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (논문 그림 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소별 출처, 관련 GitHub 표 |
| `analysis/03_analysis.md` | 탐색 예산 유도, 공개 preset 비교, ES 전처리와 PG 기울기 분석, toy |
| `analysis/04_code_verification.md` | 논문 ↔ 코드 대조(D1–D9), 수치 재계산(C1–C11), 구현 가능성 |
| `analysis/pca_check.py` | 공개 PCA 자산 검산과 설정별 초기 공분산 |
| `analysis/exploration_toy.py` | SPIDER식 표본 최적화·꼬리 확률·REINFORCE toy |
| `analysis/fig_digitize.py` | Fig. 5(DexMachina) 막대 판독 |
| `analysis/table_check.py` | 본문 수치 재계산 |
| `analysis/plot_figures.py` | 리포트 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib pyyaml pillow
git clone --depth 1 https://github.com/hgupt3/eigendexplore.git repo      # PCA 자산과 설정 (5d57180)
mkdir -p paper && curl -L -o paper/paper.pdf https://arxiv.org/pdf/2610.07681v2   # fig_digitize.py에만 필요 (poppler의 pdftoppm 사용)
python3 analysis/pca_check.py && python3 analysis/exploration_toy.py
python3 analysis/fig_digitize.py && python3 analysis/table_check.py
cd analysis && python3 plot_figures.py
```

모두 CPU에서 몇 초 안에 끝납니다(단일 스레드).

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- `pca_spectra.png`, `noise_budget.png`는 공식 코드(MIT)의 공개 PCA 자산과 설정으로, `toy_cost.png`는 toy 결과로, `dexmachina_gain.png`는 논문 Fig. 5를 판독한 수치로 직접 그렸습니다.
- 공식 코드는 이 저장소에 넣지 않았습니다. 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

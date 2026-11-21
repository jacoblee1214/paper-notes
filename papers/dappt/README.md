# DAPPT: Domain-Aware Policies from Procedural Trajectories for Sim-to-Real Dexterous Tool Use (arXiv:2611.20202) 분석 노트

*DAPPT* (arXiv v1, 2026-11-15)를 표 재계산과 CPU 토이로 읽은 자료입니다. 학회 표기는 없습니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 두 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 학습과 실로봇 재현은 불가능합니다.** IsaacLab은 macOS를 지원하지 않고, 실로봇 하드웨어가 필요합니다.
- **조건부:** 평가 시뮬레이터 Genesis는 Apple Metal을 README에서 언급하지만, 이 맥에서는 확인하지 않았습니다. 강체 도구 궤적 추종 toy는 CPU에서 가능하지만 시뮬레이터 설치가 필요합니다.
- **노력(추정):** 강체 추종 toy 1–2주. 전체 재현은 GPU 기기에서 수 주입니다.
- 공식 저장소(`princeton-vl/DAPPT`)는 README 한 장뿐입니다("Code coming soon"). 코드를 대조하지 못했으므로 `04_claim_verification.md`에서 주장을 표 값과 대조했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, 저장소와 학회 판정 |
| `analysis/03_analysis.md` | 표 재계산(배율, 하락률, Wilson 구간, 표 사이 일치), 토이 1(선택 편향), 검정 가능성 |
| `analysis/04_claim_verification.md` | 주장 C1–C12 대조, 재현 정보 공백, 구현 가능성 판정 |
| `analysis/recompute.py` → `recompute_results.json` | 표 값 재계산과 일관성 검사 (numpy, scipy, 1초 미만) |
| `analysis/toy_selection.py` → `toy_selection_results.json` | 토이 1: 잡음만으로 생기는 최선/평균 배율 (numpy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_selection.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 저장소 `princeton-vl/DAPPT`에는 라이선스가 없습니다. 이 분석에서는 얕은 clone 후 README만 정적으로 읽었습니다(실행, 설치 없음).
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

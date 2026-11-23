# Chunk-wise Model-based Policy Optimization (arXiv:2610.14564, CoRL 2026) 분석 노트

*CMPO* (arXiv v1, 2026-10-10)를 표 재계산, 그림 판독 값 기반 검사, 공식 코드 정적 대조, 작은 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 네 장은 우리가 만든 것입니다. 곡선 값은 논문 PDF의 벡터 경로를 눈금에 맞춰 읽은 값(그림 판독 값)이며 원 데이터가 아닙니다.

## 구현 가능성 (요약)

- **알고리즘의 구현은 가능하고, 논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 표 1에서 산술하면 RTX 4090 기준 19과제 x 3시드 x 1M 스텝이 약 500 GPU-시간입니다. 실로봇과 기준선 코드는 공개되지 않았습니다.
- **공식 코드로 한 과제 한 시드는 조건부입니다.** 새 환경 설치(mujoco 3.1.6 고정 등), `cuda:0` 포팅이 필요하고 속도는 측정 전입니다.
- **핵심 알고리즘(약 700줄)은 torch로 다시 쓸 수 있습니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 보고 수치, 주장과 증거의 간격, 표기 불일치 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산, 그림 판독 값 검사, 토이, 주장 판정 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 구현 가능성, 저자 질문 |
| `analysis/extract_curves.py`, `extract_fig45.py` | 논문 PDF의 벡터 그래프에서 곡선 값을 읽음 (PyMuPDF) |
| `analysis/recompute.py` | 표 수치 재계산과 판독 값 기반 검정 (numpy, scipy) |
| `analysis/toy_chunk_vs_recurrent.py` | 토이: 이중 진자에서 청크 대 재귀 모델 (torch, 기본 약 1.5분) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |
| `analysis/*_summary.json`, `fig45_values.json`, `*_results*.json` | 위 스크립트의 결과 |

## 재실행

원문 PDF는 이 폴더에 없습니다. 먼저 받아 `paper/paper.pdf`에 둡니다.

```bash
mkdir -p paper && curl -L https://arxiv.org/pdf/2610.14564v1 -o paper/paper.pdf
pip install numpy scipy matplotlib pymupdf statsmodels
python3 analysis/extract_curves.py     # analysis/curves_fig7.json 생성
python3 analysis/extract_fig45.py
python3 analysis/recompute.py
pip install torch                      # 토이에만 필요
python3 analysis/toy_chunk_vs_recurrent.py
python3 analysis/toy_chunk_vs_recurrent.py long
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-23). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/song-wx/CMPO)는 MIT 라이선스입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

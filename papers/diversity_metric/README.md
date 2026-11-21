# Diversity You Can Actually Measure (arXiv:2603.11634, CoRL 2026 구두 발표) 분석 노트

*Diversity You Can Actually Measure: A Fast, Model-Free Diversity Metric for Robotics Datasets* (arXiv v2, 2026-10-13)를 표 수치 재계산과 공식 코드의 정적 읽기로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2603.11634에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 그래프는 논문 표 수치로 우리가 다시 그렸습니다.

## 구현 가능성 (요약)

- **공식 코드 실행은 이 맥에서 불가입니다.** 최상위 모듈이 CuPy를 가져오고 CUDA를 전제합니다.
- **엔트로피를 numpy로 독립 구현하는 것은 가능합니다(추정).** 상태와 행동만 쓰고 시그니처 레벨을 낮추면 CPU로 돌아갈 것으로 보입니다. 측정하지 않았습니다.
- **정책 학습 재현(표 5, 6)과 실로봇 재현은 불가입니다.** 학습 설정이 공식 저장소에 없고 GPU와 로봇이 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 5, 6 파싱, 시드 검사, Welch 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py      # paper/paper.txt를 읽어 recompute_results.json을 쓴다
python3 analysis/plot_figures.py   # report/figures/ 에 그래프를 쓴다
```

`paper/paper.txt`는 논문 PDF를 `pdftotext -layout`으로 바꾼 결과입니다. 논문 PDF를 같은 위치에 두면 `pdftotext -layout paper/paper.pdf paper/paper.txt`로 다시 만들 수 있습니다.

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인, v1과 v2 모두). PDF 본문에서 저작권 표시를 찾지 못했습니다.
- 이 저장소는 논문 그림을 넣지 않았습니다. 그래프는 논문 표 수치를 옮긴 값으로 우리가 그렸습니다.
- 공식 코드는 `github.com/S-sirigiri/FAKTUAL`에 있습니다. 라이선스 파일이 없습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다. 정적 읽기만 했습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

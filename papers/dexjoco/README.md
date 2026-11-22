# DexJoCo (arXiv:2605.16257) 분석 노트

*DexJoCo: A Benchmark and Toolkit for Task-Oriented Dexterous Manipulation on MuJoCo* (arXiv v2, 2026-09-22)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2605.16257에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 그래프는 모두 이 노트의 작성자가 그렸습니다.

## 구현 가능성 (요약)

- **논문 수치 재현은 이 맥에서 불가능합니다.** π0.5 학습은 CUDA를 전제하고, 시연 수집은 Rokoko 장갑과 Vive 트래커 하드웨어를 씁니다.
- **공식 MuJoCo 환경 실행과 체크포인트 평가는 조건부입니다.** 공식 코드 실행과 체크포인트 다운로드 승인이 필요합니다.
- **재계산과 공개 수치 검정은 가능하고 이미 끝냈습니다.** `analysis/recompute.py`(표준 라이브러리만, 수 초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, v1과 v2의 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산, 신뢰구간, 검정, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 표 재계산과 검정 결과 |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 그래프 두 장 (우리가 그림) |

## 재실행

```bash
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

`paper/paper.txt`(pdftotext 결과)를 읽습니다. 원문 PDF는 arXiv에서 받아 `paper/paper.pdf`로 두면 됩니다.

## 출처와 라이선스

- 논문은 arXiv에서 비독점 배포(nonexclusive-distrib 1.0)로 공개되어 있습니다. CC BY가 아니므로 논문 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/brave-eai/dexjoco`(MIT)에 있습니다. 이 노트에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다. 번들된 GeoRT는 비상업 조건입니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 재계산 스크립트, 분석 문서는 이 노트의 작성자가 만들었습니다.
- 데이터셋과 가중치는 받지 않았습니다.

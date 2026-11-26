# FingerViP (arXiv:2604.21331, CoRL 2026) 분석 노트

*FingerViP: Learning Real-World Dexterous Manipulation with Fingertip Visual Perception* (arXiv v3, 2026-09-24)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2604.21331에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 그래프는 모두 이 분석에서 새로 그린 것입니다.

## 구현 가능성 (요약)

- **Apple M4 Pro(MPS)에서 논문 결과의 재현은 불가능합니다.** 하드웨어와 시연 데이터가 필요합니다. 정책 구조의 합성 입력 재구현과 속도 측정은 가능해서 해 두었습니다.
- **공식 코드는 `zhzhen0626/FingerViP` (MIT, 하드웨어 폴더는 CC BY-NC-SA 4.0)입니다.** 이 저장소는 정적으로만 읽었습니다. 실행, 설치, 가중치 다운로드는 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | venue 판정, 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 차이와 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy) |
| `analysis/toy_cost.py` | 표 2 연산량의 분석식과 임의 가중치 시간 측정 (torch) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/figures/` | 그래프 세 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_cost.py
python3 analysis/plot_figures.py
```

`recompute.py`의 입력은 논문 v3의 표 값입니다. 원시 시도 기록은 공개되지 않았습니다.

## 출처와 라이선스

- 논문 라이선스는 arXiv 비독점 배포 라이선스(nonexclusive-distrib/1.0)이고, CC 라이선스가 아닙니다. 그래서 논문 그림을 싣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 코드는 저장소에 포함하지 않았습니다.

# MaskWAM (arXiv:2606.13515, CoRL 2026 포스터) 분석 노트

*MaskWAM: Unifying Mask Prompting and Prediction for World-Action Models* (arXiv v1, 2026-06-11)를 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2606.13515에서 받을 수 있고, arXiv 비독점 배포 라이선스입니다).

## 구현 가능성 (요약)

- 공식 학습, RoboTwin, 실로봇 재현은 이 맥(Apple M4 Pro, MPS)에서 불가입니다.
- 공개 체크포인트(약 10.5 GiB) 추론은 다운로드, 설치, 코드 수정이 필요해 조건부입니다.
- 표 재계산은 완료했고(`analysis/recompute.py`), 구조의 축소 재구현은 가능합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 판본·학회·라이선스 |
| `analysis/02_prior_work.md` | 선행 계보, 코드 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스로 공개되어 있습니다(CC BY 아님). 그래서 논문 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/hanyangyu1021/maskwam`(MIT)에 있습니다. 이 폴더에는 코드를 넣지 않았고 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

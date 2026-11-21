# Action Effect Matching (arXiv:2611.25980) 분석 노트

*Learning What Actions Change: Action Effect Matching for Dynamics-Faithful World Models* (arXiv v1, 2026-11-19)을 표 재계산과 수식 검증으로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.25980에서 받을 수 있습니다). 그래프는 모두 우리가 그렸습니다.

## 구현 가능성 (요약)

- **재현은 이 맥에서 불가입니다.** 세계모델 후학습에 GPU가 필요하고, 실로봇 play 데이터도 필요합니다.
- **공식 코드와 프로젝트 페이지는 2026-11-21 기준으로 찾지 못했습니다.**
- **가능한 것은 수치 재계산과 CPU 토이입니다.** `analysis/recompute.py`(수 초), `analysis/toy_estimator.py`(수 초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 근접 대안 |
| `analysis/03_analysis.md` | 재계산 (신뢰구간, 검정), 일관성 검사, 토이 결과 |
| `analysis/04_claim_verification.md` | 코드 부재 확인, 주장 검증, 구현 판정 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy) |
| `analysis/toy_estimator.py` | 추정량 편향 토이 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_estimator.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지의 license 링크로 확인). PDF에 저작권 표시는 찾지 못했습니다. 그래도 논문 그림은 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 코드가 없어 코드를 넣지 않았습니다. 가중치와 데이터셋은 받지 않았습니다.

# ELASTIC (arXiv:2606.31132, CoRL 2026 포스터) 분석 노트

*ELASTIC: Efficiently Learning to Adaptively Scale Test-Time Compute for Generative Control Policies* (arXiv v1, 2026-06-30)를 그림에서 읽은 값의 재계산, 통계 검정, 주장 점검으로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2606.31132에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **논문 수치(확산 정책, LIBERO-10, 실로봇)의 재현은 이 맥에서 불가입니다.** GPU, 시뮬레이터, 로봇이 필요합니다.
- **공식 코드는 찾지 못했습니다.** 메타 정책 학습 코드와 Can 과제 변형이 공개되지 않았습니다.
- **식 1과 메타 MDP를 줄인 2차원 토이는 CPU에서 구현할 수 있습니다**(추정 3–5일).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 보고된 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 저장소 표 |
| `analysis/03_analysis.md` | 그림 값 재계산, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 검증, 구현 가능성 |
| `analysis/recompute.py` | 재계산과 검정 (numpy, scipy, statsmodels, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy statsmodels matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문의 arXiv 라이선스는 non-exclusive(CC BY 아님)입니다. 그래서 논문 그림은 넣지 않았고 우리가 그린 그래프만 넣었습니다.
- 그림에서 읽은 수치(성공률 ±0.3 %p, 지연 ±0.7 ms)는 분석을 위해 옮긴 것이며 원 데이터가 아닙니다.
- 공식 코드와 프로젝트 페이지는 찾지 못했습니다. 논문이 쓴 외부 도구는 각 저장소의 라이선스를 따릅니다.
- 가중치와 데이터셋은 받지 않았습니다.

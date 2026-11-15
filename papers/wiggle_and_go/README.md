# Wiggle and Go! (arXiv:2604.22102, CoRL 2026) 분석 노트

논문 v2의 표를 다시 계산하고, 주장과 실험 설정의 일치를 점검했습니다. 짧은 wiggle의 식별 가능성을 선형 numpy 토이로 봤습니다. 공식 코드가 없어 정적 대조는 하지 않았습니다.

## 구현 가능성 (요약)

- 실기 재현은 이 맥에서 불가합니다(하드웨어 필요).
- 시뮬레이션 파이프라인은 pydrake 설치와 처리량 측정 뒤 조건부로 가능합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 결과, 주장과 실험의 틈, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 표 |
| `analysis/03_analysis.md` | 표 재계산, 일관성 검사, 토이 W |
| `analysis/04_claim_verification.md` | 주장별 판정, 학습 범위 대조, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy) |
| `analysis/toy_wiggle_fisher.py` | 토이 W: Fisher 정보, CRLB, 척도 불변, 모형 오차 (numpy, 수십 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_wiggle_fisher.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv non-exclusive 라이선스입니다. 논문 그림과 원문 PDF는 이 저장소에 넣지 않았습니다.
- 공식 코드는 찾지 못했습니다. 프로젝트 페이지의 사이트 저장소는 웹 페이지 소스입니다.
- 데이터와 가중치는 받지 않았습니다.

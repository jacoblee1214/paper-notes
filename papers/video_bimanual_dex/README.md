# From Human Videos to Bimanual Dexterous Manipulation Data (PriDex, arXiv:2611.21908) 분석 노트

PriDex 논문(arXiv v1, 2026-11-16)을 표 재계산과 통계 검정으로 읽은 자료입니다. 학회 표기는 확인하지 못했습니다. 공식 코드와 프로젝트 페이지는 찾지 못했고, 그래서 주장 대조(04)로 검증했습니다.

## 구현 가능성 (요약)

- **전체 재현은 이 맥에서 불가입니다.** Isaac Sim 5.1(CUDA GPU), 양팔 다지 로봇, 공식 코드가 없습니다.
- **식 검증과 토이 수준의 신뢰도 조건부 제어는 CPU로 가능합니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 식, 실험 |
| `analysis/02_prior_work.md` | 선행연구 계보와 구성 요소별 출처 |
| `analysis/03_analysis.md` | 수치 재계산과 통계 검정 |
| `analysis/04_claim_verification.md` | 주장 대조, 불일치 목록, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 Fisher 검정 (표준 라이브러리, 1초 안팎) |
| `report/plot_own.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
python3 analysis/recompute.py
python3 report/plot_own.py
```

## 출처와 라이선스

- 논문은 arXiv non-exclusive 라이선스(nonexclusive-distrib/1.0)로 공개되어 있습니다. CC BY가 아니므로 논문 그림은 넣지 않았습니다.
- 그래프는 논문 표 수치를 다시 그린 것입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.21908에서 받을 수 있습니다).

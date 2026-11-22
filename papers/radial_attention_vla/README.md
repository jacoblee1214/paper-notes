# Beyond the Sink Interpretation (arXiv:2611.26385) 분석 노트

*Beyond the Sink Interpretation: Radial Attention in Flow-Based VLA Policies* (arXiv v1, 2026-11-19)를 표 재계산, 쌍체 검정, 내부 일관성 검사로 분석한 자료입니다. 공식 코드가 없어 주장 검증(C1–C20)으로 대조했습니다. 논문 원문과 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **논문의 측정(끝점 질량, value 개입, 폐루프 BOS 실험)은 이 맥에서 재현하지 않습니다.** GPU와 시뮬레이터, 긴 실행 시간이 필요합니다.
- **측정 도구 구현은 가능합니다.** attention hook, RoPE 제거, 개입, 쌍체 집계는 PyTorch로 1–2주(추정)입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 표 |
| `analysis/03_analysis.md` | 수치 재계산, 합계와 일관성 검사, 실행 간 변동 |
| `analysis/04_claim_verification.md` | 주장 검증표 C1–C20, 논문 내부 불일치 D1–D6, 구현 가능성 |
| `analysis/recompute.py` | 쌍체 검정과 구간 재계산, 합계 검사 (표준 라이브러리) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스(nonexclusive-distrib 1.0)로 공개되어 있습니다. 이 라이선스는 CC BY가 아니므로 논문 그림은 게시하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
- 원문 PDF는 arXiv:2611.26385에서 받을 수 있습니다. 공식 코드와 체크포인트는 받지 않았습니다.

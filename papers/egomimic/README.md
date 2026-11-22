# EgoMimic (arXiv:2410.24221) 분석 노트

*EgoMimic: Scaling Imitation Learning via Egocentric Video* (arXiv v1, 2024-10-31)를 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2410.24221에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **학습 재현은 이 맥에서 불가능합니다.** 공식 코드가 CUDA를 고정하고, 샘플 데이터는 약 243 GB입니다.
- **정규화 통계와 작은 정책 실험은 가능합니다.** 데이터 다운로드와 직접 짠 코드가 전제입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전과 권리 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산 (Wilson, Fisher), 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 차이 D1–D7, 구현 가능성 |
| `analysis/recompute.py` | 표 수치 재계산과 검정 (표준 라이브러리만, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문 라이선스: arXiv nonexclusive-distrib 1.0 (CC BY가 아님). 논문 그림과 원문을 싣지 않았습니다.
- 공식 코드: GitHub SimarKareer/EgoMimic, MIT. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 분석, 스크립트는 이 저장소 작성자가 만들었습니다.

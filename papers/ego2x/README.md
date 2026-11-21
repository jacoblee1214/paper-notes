# Ego2X (arXiv:2611.24138) 분석 노트

*Ego2X: Contact-aware Transfer from Egocentric Human Data to Any Robot* (arXiv v2, 2026-11-19)을 표 재계산, 정확 검정, 쌍 수 점검으로 읽은 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.24138에서 받을 수 있습니다). 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **이 맥에서 전체 재현은 불가능합니다.** 공식 코드와 데이터가 없고, 단안 영상 파이프라인과 X-VLA 학습은 GPU와 여러 대형 모델을 전제로 합니다.
- **표현과 검사 도구는 가능합니다.** 접촉 기록 스키마, 깊이 보정 식의 소규모 검사는 CPU로 할 수 있습니다.
- **재계산은 완료했습니다.** `analysis/recompute.py`(numpy, scipy, 수 초)로 비율 구간, 정확 검정, 쌍 수 점검을 다시 했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 기록 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 코드와 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 쌍 수 점검 |
| `analysis/04_claim_verification.md` | 주장 C1–C12 검증, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문 라이선스는 arXiv 비독점 배포(`arxiv.org/licenses/nonexclusive-distrib/1.0/`)입니다. CC BY가 아니므로 논문 그림은 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 코드와 프로젝트 페이지는 확인되지 않았습니다(2026-11-21 기준). 공식 코드를 받거나 실행하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다.

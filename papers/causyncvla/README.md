# CauSyncVLA (arXiv:2611.25667) 분석 노트

*CauSyncVLA: Causal Latent World Modeling for Asynchronous Robot Control* (arXiv v1, 2026-11-19)을 표 재계산, 쌍체 검정, 지연 분해, 공식 코드 정적 대조로 분석한 자료입니다. 학회 표기는 없습니다(프로젝트 페이지 "Under review"). 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.25667에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- **이 맥(CUDA 없음)에서 논문 수치 재현은 불가합니다.** 공식 코드는 CUDA, Linux EGL, Python 3.11을 전제합니다. 실로봇 가중치는 비공개입니다.
- **재계산과 검정은 끝냈습니다.** LIBERO 헤드라인 차이는 쌍 정보가 없어 검정이 불가하고, 실로봇 차이는 p = 0.056(풀링 Fisher)입니다.
- **구조 재구현은 가능합니다**(추정 1–2주, 측정하지 않음).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 저자 인정 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, 공식 코드 표 |
| `analysis/03_analysis.md` | 재계산과 검정, 지연 분해, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 비독점 배포 라이선스로 공개되어 있습니다(CC BY 아님). 그래서 논문 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/TwSphinx54/CauSyncVLA`(Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

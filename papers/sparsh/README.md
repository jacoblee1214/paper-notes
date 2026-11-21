# Sparsh (arXiv:2410.24090, CoRL 2024) 분석 노트

*Sparsh: Self-supervised touch representations for vision-based tactile sensing* (arXiv v1, 2024-10-31; CoRL 2024, PMLR 270:885–915)을 보고 수치 재계산, 신뢰구간 점검, 공식 코드의 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다. 재현은 하지 않았습니다.

## 구현 가능성 (요약)

- **전체 사전학습 재현은 이 맥(CUDA 없음)에서 불가능합니다.** A100 8장 기준 설정입니다.
- **공식 가중치 추론과 동결 특징 평가는 조건부입니다.** 가중치 다운로드와 공식 코드 설치 승인이 필요합니다.
- **보고 수치 재계산과 검정은 완료했습니다.** `analysis/recompute.py`(numpy, scipy, 수 초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그래프 세 장, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소와 가중치 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드와 논문 설정의 차이, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 보고 수치의 신뢰구간과 검정 재계산 (numpy, scipy) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장 (직접 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문의 arXiv 라이선스는 nonexclusive 1.0이고 CC BY가 아닙니다. 논문 그림은 싣지 않았습니다.
- 공식 코드(`facebookresearch/sparsh`)는 CC BY-NC 4.0입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

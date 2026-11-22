# SayCan: Do As I Can, Not As I Say (arXiv:2204.01691, CoRL 2022) 분석 노트

*Do As I Can, Not As I Say* (arXiv v2, 2022-08-16, CoRL 2022 camera-ready)를 표 재계산, 공식 노트북 정적 대조, 판본 대조로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 두 장은 우리가 만든 것입니다.

## 요약

- **헤드라인:** v2의 계획 84%, 실행 74%는 PaLM 540B 결과입니다. v1의 70%, 61%는 FLAN 결과였고 v2에서 기준선으로 남았습니다.
- **재계산:** 실제 주방 열의 가족별 값을 더하면 계획 79%, 실행 58%입니다. 보고 총계 81%, 60%와 다릅니다.
- **주장:** "거의 두 배"는 계획 성공률에서 1.13–1.25배입니다.
- **구현 가능성:** 논문 표 재현은 이 맥에서 불가능합니다. 공식 노트북은 CUDA와 유료 API가 필요합니다. 선택 규칙의 CPU 실험은 가능하나 이 분석에서는 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 판본 차이, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(K1–K7), 검정, 주장 판정(C1–C11) |
| `analysis/04_code_verification.md` | 논문과 공식 노트북의 차이(D1–D17), 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 노트북(https://github.com/google-research/google-research/tree/master/saycan)은 Apache-2.0입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

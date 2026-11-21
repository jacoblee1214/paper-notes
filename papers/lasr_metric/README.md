# LASR (arXiv:2611.17775) 분석 노트

*Beyond Prediction Gain: LASR as a Metric for Action-Grounded Latents in World Models* (arXiv v1, 2026-11-13 제출) 을 표 재계산과 합성 검사로 읽은 자료입니다. 학회 표기는 없습니다. 공식 코드와 프로젝트 페이지는 찾지 못했습니다. 분석일 2026-11-21.

## 구현 가능성 (요약)

- **부분 구현 가능.** LASR 평가기와 합성 검사는 CPU로 됩니다. 실데이터 한 표현 검증은 데이터와 체크포인트 승인이 필요합니다.
- **전체 재현은 보류.** 다섯 표현 추출, 매칭 시뮬 패널, 세계모델 예측은 GPU와 대형 자원이 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그림은 우리가 그린 세 장) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 코드 검색 결과, 기준선 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산, 구간과 검정, 합성 검사 두 가지, 미확인 항목 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C20, 구현 가능성 판정 |
| `analysis/recompute.py` | 표 재계산과 합성 검사 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (논문 표 값 입력) |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv abs 페이지에서 비독점 라이선스(nonexclusive-distrib 1.0)로 표시되어 있습니다. CC BY 4.0이 아닙니다. 그래서 논문 그림은 넣지 않았습니다. 본문 PDF에서 저작권 고지는 찾지 못했습니다.
- 그래프와 분석 문서, 스크립트는 이 노트의 작성자가 만들었습니다. 논문 표 값은 인용과 재계산을 위해 옮겼습니다.
- 공식 코드는 찾지 못했습니다. 기준선 저장소는 메타데이터만 확인했고 복제, 설치, 실행하지 않았습니다.
- 원문 PDF와 가중치, 데이터는 넣지 않았습니다. 원문은 arXiv:2611.17775에서 받을 수 있습니다.

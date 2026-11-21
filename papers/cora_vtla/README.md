# CoRA-VTLA (arXiv:2611.23963) 분석 노트

*CoRA-VTLA: Contact-Guided Attention and Queryable Response Memory for Tactile Post-Training* (arXiv v1, 2026-11-18)을 표 재계산, 통계 검정, 주장 대조로 분석한 자료입니다. 학회 표기는 없습니다. 공식 코드는 아직 없습니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 비독점 라이선스). 그래프는 우리가 논문 표 값으로 그렸습니다.

## 구현 가능성 (요약)

- **구현은 조건부입니다.** 공식 코드가 없고 데모 수, 학습 설정, 시드가 본문에 없습니다.
- **재현은 불가입니다.** 학습은 GPU가 필요하고 MuJoCo 과제와 실로봇 하드웨어는 공개되어 있지 않습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 결과 |
| `analysis/02_prior_work.md` | 선행연구 계보, 가까운 대안, 공식 저장소 상태 |
| `analysis/03_analysis.md` | 표 재계산, 검정, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 대조, 구현 가능성, 라이선스 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스(nonexclusive-distrib/1.0)입니다. CC BY가 아니므로 논문 그림은 싣지 않았습니다.
- 공식 코드는 아직 공개되지 않았습니다. 프로젝트 페이지의 "coming soon" 표시를 기준으로 했습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.

# HolidayFactory 1.0 (arXiv:2611.11789) 분석 노트

*HolidayFactory 1.0: Simulating, Generating, and Evaluating Industrial Robot Tasks* (Holiday Robotics 외 13인, arXiv v1 2026-11-09)을 표 재계산, 검정, 검정력 토이, 공개 자료 확인으로 분석한 자료입니다. 논문 원문과 그림은 넣지 않았습니다(arXiv 비독점 배포 라이선스). 학회 채택은 확인하지 못했습니다(프리프린트로 기록).

## 판정 한 줄

공식 코드와 자산이 공개되어 있지 않아 **논문 재현은 불가능합니다.** 이 맥(CUDA GPU 없음)에서는 평가 프로토콜(작업 이벤트, 실행 단위 검정)만 CPU로 구현할 수 있습니다.

## 파일

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 시뮬레이터, 생성 파이프라인, 벤치마크, 한계 |
| `analysis/02_prior_work.md` | 선행연구, 관련 분석 연결, 공개 자료 확인 결과 |
| `analysis/03_analysis.md` | 표 재계산, 쌍 검정 구간, 유도, 검정력 토이 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C21, 공개 증거, 구현 가능성 |
| `analysis/table_check.py` | 표 재계산과 검정 (1초 미만) |
| `analysis/power_toy.py` | 작업 독립 검정과 실행 검정의 1종 오류, 검정력, 포함 확률 (수 초, numpy·scipy) |
| `analysis/plot_figures.py` | 그래프 세 장 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/power_toy.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.** 원문은 arXiv에서 받을 수 있습니다.
- 공식 코드 저장소는 찾지 못했습니다(조직 공개 저장소 0개, 논문에 코드 링크 없음). 그래서 클론하지 않았습니다.
- 토이의 가정(실행당 작업 수, 실행 효과 크기, 기준 성공률)은 논문에 없는 설계 값입니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

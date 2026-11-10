# ContactBench 1.0 (arXiv:2610.24512) 분석 노트

*ContactBench 1.0: A Benchmark for Everyday Contact-Rich Robotic Manipulation* (Xiang, Wang, Bohg, Yuan; arXiv v2 2026-10-23)을 표 재계산, 검정, 검정력 토이, 공식 시뮬레이션 코드 정적 대조로 분석한 자료입니다. 논문 원문과 그림은 넣지 않았습니다(arXiv 비독점 배포 라이선스). 학회 채택은 확인하지 못했습니다(프리프린트로 기록).

## 판정 한 줄

이 맥(CUDA GPU 없음)에서 벤치마크 재현은 불가능합니다. 공식 시뮬레이터는 Linux와 CUDA GPU가 필요하고, 정책 평가 파이프라인은 저장소에 없습니다. 실물 재현도 불가능합니다. 할 수 있는 것은 표 재계산, 검정, toy입니다.

## 파일

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 과제, 시뮬, 평가 프로토콜, 결과, 한계 |
| `analysis/02_prior_work.md` | 선행연구, v1 → v2 차이, GitHub 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산, 유의성(Fisher, Wilson, CMH, Holm, Newcombe), 검정력, 토이 |
| `analysis/04_code_verification.md` | 공식 저장소와 논문 대조(D1–D20), 구현 가능성, 권리 |
| `analysis/table_check.py` | 표 재계산 (1초 미만) |
| `analysis/power_toy.py` | 검정력, 초기 조건 공유, 과제 가중치 토이 (약 20초, numpy 단일 스레드) |
| `analysis/plot_figures.py` | 그래프 네 장 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/power_toy.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- 공식 시뮬레이션 저장소는 Michaelwyx/contactbench-sim(MIT)입니다. 클론하지 않았고, 대조 결과만 `04_code_verification.md`에 있습니다.
- 그래프는 논문 표의 수치와 직접 만든 토이로 그렸습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

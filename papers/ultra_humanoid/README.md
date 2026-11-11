# ULTRA (arXiv:2603.03279) 분석 노트

*ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation* (He, Xu, Li, Dong, Bian, Wang, Gui; arXiv v3 2026-10-17)을 표 재계산, 검정, 토이, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문과 표·그림은 넣지 않았습니다(arXiv 비독점 배포 라이선스). IROS 2026 채택(Best Mobile Manipulation Paper Award Finalist)으로 기록합니다.

## 판정 한 줄

공식 학습 파이프라인과 실물 재현은 CUDA GPU와 실물 장비가 필요해 이 자료에서 재현하지 않았습니다. 학생 가중치가 공개되지 않아 핵심 표는 저장소만으로 재현할 수 없습니다. 공개 교사 가중치의 CPU 추론은 조건부로 가능할 수 있으나 미검증입니다.

## 파일

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용, 논문 표 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 표 읽기, 한계, 주장 수준 |
| `analysis/02_prior_work.md` | 버전 차이, 선행연구, GitHub 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산, Fisher·Wilson·Holm·McNemar 범위, 검정력, 토이 |
| `analysis/04_code_verification.md` | 논문 ↔ 공식 코드 대조(D1–D24), 구현 가능성 |
| `analysis/table_check.py` | 표 재계산과 검정 (1초 미만) |
| `analysis/power_toy.py` | 실물 소표본 검정력 (수십 초) |
| `analysis/retarget_toy.py` | 리타게팅 토이: 토크 실행 가능성 (1초 미만) |
| `analysis/plot_figures.py` | 그래프 네 장 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy statsmodels matplotlib
python3 analysis/table_check.py
python3 analysis/retarget_toy.py
python3 analysis/power_toy.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포). 이 저장소에는 **논문 표와 그림, 원문 PDF를 넣지 않았습니다.**
- 공식 코드는 github.com/Sirui-Xu/ULTRA(Apache-2.0, 커밋 146ddd5)입니다. 클론하지 않았고, 대조 결과만 `04_code_verification.md`에 있습니다.
- 교사 가중치는 AMASS와 BONES-SEED 데이터 조건을 따릅니다(비상업 연구·교육). 내려받아 실행하지 않았습니다.
- 그래프는 논문 표의 수치와 직접 만든 토이로 그렸습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

# TactileReflex (arXiv:2605.23568, IROS 2026 채택) 분석 노트

*TactileReflex: Noise-Statistics-Driven Vision-Tactile Reflex Control for Force-Sensitive Manipulation* (IROS 2026 채택)을 표 재계산, 통계 검정, 임계값 유도 검토, 공식 저장소 대조, numpy 토이 두 개로 분석한 자료입니다. 논문 원문과 그림은 넣지 않았습니다(라이선스).

## 구현 가능성 (요약)

- **공식 재현:** MC-Tac 센서 두 개, Piper 그리퍼, CAN 어댑터, 리눅스 PC가 있을 때만 가능합니다. 카메라 탐지와 CAN이 리눅스 전용입니다.
- **논문 수치 재현:** Sy 정의와 보정 절차가 저장소 두 커밋 사이에서 바뀌었으므로 보장되지 않습니다.
- **재구현 노력:** 제어 로직은 쉽고, 센서 동기화와 실물 튜닝이 실제 노력입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 저자 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, 버전 차이, 관련 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산, Fisher·Wilson·Holm 검정, 민감도, 임계값 유도, 토이 결과 |
| `analysis/04_code_verification.md` | 논문 ↔ 코드 대조 항목, 커밋 차이, 구현 가능성 |
| `analysis/table_check.py` | 표 재계산과 검정 (scipy, 약 3초) |
| `analysis/toy_threshold.py` | 토이 A: 노이즈 분위수 대 표본 내 튜닝, 드리프트 (numpy, 약 3초) |
| `analysis/toy_control.py` | 토이 B: 검출 지연, 채널 절제, 여유 스윕 (numpy, 약 2초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
OMP_NUM_THREADS=1 python3 analysis/toy_threshold.py
OMP_NUM_THREADS=1 python3 analysis/toy_control.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.** 원문은 arXiv:2605.23568에서 받을 수 있습니다.
- 공식 코드 저장소 shayfeng/tactile-reflex-open-source에는 라이선스 파일이 없습니다. 코드는 옮기지 않았고, 대조는 파일과 줄 번호로만 했습니다. 저장소를 실행하지 않았습니다.
- 그래프와 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다. 토이의 파라미터는 설계 가정이며 논문 값에 맞추지 않았습니다.

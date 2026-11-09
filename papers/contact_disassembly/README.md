# CONTACT (arXiv:2603.08560) 분석 노트

*CONTACT: CONtact-aware TACTile Learning for Robotic Disassembly* (Saka et al., IROS 2026)을 표 재계산, 통계 가정 점검, 공식 코드 대조, numpy 토이 두 개로 분석한 자료입니다. 논문 원문과 그림은 넣지 않았습니다(라이선스).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 결과 표 요약 |
| `analysis/02_prior_work.md` | 선행연구, 관련 GitHub 저장소 표, v1 → v2 차이 |
| `analysis/03_analysis.md` | 표 재계산, 시뮬·실물 유의성, 검정력, 토이 결과 |
| `analysis/04_code_verification.md` | 공식 저장소 대조(D1–D20), 구현 가능성 판정 |
| `analysis/table_check.py` | 표 재계산 (1초 미만) |
| `analysis/concat_toy.py` | 토이 A: 중복 입력 결합, 힘장 척도 (numpy 1스레드, 약 17초) |
| `analysis/lowdim_toy.py` | 토이 B: 2차원 요약과 원시 영상 (numpy 1스레드, 약 40초) |
| `analysis/plot_figures.py` | 리포트 그래프 네 장 |
| `report/crop_figures.py` | 논문 그림을 잘라내는 스크립트(개인용 리포트 전용, 공개본 그림에는 쓰지 않음) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/concat_toy.py
python3 analysis/lowdim_toy.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- 공식 코드 저장소는 AdeeshDesai/CONTACT(MIT)입니다. 이 저장소에는 클론하지 않았고, 대조 결과만 `04_code_verification.md`에 있습니다.
- 그래프는 논문 표의 수치와 직접 만든 토이로 그렸습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

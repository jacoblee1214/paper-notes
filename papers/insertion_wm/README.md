# Generalizable Robotic Insertion with World Models (arXiv:2609.28258) 분석 노트

*Generalizable Robotic Insertion with World Models* (Hansen 외 10인, arXiv:2609.28258v1)을
수치 재계산, 통계 검정, 2D toy로 분석한 자료입니다.
학회는 arXiv 메타데이터에 "IROS 2026"으로만 적혀 있고, 논문에 채택 문구는 없습니다(상태 불명).
공식 코드는 찾지 못했습니다(2026-11-10 확인).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림·표 제외) |
| `analysis/01_paper_analysis.md` | 논문 분석: Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 관련 GitHub 저장소 |
| `analysis/03_analysis.md` | 수치 재계산과 유의성, 계획 계산량 추정, toy 결과 |
| `analysis/04_claim_verification.md` | 주장별 판정, 빠진 정보, 구현 가능성 |
| `analysis/table_check.py` | 수치 재계산 (약 1초) |
| `analysis/insertion_toy.py` | 2D peg-in-hole toy: MPPI(모델 오차 주입)와 모델 없는 kNN 정책 (CPU, 약 73초) |
| `analysis/plot_figures.py` | 직접 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py && python3 analysis/insertion_toy.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림, 논문 표 이미지, 원문 PDF를 넣지 않았습니다.**
- 그래프는 논문 수치를 눈으로 읽어 다시 그리거나, 직접 만든 toy 결과로 그렸습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

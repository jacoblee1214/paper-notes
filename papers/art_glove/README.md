# ART-Glove (arXiv:2606.16370) 분석 노트

*ART-Glove: Articulated Tactile Glove for Contact-Grounded Dexterous Interaction Capture* (Lin, Zhao)을 산술 재계산, 공개 기계 파일 대조, numpy 토이 세 개로 분석한 자료입니다. 학회 위치는 확인하지 못했습니다(arXiv 댓글과 저널 참조 없음). 논문 원문과 그림은 넣지 않았습니다(arXiv 비독점 배포 라이선스, CC BY 아님).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 설계, 센싱, 실험, 한계 |
| `analysis/02_prior_work.md` | 선행연구, 관련 저장소 표, 공개 상태 조사 |
| `analysis/03_analysis.md` | 재계산, 유도(히스테리시스 편향, 드리프트, 접촉 항 선형화), 토이 결과 |
| `analysis/04_code_verification.md` | 공개 저장소 대조와 주장 검증, 구현 가능성 판정 |
| `analysis/table_check.py` | 산술 재계산과 BOM 합계 (표준 라이브러리) |
| `analysis/taxel_loc_toy.py` | 토이: 접촉 위치 오차 vs taxel 간격과 잡음 (numpy, 1스레드) |
| `analysis/hysteresis_toy.py` | 토이: 히스테리시스와 드리프트가 힘 추정에 주는 오차 (numpy, 1스레드) |
| `analysis/retarget_contact_toy.py` | 토이: 접촉 라벨을 관절 공간 리타게팅에 넣었을 때 (numpy, 1스레드) |
| `analysis/extra_taxel_grid.py` | 추가 검증: taxel 간격, 접촉 폭, 잡음에 따른 위치 오차와 2 mm 기준 (numpy, 1스레드, 약 1초) |
| `analysis/plot_figures.py` | 그래프 네 장 |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/table_check.py       # BOM 파일 경로는 공식 저장소 복제본이 있어야 합계가 나옴
python3 analysis/taxel_loc_toy.py
python3 analysis/hysteresis_toy.py
python3 analysis/retarget_contact_toy.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- 공식 저장소(기계 파일과 BOM만 공개, 라이선스 표기 없음)는 복제하지 않았습니다. 대조 결과만 적었습니다.
- 그래프와 분석 문서, 스크립트는 직접 만들었습니다. 토이의 계수와 설정은 모두 가정입니다.

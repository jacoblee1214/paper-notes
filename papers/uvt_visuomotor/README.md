# UVT (arXiv:2608.03563) 분석 노트

*Unified Visuomotor Targets: Supervising VLAs Beyond Physical Actions*
(Feng, Jain, IROS 2026 채택)을 표 재계산, 통계 검정, 공식 코드 대조, 선형 toy로 분석한 자료입니다.
공식 저장소는 `DeFisch/UVT`(라이선스 파일 없음, 커밋 `4c46246`)입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 표·그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 주장과 증거 수준, 빠진 정보 |
| `analysis/02_prior_work.md` | 선행연구 계보, 가까운 대안, GitHub 자료 |
| `analysis/03_analysis.md` | LIBERO와 실물 검정, 유도, 선형 toy |
| `analysis/04_code_verification.md` | 논문 ↔ 공식 저장소 대조(D1–D27), 구현 가능성 판정 |
| `analysis/table_check.py` | 표 재계산과 검정 (수 초) |
| `analysis/uvt_toy.py` | 선형 toy: 동역학 고유 이득과 목표 압축 효과 (numpy 1스레드, 수 초) |
| `analysis/plot_figures.py` | 직접 만든 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/uvt_toy.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에서 CC BY-NC-SA 4.0입니다. 그래서 이 저장소에는 **논문 그림, 논문 표 이미지, 원문 PDF를 넣지 않았습니다.**
- 그래프는 논문 표의 수치와 직접 만든 toy로 그렸습니다.
- 공식 코드는 복사하지 않았고, 파일 이름과 줄 번호로만 인용했습니다. 저장소에는 LICENSE 파일이 없습니다.

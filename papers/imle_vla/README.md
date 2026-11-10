# IMLE-VLA (arXiv:2609.10915) 분석 노트

*IMLE-VLA: Fast Single-Step Action Generation for Vision-Language-Action Policies*
(Hosseinkhani 외, IROS 2026 채택)을 표 재계산, 지연 산술, 공식 코드 대조, 다봉 toy로 분석한 자료입니다.
공식 코드는 `kianhk6/IMLE-VLA`(Apache-2.0, 커밋 `0f08e32`)입니다(2026-11-10 확인).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림·표 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations, 빠진 정보 |
| `analysis/02_prior_work.md` | 선행연구 계보, 가까운 대안, GitHub·Hugging Face 자료 |
| `analysis/03_analysis.md` | LIBERO·실물 검정, 지연 산술과 분해, VLA-WC 불일치, 유도, toy |
| `analysis/04_code_verification.md` | 논문 ↔ 공식 저장소 대조(D1–D15), 구현 가능성 판정 |
| `analysis/table_check.py` | 표 재계산과 지연 산술 (수 초) |
| `analysis/imle_toy.py` | 4봉우리 링에서 cIMLE 대 흐름 toy (numpy 1스레드, 약 3분). 결과와 샘플 저장 |
| `analysis/plot_figures.py` | 직접 만든 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/imle_toy.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림, 논문 표 이미지, 원문 PDF를 넣지 않았습니다.**
- 그래프는 논문 표와 그림의 수치, 그리고 직접 만든 toy로 그렸습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다. 공식 코드는 복사하지 않았고, 파일 이름과 줄 번호로만 인용했습니다.

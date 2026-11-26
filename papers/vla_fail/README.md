# VLA-FAIL: Efficient Task Failure Detection for Finetuned Vision-Language-Action Models (arXiv:2606.21386, CoRL 2026) 분석 노트

*VLA-FAIL* (arXiv v2, 2026-10-09)을 표 재계산, 정수 복원, 작은 numpy 토이로 분석한 자료입니다. 논문 원문은 넣지 않았고, 논문 그림은 CC BY 4.0이라 두 장(그림 2, 그림 3)만 출처와 함께 넣었습니다. 공식 코드 저장소는 2026-11-26 기준 확인하지 못해 코드 대조 대신 주장 대조를 썼습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 약 24 GiB, CUDA 없음)에서 불가능합니다.** 실제 로봇 6과제 데이터와 미세조정 체크포인트가 없고, LIBERO-Plus 9,608 롤아웃 평가는 GPU가 필요합니다.
- **방법 자체(LLMD, ACC, FAIL, AUCPDT)는 소형 정책에서 구현할 수 있습니다.** 표 재계산과 토이 세 개는 numpy로 끝냈습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 설계, 결과 표, 한계, v1 ↔ v2 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(V1–V12), 표 사이 일치(K1–K4), 토이 1–3 |
| `analysis/04_claim_verification.md` | 주장 C1–C16 대조, 평가 설계 점검, 구현 가능성, 저자 질문 |
| `analysis/extract_tables.py` | 논문 PDF의 표를 글자 크기와 좌표로 읽어 `tables_raw.json`으로 저장 (PyMuPDF, 원문 PDF 필요) |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, scipy, 수 초) |
| `analysis/toy_metrics.py` | 토이 1·2: AUCPR·AUCPDT 구현, 롤아웃 재표본 변동, 신호 없는 검출기 (numpy, 수 초) |
| `analysis/toy_llmd_cov.py` | 토이 3: LLMD 공분산, 유효 표본 수 x λ (numpy, 약 20초) |
| `analysis/plot_figures.py` | 그래프 다섯 장 생성 (matplotlib) |
| `report/crop_figures.py` | 논문 그림 2, 3을 원문 PDF에서 자름 (원문 PDF 필요) |

## 재실행

```bash
pip install numpy scipy matplotlib pymupdf pillow
curl -L -o paper/paper.pdf https://arxiv.org/pdf/2606.21386v2   # extract_tables.py, crop_figures.py에만 필요
python3 analysis/recompute.py
python3 analysis/toy_metrics.py
python3 analysis/toy_llmd_cov.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv abs 페이지에서 CC BY 4.0(`creativecommons.org/licenses/by/4.0/`)으로 표시되어 있습니다(2026-11-26 확인). 논문 그림 2, 3은 이 라이선스에 따라 넣었고 각 그림에 "© 저자, CC BY 4.0, 출처 arXiv:2606.21386"을 표기했습니다. 원문 PDF는 넣지 않았습니다.
- 프로젝트 페이지: https://vla-fail.github.io/vla-fail-2026
- 공식 코드 저장소는 2026-11-26 기준 확인하지 못했습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

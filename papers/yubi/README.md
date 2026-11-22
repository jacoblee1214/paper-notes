# YUBI (arXiv:2606.10244, CoRL 2026 포스터) 분석 노트

*YUBI: Yielding Universal Bidigital Interface for Bimanual Dexterous Manipulation at Scale* (arXiv v1, 2026-06-08)을 보고 수치 재계산, 통계 검정, 공개 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2606.10244에서 받을 수 있고, 논문 라이선스는 CC BY-SA 4.0입니다).

## 구현 가능성 (요약)

- **논문 실험(사용자 연구, 로봇 배치, 수집)의 재현은 이 맥(Apple M4 Pro, CUDA 없음)에서 불가입니다.** 장비, 사람, GPU, 접근 제한 데이터가 필요합니다.
- **수치 재계산과 일관성 검사는 끝냈습니다.** `analysis/recompute.py`(수 초, numpy와 scipy).
- **공개 코드는 하드웨어와 수집 스택입니다.** 필터, 보정, 정책 학습 코드는 없습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V12, 주장 판정 C1–C20 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D17, 라이선스, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 표 수치 재계산과 검정 |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv abs 페이지의 license 링크는 CC BY-SA 4.0입니다. 공유 조건 때문에 논문 그림은 쓰지 않았습니다.
- 공식 저장소는 `github.com/Toyota/yubi-hw`(CERN-OHL-W-2.0)와 `github.com/airoa-org/yubi-sw`, yubi-core, yubi-app, rebake(Apache-2.0)입니다. 코드는 정적으로 읽기만 했고 이 폴더에 넣지 않았습니다.
- 데이터셋은 접근 제한이라 열람하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

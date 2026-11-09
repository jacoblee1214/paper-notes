# You don't need long context (arXiv:2611.05306) 분석 노트

*You don't need long context* (Belcak et al., NVIDIA, arXiv:2611.05306v1, 2026-11-03)를 논문 표 재계산과
구조별 비용 모델로 분석한 자료입니다. 공식 코드는 공개되지 않았습니다(2026-11-09 확인).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 빠진 기준선, GitHub 자료 |
| `analysis/03_analysis.md` | "7배"의 구조 의존성, 비용 분해, 기준선 가격 규칙, 정확도 통계, 긴 입력 기준선 길이 |
| `analysis/04_claim_verification.md` | 주장별 판정(C1–C16), 비공식 구현 대조(U1–U9), 구현 가능성 |
| `analysis/table_check.py` | Table 1–17과 Eq. 1–4 재계산 |
| `analysis/cost_model.py` | 메모리 한계 서빙 모델, 배치 1 지연, r 민감도, 요청 단위 가격 몬테카를로 |
| `analysis/plot_figures.py` | 리포트 그래프 네 장 |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/table_check.py && python3 analysis/cost_model.py && python3 analysis/plot_figures.py
```

`cost_model.py`의 Nemotron 3.5 Super VL 값(FP8 KV 4,096 B/토큰, SSM 상태 FP16 80 MiB + conv 2.34 MiB)은 공개 config에서 계산한 값으로 스크립트에 상수로 들어 있습니다.

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스, PDF에 "© 2026 NVIDIA. All rights reserved."). 그래서 이 폴더에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- 그래프는 논문 표의 수치와 직접 만든 비용 모델로 그렸습니다.
- 분석 문서와 스크립트는 이 폴더 작성자가 만들었습니다.

# PaceVLA (arXiv:2610.25961) 분석 노트

*PaceVLA: Adaptive Temporal Action Quantization in Vision-Language-Action Models*
(Jeon, Lee et al., KAIST · RLWRLD · 고려대 · 서울대, 2026)을 표·그림 재계산과 지표·추적 toy로 분석한 자료입니다.
공식 코드는 공개되지 않았습니다(2026-10-29 확인). 관련 공개 코드(SAIL 지표 함수, VLASH 양자화 루프, GR00T RoboCasa 과제 표)는 대조했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations, v1 ↔ v2 차이 |
| `analysis/02_prior_work.md` | 선행연구 계보, 구성 요소별 출처, 관련 GitHub 저장소 표 |
| `analysis/03_analysis.md` | TPR·ES 해부, t_max 민감도, 추적 toy, SO(3) 집계 toy, 실물 결과 구조 |
| `analysis/04_claim_verification.md` | 재현 정보 체크리스트, 주장별 판정(C1–C17), 구현 가능성 |
| `analysis/table_check.py` | Table 1–6, 본문, Fig. 1(b) 재계산 |
| `analysis/quant_toy.py` | SO(3) 집계, 2차원 추적 toy, ES의 t_max 의존성 (numpy, 단일 스레드, 10초 이내) |
| `analysis/plot_figures.py` | 리포트 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
cd analysis && python3 table_check.py && python3 quant_toy.py && python3 plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.** 원문은 https://arxiv.org/abs/2610.25961 에 있습니다.
- 그래프는 논문 표의 수치(`real_tradeoff.png`, `es_vs_tmax.png`)와 직접 만든 toy(`toy_window.png`)로 그렸습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

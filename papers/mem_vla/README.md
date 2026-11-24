# MEM: Multi-Scale Embodied Memory for Vision Language Action Models (arXiv:2603.03596, CoRL 2026) 분석 노트

*MEM* (CoRL 2026 스포트라이트, arXiv v2 2026-03-08)을 막대 그림 판독과 시행 수 복원, Fisher·Wilson·정규 근사 재계산, 내부 일관성 검사, 토이 두 개로 분석한 자료입니다. 논문에는 표가 없어 결과를 PDF 벡터 그림에서 읽었습니다.

## 구현 가능성 (요약)

- **재현:** π0.6-MEM(약 5.3B) 학습·추론과 실로봇 평가는 가중치, 코드, 데이터가 공개되지 않아 이 맥(M4 Pro, CUDA 없음)에서 불가능합니다. 공식 코드는 확인하지 못했습니다.
- **소규모 재구현:** 영상 인코더 층(시공간 분리 어텐션, 과거 토큰 삭제)과 언어 기억의 축소판은 작은 모델로 가능합니다(1–5일, 추정).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 방법, 실험 설정, 버전 대조, 그림 판독으로 읽은 것 |
| `analysis/02_prior_work.md` | 구성 요소별 출처, 코드와 자료 표 |
| `analysis/03_analysis.md` | 그림 판독과 시행 수 복원, 재계산(V1–V22), 토이 결과 |
| `analysis/04_claim_verification.md` | 주장 C1–C14, 일관성 K1–K9, 공개 코드 정적 대조 O1–O7, 빠진 값 M1–M8, 구현 가능성 |
| `analysis/extract_figures.py`, `fig_readings.json` | 논문 PDF 벡터에서 막대 높이·오차 막대·지연 곡선 읽기 (원문 PDF 필요) |
| `analysis/recompute.py`, `recompute_results.json` | 읽은 값 재계산, 시행 수 복원, 검정 (scipy, 수 초) |
| `analysis/toy_memory.py`, `toy_memory_results.json` | 토이: 시간 어텐션 K = 1 동일성, 지시문 이력 분포 이동 (torch CPU, 수 초) |
| `analysis/plot_figures.py` | 우리가 그린 그래프 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib torch pymupdf
# 원문 PDF를 paper/paper.pdf 로 내려받아 둠: https://arxiv.org/pdf/2603.03596v2
python3 -I analysis/extract_figures.py
python3 -I analysis/recompute.py
python3 -I analysis/toy_memory.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY 4.0입니다(arXiv abs 페이지 확인 2026-11-24). `report/figures/paper_fig*.png`는 논문 그림 3, 6, 8, 9, 10의 그래프 부분을 잘라 쓴 것이며 © Torne 외, CC BY 4.0, 출처 arXiv:2603.03596입니다. 사진이 들어간 그림은 넣지 않았습니다. 원문 PDF는 이 저장소에 넣지 않았습니다.
- 그림에서 읽은 값은 원자료가 아닙니다. 시행 수는 막대 값의 정수성과 오차 막대에서 복원한 가정입니다.
- 공식 코드는 확인하지 못했습니다.
- 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다. 토이의 파라미터는 설계 가정이며 논문 값에 맞추지 않았습니다.

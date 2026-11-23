# ArtManip (arXiv:2609.12498, CoRL 2026 포스터) 분석 노트

*ArtManip: Category-Level Articulated In-Hand Manipulation* (arXiv v1, 2026-09-11)을 표 재계산, 군집 구간 추정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2609.12498에서 받을 수 있습니다). 논문 그림은 세 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **교사 학습, grasp 합성, 시뮬 평가, 실로봇은 이 맥(Apple M4 Pro, CUDA 없음)에서 불가입니다.** IsaacGym(Linux, CUDA)과 NVIDIA GPU 전용 grasp 합성, Sharpa 손이 필요합니다.
- **표 재계산은 완료했습니다.** `analysis/recompute.py`(수 초, numpy와 scipy).
- **가볍게 해 볼 수 있는 것:** 에셋 생성기(numpy 수준으로 보임), 관절 하나의 물체를 쥔 단순한 손의 축소 모형(직접 구현 2–3주 추정, 측정 전).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 세 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V, 일관성 K, 주장 판정 C |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D16, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 세 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지의 라이선스 링크 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 세 장(그림 2, 그림 3 위쪽, 그림 5)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2609.12498"). 실로봇 사진과 시뮬레이션 장면 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/youngcv/artgym`에 있습니다. 루트에 라이선스 파일이 없고, 서브모듈 `func_lygra`의 LICENSE는 CC BY-NC입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

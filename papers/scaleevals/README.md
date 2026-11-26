# ScaleEvals (arXiv:2611.28411, CoRL 2026) 분석 노트

*ScaleEvals: Scaling Generalist Policy Evaluation through Reset-Free Autonomous Play* (arXiv v1, 2026-11-20)를 표 재계산, 통계 검정, 작은 모의 실험으로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.28411에서 받을 수 있습니다). 논문 그림은 차트 세 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **판정: 부분적으로 가능.** PlayArena 통계 파이프라인(매칭, 소프트 BT, ANOVA, 조건부 예측)과 시뮬레이션 축소판은 노트북(CPU)에서 구현할 수 있습니다.
- 논문의 실험(DROID Franka 여러 대, 정책 5개 서빙, 상용 VLM 채점, 3,000+ 롤아웃)은 로봇과 GPU, 공개되지 않은 데이터가 필요합니다.
- 공식 코드는 2026-11-26 기준 확인하지 못했습니다. 프로젝트 페이지: https://scaleevals.github.io/

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 세 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 관련 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V13, 모의 실험 두 개 |
| `analysis/04_claim_verification.md` | 주장 C1–C20 대조, 구현 가능성, 저자에게 물을 점 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_bt_matching.py` | 모의 실험: 소프트 BT와 Hessian 표준오차 (numpy) |
| `analysis/toy_prompt_order.py` | 모의 실험: 프롬프트당 8회로 순서 예측의 검정력 (numpy) |
| `analysis/plot_figures.py` | 우리가 그린 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 세 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_bt_matching.py
python3 analysis/toy_prompt_order.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지의 license 링크 확인, 2026-11-26). 원문 PDF에 저작권 표시가 없습니다.
- 논문 그림 세 장(그림 4, 5, 16)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2611.28411"). 사진, 장면 화면, 대시보드 화면은 쓰지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 모의 실험, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 롤아웃 영상, 가중치, 데이터셋은 받지 않았습니다.

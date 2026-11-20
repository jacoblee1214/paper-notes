# WorldArena (arXiv:2602.08971, 학회 표기 없음) 분석 노트

*WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models* (arXiv v2, 2026-02-11)을 표 재계산, 통계 검정, 작은 모의 실험으로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2602.08971에서 받을 수 있습니다). 논문 그림은 두 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **판정: 조건부.** 재계산과 통계, 모의 실험은 이 맥에서 완료했습니다.
- 세계 모델 생성, 정책 학습, 시뮬레이터 롤아웃은 GPU와 로봇 시뮬레이터가 필요해 이 맥에서 불가능합니다.
- 공식 저장소(1.0)는 루트에 LICENSE가 없어, 코드는 넣지 않았고 실행하거나 설치하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 통계, 상관, 주장 대조, 모의 실험 결과 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조, 차이 목록, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_dynamic_gate.py` | 모의 실험: 정적 영상 페널티 규칙 차이 (numpy) |
| `analysis/toy_small_n_corr.py` | 모의 실험: 모델 6개와 14개에서 상관 검정력 (numpy) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 두 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_dynamic_gate.py
python3 analysis/toy_small_n_corr.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF에 저작권 표시가 없습니다.
- 논문 그림 두 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2602.08971"). 사진이나 화면 캡처는 쓰지 않았습니다.
- 공식 코드 저장소 `tsinghua-fib-lab/WorldArena`에는 루트 LICENSE가 없습니다. 하위 폴더에만 개별 라이선스가 있습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 모의 실험, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

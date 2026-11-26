# Intercepting the Future (arXiv:2606.02486, CoRL 2026 포스터) 분석 노트

*Intercepting the Future: Latent-Space Predictive World Model for Dynamic VLA Manipulation* (AHEAD, arXiv v2, 2026-10-03)를 표 재계산, 일관성 검사, 작은 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.02486에서 받을 수 있습니다). 논문 그림도 넣지 않았고 우리가 그린 그래프 네 장만 있습니다. 공식 코드와 프로젝트 페이지는 2026-11-26 기준으로 확인하지 못했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이(v1 → v2) |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 표 |
| `analysis/03_analysis.md` | 표 재계산(V1–V11), 토이 |
| `analysis/04_claim_verification.md` | 주장 C1–C30 판정, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산 (numpy, scipy, 수 초) |
| `analysis/toy_kinematic.py` | 토이: 정속 대 등가속 외삽 (numpy, 약 3초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_kinematic.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에서 CC BY 4.0으로 공개되어 있습니다(arXiv 페이지의 라이선스 링크 확인). 이 저장소는 논문 그림을 포함하지 않습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 시뮬레이션 셀의 검정은 부록 C.5의 "5 시드 × 100 롤아웃"을 가정한 값입니다. 가정은 각 문서에 적었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

# LiMA (arXiv:2609.28431) 분석 노트

*LiMA: Bridging Long-term Imagination to Real-time Dexterous Manipulation via Asynchronous Diffusion* (arXiv v1, 2026-09-23, CoRL 2026 포스터)를 표 재계산, 통계 검정, 주장 대조, 작은 수치 토이로 분석한 자료입니다. 공식 코드는 아직 공개되지 않았습니다(프로젝트 페이지 "Code Coming soon"). 논문 원문 PDF는 넣지 않았습니다(arXiv:2609.28431에서 받을 수 있습니다). 논문 그림은 순수 차트 두 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 공식 코드, 가중치, 시연 데이터가 없고 실험은 양팔 UR5와 22-DoF 손 두 대를 전제합니다.
- **재계산, 검정, 토이는 완료했습니다.** `analysis/recompute.py`(수 초)와 `analysis/toy_bridge_posterior.py`(1초 안팎).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 논문이 적지 않은 것 |
| `analysis/02_prior_work.md` | 선행 계보, 코드 표, I²SB 원 논문과의 대조 |
| `analysis/03_analysis.md` | 재계산, 통계 비교, 토이, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 대조, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy) |
| `analysis/toy_bridge_posterior.py` | 브리지 사후분포 점검 토이 (numpy) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 두 장(© 저자, CC BY 4.0) |

## 재실행

필요한 패키지는 numpy, scipy, matplotlib입니다.

```bash
python3 analysis/recompute.py
python3 analysis/toy_bridge_posterior.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 논문 그림 두 장(그림 1의 막대그래프, 그림 5)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2609.28431"). 로봇 사진, 실로봇 관측 프레임, 일러스트는 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

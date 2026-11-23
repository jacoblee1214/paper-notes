# MotionDisco (arXiv:2606.06139, CoRL 2026 포스터) 분석 노트

*MotionDisco: Motion Discovery for Extreme Humanoid Loco-Manipulation* (arXiv v1, 2026-06-04)을 표 1 재계산, 공식 코드 정적 대조, 부모 선택식 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.06139에서 받을 수 있습니다). 논문 그림은 세 장(그림 2, 3, 4)만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **전체 재현은 Apple M4 Pro(CUDA 없음)에서 불가능합니다.** 추종 정책 RL 학습과 실로봇이 필요하고 저장소에 추종 학습·배포 코드가 없습니다.
- **공식 탐색 파이프라인은 조건부입니다.** 솔버는 CPU지만 새 환경, acados/moto 빌드, LLM API 호출이 필요합니다(이 분석에서는 실행하지 않았습니다).
- **표 재계산과 토이는 가능하고 끝냈습니다.** `analysis/recompute.py`(수 초), `analysis/toy_parent_weights.py`(1초 미만).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 세 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 표 1, 간극 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V6, 일관성 V7–V9, 토이, 주장 판정 C1–C11 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D18, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 1 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_parent_weights.py` | 토이: 부모 선택식 (numpy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 세 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_parent_weights.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 고지는 찾지 못했습니다.
- 논문 그림 세 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2606.06139"). 저자가 만든 도식과 시뮬레이션 그래프·렌더만 골랐고 실로봇 사진 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/Atarilab/motiondisco`(Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 노트의 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

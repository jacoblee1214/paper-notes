# MAPO (arXiv:2611.02613, CoRL 2026 포스터) 분석 노트

*MAPO: Mixture of Experts with Alternating Policy Optimization for Universal Humanoid Motion Tracking* (arXiv v1, 2026-11-02)를 표 재계산, 시드 3개 요약 통계 검정, 식 검증, 작은 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.02613에서 받을 수 있습니다). 논문 그림은 세 장(그림 2, 4, 5)만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 규모의 학습(표 1)은 Isaac Lab과 A6000급 GPU가 필요합니다.** 한 번 약 14–41 시간(환경 속도 가정), 논문 전체 약 37회입니다. 공식 코드는 확인하지 못했습니다(2026-11-26).
- **식 검증과 MAPO 대리 손실 구현, 작은 과제의 PPO 비교는 CPU로 가능합니다.** `analysis/toy_algebra.py`(수 초)와 `analysis/toy_moe_ppo.py`(약 75초)로 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 세 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 표 1, 부록, 버전, 라이선스 |
| `analysis/02_prior_work.md` | 선행 계보, GitHub 표, 구성 요소 출처 |
| `analysis/03_analysis.md` | 재계산 V1–V12, 일관성 검사 K1–K10, 토이 |
| `analysis/04_claim_verification.md` | 주장 C1–C22 검증, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_algebra.py` | 토이: 식 3–6과 부록 B 기울기 검증 (torch, 수 초) |
| `analysis/toy_moe_ppo.py` | 토이: 한 단계 밴딧에서 PPO 대 MAPO (torch CPU, 약 75초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 세 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_algebra.py
python3 analysis/toy_moe_ppo.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 세 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2611.02613"). 시뮬레이션 화면과 데이터셋 프레임이 든 그림은 넣지 않았습니다.
- 공식 코드는 확인하지 못했습니다(2026-11-26). 프로젝트 페이지는 https://mapo-corl2026.github.io/MAPO/ 입니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

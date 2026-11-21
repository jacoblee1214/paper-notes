# FR-VLA (arXiv:2610.25981, CoRL 2026 포스터) 분석 노트

*FR-VLA: Implicit Force Reasoning VLA for Contact-Rich Manipulation* (arXiv v1, 2026-10-19)을 표·그림 수치 재계산과 주장 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2610.25981에서 받을 수 있습니다). 논문 그림은 도식 한 장(발췌)만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **원 실험 재현은 이 맥에서 불가입니다.** 공식 코드가 없고, 실로봇(UR5e와 UR3e)과 과제당 70–90회 시연, GPU 학습이 필요합니다.
- **구성 요소 재구현은 가능합니다.** 위상 라벨 규칙, admission 게이트와 주의, 힘 VQ 토크나이저는 PyTorch로 옮길 수 있습니다. 노력은 1–2주로 추정합니다(측정 전).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 발췌 한 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C19, 구현 가능성 판정 |
| `analysis/recompute.py` | 표·그림 수치 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |
| `report/figures/` | 그래프 두 장(우리가 그림)과 논문 그림 발췌 한 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF의 저작권 표시는 찾지 못했습니다.
- 논문 그림 2의 (b)(c)만 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2610.25981"). 로봇 사진이 있는 (a)는 싣지 않았습니다.
- 공식 코드는 찾지 못했습니다. 기준선 저장소(VLA-Adapter, ForceVLA, TA-VLA)는 이 저장소에 넣지 않았고 실행하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트와 분석 문서는 이 저장소 작성자가 만들었습니다.

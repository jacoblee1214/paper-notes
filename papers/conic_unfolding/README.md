# Scalable Deep Unfolding of Conic Optimizers (arXiv:2606.13825, CoRL 2026 스포트라이트) 분석 노트

*Scalable Deep Unfolding of Conic Optimizers* (arXiv v3, 2026-11-21)를 표 재계산, 통계 검정, 두 역전파 규칙의 작은 토이, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.13825에서 받을 수 있습니다). 논문 그림은 세 장(그림 1, 2, 4)만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **두 역전파 규칙과 펼침 학습의 축소판은 CPU에서 구현했고 확인을 끝냈습니다.** `analysis/toy_dk_cg_checks.py`(수 초)와 `analysis/toy_du_maxcut.py`(약 40초)는 논문 식만 보고 새로 짠 torch 코드입니다.
- **논문 수치의 전체 재현은 일반 노트북에서 어렵습니다.** 표 2(GPU 배치)와 그림 2의 n = 50k는 24 GB CUDA GPU와 Julia(COSMO.jl)가 필요합니다.
- **공식 코드의 표 1 계열은 CPU에서 일부 가능합니다.** 설치(clarabel, jaxtyping, Python 3.14 이상)와 설정의 device 변경이 필요하고, 이 분석은 공식 코드를 실행하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 세 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 표 1·2·3·6, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 결과, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_dk_cg_checks.py` | 토이: 식 5–6과 식 9–10 확인 (torch, 수 초) |
| `analysis/toy_du_maxcut.py` | 토이: max-cut 12 노드에서 ρ와 α 일정 학습 (torch, 약 40초) |
| `analysis/plot_figures.py` | 그래프 6장 생성 |
| `report/figures/` | 우리가 그린 그래프 6장과 논문 그림 3장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_dk_cg_checks.py
python3 analysis/toy_du_maxcut.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF의 저작권 충돌 표시는 찾지 못했습니다.
- 논문 그림 세 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2606.13825"). 선 그래프만이고 사진은 없습니다.
- 공식 코드는 `github.com/alexoshin/deep-conic-optimization`(Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

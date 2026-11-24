# W2A (arXiv:2611.02600) 분석 노트

*W2A: Distilling World Priors into Action Models for Real-Time Robotic Control* (arXiv v1, 2026-11-02, CoRL 2026 포스터)을 표 재계산, 통계 검정, 주장 대조, 작은 토이로 분석한 자료입니다. 공식 코드는 아직 공개되지 않았습니다(저자 저장소 README에 "coming soon..."). 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.02600에서 받을 수 있습니다). 논문 그림은 순수 그래프 세 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **전체 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 교사(Cosmos-Predict2) 적응과 증류는 CUDA GPU가 필요하고 코드도 없습니다.
- **재계산, 검정, 토이는 완료했습니다.** `analysis/recompute.py`(수 초)와 `analysis/toy_pca_distill.py`(약 1분).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 세 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 논문이 적지 않은 것 |
| `analysis/02_prior_work.md` | 선행 계보, 코드 표, 표 1 기준선 값의 출처 |
| `analysis/03_analysis.md` | 재계산, 통계 비교, 시드 변동, 토이, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 대조, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy) |
| `analysis/toy_pca_distill.py` | PCA 증류 토이 (torch, numpy) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 세 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_pca_distill.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 논문 그림 세 장(그림 6, 9, 10)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2611.02600"). 로봇·시뮬 프레임, 실로봇 사진, 영상 잠재 시각화는 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

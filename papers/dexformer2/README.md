# DexFormer 2.0 (arXiv:2611.20273) 분석 노트

*DexFormer 2.0: Towards a Cross-Embodiment Foundation Policy for Dexterous Grasping* (arXiv v2, 2026-11-17)을 표 재계산, 통계 검정, 일관성 검사로 분석한 자료입니다. 학회 표기는 찾지 못했습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.20273에서 받을 수 있습니다). 논문 그림은 그림 2의 MorphNode 블록 한 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **재현은 이 맥에서 불가능합니다.** 정책 코드, 학습 코드, 체크포인트, 손·팔 자산 번들이 공개되지 않았습니다. 논문은 Isaac Lab(GPU 전제)을 씁니다.
- **아키텍처 구현은 가능합니다**(추정 1–2주). 보충 자료가 없어 descriptor 세부와 보상 가중치는 가정해야 합니다.
- **재계산은 완료했습니다.** `analysis/recompute.py`(수 초, numpy·scipy)로 표의 평균과 표준편차에서 구간과 검정을 계산합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 한 장과 우리가 그린 그래프 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법(식 1–5), 실험, 버전 차이, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사 K1–K10, 검정력 |
| `analysis/04_claim_verification.md` | 공식 저장소 정적 확인, 주장 검증 C1–C22, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 2 블록(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF에 저작권 표시가 없습니다.
- 논문 그림 2의 MorphNode 블록 도식만 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2611.20273"). 로봇 그림과 시뮬 렌더는 넣지 않았습니다.
- 공식 저장소 `github.com/DavidLXu/dexformer2`는 정책 코드가 없고 영상 렌더 코드와 기록 궤적을 담고 있습니다. 이 노트에서는 코드를 넣지 않았고 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 노트의 작성자가 만들었습니다.

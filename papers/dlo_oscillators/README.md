# Φ-CON (arXiv:2611.23563) 분석 노트

*Learning the Dynamics of Deformable Linear Objects Through Coupled Oscillators for Bimanual Robotic Manipulation* (arXiv v1, 2026-11-17)을 표 재계산, 통계 검정, 주장 대조로 분석한 자료입니다. 이 논문에는 공식 코드가 없어서 코드 대조 대신 주장 대조를 했습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.23563에서 받을 수 있습니다). 논문 그림은 넣지 않았고 우리가 그린 그래프만 씁니다.

## 구현 가능성 (요약)

- **실로봇 검증은 이 맥에서 불가능합니다.** 두 대의 로봇, 힘 센서, 모션캡처가 필요합니다.
- **시뮬 수준 재구현은 조건부로 가능합니다.** 모형이 작아서 CPU나 MPS 학습이 가능할 가능성이 큽니다. 학습 시간은 측정하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 코드 현황 |
| `analysis/03_analysis.md` | 재계산, 통계 검정, 논문 내부 불일치 |
| `analysis/04_claim_verification.md` | 주장 대조, 증명 검토, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 수치 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF 첫 쪽에 저자 저작권 표시가 있고, 충돌하는 고지는 찾지 못했습니다.
- 공식 코드는 공개되어 있지 않습니다. 선행 연구의 코드(NeurIPS 2024 CON)는 이 폴더에 복제하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다. 아무것도 설치하거나 실행하지 않았습니다(재계산 스크립트 제외).

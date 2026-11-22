# TouchTwin (arXiv:2611.17619) 분석 노트

*TouchTwin: Scaling Human Touch Data via Real-to-Sim-to-Real* (arXiv v1, 2026-11-13)을 표 재계산과 검정, 공식 코드 부재 확인으로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.17619에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 그래프는 논문 표의 값을 옮겨 우리가 다시 그렸습니다.

## 구현 가능성 (요약)

- **논문 규모의 재현은 이 맥에서 불가능합니다.** 정책 학습과 생성 영상은 GPU 전제이고, 촉각 기록 데이터(500시간 규모)와 접촉 예측기 구조가 없습니다.
- **방법의 핵심 원리(촉각으로 닫힘 힘을 정해 물리 재생을 만드는 것)는 CPU MuJoCo로 작게 구현할 수 있습니다.** 닫힘 힘을 정하는 식이 논문에 없어서 그 부분은 직접 정해야 합니다.
- **공식 코드는 없습니다.** 프로젝트 페이지에 코드 링크가 없고, 이름이 같은 다른 저장소는 논문과 무관합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 |
| `analysis/02_prior_work.md` | 선행 계보, 저장소 표 |
| `analysis/03_analysis.md` | 재계산 (구간, 검정, 검정력), 일관성 검사 K1–K8 |
| `analysis/04_claim_verification.md` | 주장 판정 C1–C16, 구현 가능성, 라이선스 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF에 저작권 표시는 찾지 못했습니다.
- 논문 그림은 실제 기록 프레임과 생성 영상이 섞여 있어 이 자료에 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트와 분석 문서는 이 자료의 작성자가 만들었습니다.
- 공식 코드와 데이터는 받지 않았습니다. TouchScale 데이터의 공개 여부는 확인하지 못했습니다.

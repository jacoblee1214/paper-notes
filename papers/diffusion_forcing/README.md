# Diffusion Forcing (arXiv:2407.01392, NeurIPS 2024) 분석 노트

*Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion*(NeurIPS 2024, arXiv v4)을 코드 대조, 표 재계산, 신뢰구간으로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2407.01392에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 그래프는 모두 표 값으로 우리가 그렸습니다.

## 구현 가능성 (요약)

**부분 가능.** 구조 재구현은 할 수 있습니다. 논문 표 수치를 다시 얻는 것은 대부분 불가능합니다.

- **미로 계획과 시계열: 조건부.** 논문 기준 2080Ti에서 4–8시간입니다.
- **영상: 불가.** A100 8장 12시간입니다.
- **로봇: 불가.** 실로봇 데이터와 코드가 공개되지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (그래프 두 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 코드 표 |
| `analysis/03_analysis.md` | 재계산, 신뢰구간, 순위, 일관성 검사 |
| `analysis/04_code_verification.md` | 공식 코드 대조와 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/recompute_results.json` | 재계산 결과 |
| `analysis/plot_figures.py` | 그래프 생성 |
| `report/figures/` | 그래프 두 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지 확인). PDF에는 저작권이나 권리 고지가 없습니다.
- 공식 코드는 `github.com/buoyancy99/diffusion-forcing`에 있고 MIT입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

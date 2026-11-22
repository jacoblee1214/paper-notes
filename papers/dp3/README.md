# DP3 (arXiv:2403.03954, RSS 2024) 분석 노트

*3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations* (arXiv v7, 2024-09-27)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2403.03954에서 받을 수 있습니다). 논문 그림은 실로봇 사진을 포함하므로 넣지 않았고, 그림은 모두 우리가 그린 그래프입니다.

## 구현 가능성 (요약)

- **이 맥에서 논문 표 재현은 불가능합니다.** 시뮬 72 과제 학습은 CUDA 전제이고, 실로봇은 장비가 필요합니다.
- **재계산과 정적 코드 대조는 가능하고 끝냈습니다.** 설치와 실행은 하지 않았습니다.
- **인코더와 결정부의 소형 재구현은 가능합니다 (MPS, 노력 3–5일, 측정 전 추정).**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그래프 두 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 버전 차이 (v1 대 v7) |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 C1–C23 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D18, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 생성 |
| `report/figures/` | 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF 본문에서 저작권 표시를 찾지 못했지만, 그림에는 실로봇 사진이 있어 쓰지 않았습니다.
- 공식 코드는 `github.com/YanjieZe/3D-Diffusion-Policy`에 있고 MIT입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

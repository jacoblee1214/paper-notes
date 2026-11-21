# PGND (arXiv:2506.15680, RSS 2025) 분석 노트

*Particle-Grid Neural Dynamics for Learning Deformable Object Models from RGB-D Videos* (arXiv v2, 2025-11-06)를 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2506.15680에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- **학습 재현은 이 맥에서 불가입니다.** 공식 코드는 CUDA를 전제하고 논문은 단일 4090에서 약 8시간이라고 씁니다.
- **공식 체크포인트 추론은 조건부입니다.** 체크포인트는 작지만 Warp CPU 설치와 데이터 다운로드가 필요합니다.
- **영상 지표 재현은 불가입니다.** 3DGS 렌더링에 CUDA 래스터라이저가 필요합니다.
- **직접 짠 소형 구현은 가능합니다.** 1–2주 추정이며 측정하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조, 라이선스, 구현 가능성 |
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

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다. 논문 그림 2에는 실로봇과 실제 물체 사진이 들어 있고, 다른 그림의 권리 상태를 확인하지 못해 논문 그림은 싣지 않았습니다.
- 공식 코드는 `github.com/kywind/pgnd`에 있습니다(MIT). LICENSE의 저작권 줄은 GitHub 템플릿 문구입니다. 이 저장소에는 코드를 넣지 않았고 실행하거나 설치하지 않았습니다.
- 제3자 렌더러(`diff-gaussian-rasterization-w-depth`)는 Gaussian-Splatting License(비상업)입니다.
- 가중치와 데이터셋은 받지 않았습니다. 매니페스트 JSON 두 개만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

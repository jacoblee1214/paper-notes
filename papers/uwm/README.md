# UWM (arXiv:2504.02792, RSS 2025) 분석 노트

*Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets* (arXiv v3, 2025-05-23)를 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2504.02792에서 받을 수 있습니다). 논문 그림은 한 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **이 맥(Apple M4 Pro, CUDA 없음)에서 DROID 사전학습과 실로봇 재현은 불가능합니다.** 원 설정은 A100 4장 24시간입니다.
- **구조 재구현과 작은 모델의 샘플러 검사는 가능합니다.**
- **LIBERO 사전학습 체크포인트의 추론 평가는 조건부로 가능합니다.** 설치, MPS 포팅, 체크포인트 다운로드가 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 한 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 표 재계산과 검정 (scipy, statsmodels, 수 초) |
| `analysis/plot_figures.py` | 자체 그래프 두 장 생성 (matplotlib) |
| `report/figures/` | 자체 그래프 두 장과 논문 그림 한 장(그림 6, © 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib statsmodels
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지의 license 링크 확인). 원문 PDF에서 저작권 표시는 찾지 못했습니다.
- 논문 그림 한 장(그림 6)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2504.02792 (v3)"). 로봇 사진, 장면 프레임, 시뮬 화면이 있는 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/WEIRDLabUW/unified-world-model`에 있습니다. LICENSE 파일과 메타데이터가 없습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 재계산 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

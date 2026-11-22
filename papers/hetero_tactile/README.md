# HTT (arXiv:2606.29948, CoRL 2026 포스터) 분석 노트

*Heterogeneous Tactile Transformer* (arXiv v2, 2026-10-13)를 표 재계산, 통계 검정, 공식 코드 정적 대조, CPU 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.29948에서 받을 수 있습니다). 논문 그림은 두 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **전체 재현은 맥 한 대(Apple M4 Pro, MPS, CUDA 없음)에서는 어렵습니다.** 사전학습 코드는 CUDA가 없으면 오류를 내고, 실로봇과 ManiFeel, 검색 코드는 저장소에 없습니다.
- **공개 체크포인트(69 MB) 추론과 소규모 하류 평가는 다운로드와 설치가 허용되면 가능해 보입니다.** 속도는 측정하지 않았습니다.
- **재계산과 토이는 끝냈습니다.** `analysis/toy_checks.py`는 수 초 안에 돕니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 학회 상태, 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, statsmodels, 수 초) |
| `analysis/toy_checks.py` | 토이 두 개 (torch는 첫 토이에만 필요) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 두 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy statsmodels matplotlib torch
python3 analysis/recompute.py
python3 analysis/plot_figures.py
python3 analysis/toy_checks.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 두 장(그림 3, 그림 2의 B 패널)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2606.29948"). 실로봇·센서 사진과 촉각 프레임이 든 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/jxbi1010/HTT`(CC BY-NC 4.0)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 데이터셋과 체크포인트는 Hugging Face(CC BY-NC 4.0)에 있으며 받지 않았습니다. 데이터셋 통계는 데이터셋 카드의 값을 옮겼습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

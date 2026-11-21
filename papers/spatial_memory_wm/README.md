# Video World Models with Long-term Spatial Memory (arXiv:2506.05284, NeurIPS 2025) 분석 노트

*Video World Models with Long-term Spatial Memory* (arXiv v1, 2025-06-05, NeurIPS 2025 본 학회 트랙)을 표 재계산과 통계 검정으로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2506.05284에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- **학습은 불가능합니다.** CUDA 전제이고 A100 8장 규모입니다.
- **추론은 조건부입니다.** 코드가 CUDA에 묶여 있어서 장치 변경과 메모리 측정이 먼저 필요합니다.
- **논문 표 재현은 불가능합니다.** 평가 코드와 기준선 코드가 저장소에 없습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 본문에 없는 것 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 (사용자 연구 검정, 뷰 리콜, VBench, 절제), 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 권리, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (표준 라이브러리, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (numpy, matplotlib) |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스입니다 (CC BY가 아닙니다). 논문 그림과 원문은 이 저장소에 넣지 않았습니다.
- 공식 코드는 `github.com/spmem/spmem`에 있습니다 (Apache-2.0, LICENSE.txt). 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

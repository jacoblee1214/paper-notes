# PLaW-VLA (arXiv:2610.12285, CoRL 2026 포스터) 분석 노트

*PLaW-VLA: Predictive Latent World Modeling for Vision-Language-Action Policies* (arXiv v1, 2026-10-08)를 표 재계산, 통계 검정, 공식 코드의 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(논문은 arXiv:2610.12285에서 받을 수 있고, arXiv 라이선스가 CC BY가 아닙니다). 그림은 모두 우리가 그린 그래프입니다.

## 구현 가능성 (요약)

- **공식 코드와 논문 수치의 재현은 이 맥(Apple M4 Pro, CUDA 없음)에서 불가능합니다.** 코드가 CUDA를 전제하고 사전학습 체크포인트가 공개되지 않았습니다.
- **재계산, 정적 코드 대조, 어텐션 마스크와 비용 토이는 가능합니다.** 모두 수 초입니다.
- **소형 구조 재구현은 가능합니다.** 노력은 2–3주(근거 없는 추정)입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V9, 불일치 K1–K10, 토이, 주장 판정 C1–C14 |
| `analysis/04_code_verification.md` | 공식 코드 대조 (일치 M1–M9, 차이 D1–D17), 구현 가능성 |
| `analysis/recompute.py`, `paper_tables.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_attention_mask.py` | 블록 인과 어텐션 마스크 확인 (numpy) |
| `analysis/cost_accounting.py` | 파라미터·토큰 수 계산 (표준 라이브러리) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_attention_mask.py
python3 analysis/cost_accounting.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 표시는 찾지 못했습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 코드는 `github.com/RainyRobo/PLaW-VLA`(Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았고 실행하거나 설치하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다.

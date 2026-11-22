# X-WBC (arXiv:2609.15213, CoRL 2026 포스터) 분석 노트

*X-WBC: A Cross-Embodiment Foundation Model for Humanoid Whole-Body Control* (arXiv v2, 2026-10-25)을 표 재계산, 통계 검정, 작은 토이로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2609.15213에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **논문 수치 재현은 개인 환경에서 어렵습니다.** 공식 저장소에 코드와 체크포인트가 없고, 학습은 H100 8장으로 약 2일입니다.
- **구조 구현과 메커니즘 토이는 가능합니다.** `analysis/toy_alignment.py`(약 1분, torch CPU)로 정렬 손실과 검색 지표를 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식·관련 저장소 표 |
| `analysis/03_analysis.md` | 재계산과 일관성 검사 K1–K23, 토이 결과 |
| `analysis/04_claim_verification.md` | 주장 C1–C16, 구현 가능성 판정 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_alignment.py` | 토이: 정렬 손실과 검색 Recall@1 (torch, 약 1분) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_alignment.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문의 arXiv 라이선스는 arXiv.org 비독점 배포 라이선스입니다(CC BY가 아님). 그래서 논문 그림은 넣지 않고 우리가 그린 그래프만 넣었습니다.
- 공식 저장소(`github.com/LogosIntelligenceLab/x-wbc`)에는 프로젝트 페이지만 있고 라이선스 파일이 없습니다. 코드를 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

# Assembling Two Parts in One Hand (arXiv:2609.10137, CoRL 2026 포스터) 분석 노트

*Assembling Two Parts in One Hand*를 표·그림 수치 재계산, 통계 검정, 주장 검증으로 읽은 자료입니다. 공식 저장소(`LTBGbird/in-hand-assembly`)에는 구현이 없습니다(README와 TODO, 2026-11-24 확인). 논문 원문 PDF는 넣지 않았습니다(arXiv:2609.10137에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **이 맥(Apple M4 Pro, MPS, CUDA 없음)에서 논문의 학습과 실로봇 결과는 구현할 수 없습니다.** Isaac Sim과 NVIDIA GPU, Sharpa Wave 손과 Franka 팔이 필요하고 공식 코드가 없습니다.
- **보상·관측·무작위화 명세의 PyTorch 구현은 가능합니다**(수 시간-2일, 추정). 소형 MuJoCo 절제는 조건부입니다(2-6주 이상, 추정, 논문 수치와 비교 불가).
- **재계산은 `analysis/recompute.py`로 수 초에 다시 할 수 있습니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계, 버전 이력 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, GitHub 표 |
| `analysis/03_analysis.md` | 수치 재계산 V1-V12, 일치 확인 K1-K7 |
| `analysis/04_claim_verification.md` | 주장 C1-C14, 구현 가능성 |
| `analysis/recompute.py`, `analysis/recompute_results.json` | 재계산, 검정, 보상 식 계산 |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에서 non-exclusive-distrib 라이선스(arxiv.org/licenses/nonexclusive-distrib/1.0)로 공개되어 있습니다. 그래서 논문 그림은 넣지 않았고 우리가 그린 그래프만 넣었습니다.
- 프로젝트 페이지는 ltbgbird.github.io/in-hand-assembly-page/ 에 있습니다. 공식 저장소에는 구현이 없습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 재계산 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.

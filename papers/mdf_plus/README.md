# MDF++ (arXiv:2611.09293, CoRL 2026 포스터) 분석 노트

*MDF++: Visuo-Tactile World Action Model with End-to-end Representation Learning* (arXiv v1, 2026-11-06)을 표 재계산, 검정, 주장별 검증으로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.09293에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **원 실험 재현은 이 맥(CUDA 없음)에서 불가입니다.** 시뮬은 Isaac Lab과 NVIDIA GPU가, 실로봇은 Franka와 GelSight, UMI 방식 수집 장치가 필요합니다.
- **축소판 구현은 조건부로 가능합니다.** 선택적 그래디언트 규칙은 식 한 줄로 구현할 수 있고, 합성 데이터에서 naive 대 선택적 비교를 할 수 있습니다.
- **공식 코드는 찾지 못했습니다.** 프로젝트 페이지와 GitHub 검색 결과를 확인했습니다(`analysis/02_prior_work.md` §2.1).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 정보 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 코드 조사 결과 |
| `analysis/03_analysis.md` | 재계산, 검정, 실로봇 표 9 검산, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장별 판정(C1–C26), 권리, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 표 값 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문 라이선스는 arXiv 비독점 배포 라이선스 1.0입니다(CC BY가 아님). 그래서 논문 그림은 넣지 않았습니다. 표 값은 인용과 재계산을 위해 옮겼고 출처를 적었습니다.
- 그래프, 재계산 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다. 공식 코드는 찾지 못했으므로 실행하거나 설치하지 않았습니다.

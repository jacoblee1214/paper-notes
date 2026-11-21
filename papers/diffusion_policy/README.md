# Diffusion Policy (arXiv:2303.04137) 분석 노트

*Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (IJRR 2024, arXiv v5 2024-03-14)의 표 수치를 다시 계산하고 본문과 맞춰 본 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다 (arXiv 비독점 배포 라이선스). 재현 실험은 하지 않았습니다.

## 구현 가능성 (요약)

- **논문 표 재현은 이 맥에서 불가입니다.** 시뮬 전체는 GPU와 수천 에폭을 전제하고, 실로봇은 하드웨어가 필요합니다.
- **구조 재구현과 2차원 토이는 가능합니다.** 확산 정책의 행동 시퀀스와 추론 단계 수를 CPU에서 검사할 수 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전과 라이선스 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 (평균 향상률, 실로봇 Wilson과 Fisher), 일관성 검사 |
| `analysis/04_code_verification.md` | 공식 코드 대조와 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (scipy, 수 초) |
| `analysis/recompute_results.json` | 재계산 결과 |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 비독점 배포 라이선스(`http://arxiv.org/licenses/nonexclusive-distrib/1.0/`)로 공개되어 있습니다. CC BY가 아니므로 논문 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/real-stanford/diffusion_policy`에 있고 MIT 라이선스입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치, 데이터셋, 실험 로그는 받지 않았습니다.

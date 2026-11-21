# EgoEngine (arXiv:2606.12604, CoRL 2026 포스터) 분석 노트

*EgoEngine: From Egocentric Human Videos to High-Fidelity Dexterous Robot Demonstrations* (arXiv v1, 2026-06-10)을 표 수치 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.12604에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다. 논문 라이선스가 arXiv 비독점 1.0이라 CC BY가 아닙니다.

## 구현 가능성 (요약)

- **전체 결과는 이 맥에서 재현할 수 없습니다.** 행동 분기는 CUDA를 요구하고, 디지털 트윈과 정책 학습 코드는 공식 릴리스에 없습니다.
- **시각 분기 렌더링과 합성은 조건부로 가능합니다** (macOS 지원 시각 환경).
- **통계 재계산은 완료했습니다.** 시도 수는 논문에 없어서 표 값에서 추정했습니다 (`analysis/03_analysis.md` §0).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 수치 재계산과 검정 (scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv:2606.12604, arXiv 비독점 배포 라이선스 1.0 (CC BY 아님). 논문 그림은 넣지 않았습니다.
- 공식 코드: `github.com/GaTech-RL2/EgoEngine-CoRL2026` (커밋 `b1363c2`). 혼합 라이선스이며 `spider/`와 `action/`은 CC BY-NC 4.0입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다. 정적 읽기만 했습니다.
- 표 수치와 본문 인용은 분석을 위해 옮겼습니다. 그래프와 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.

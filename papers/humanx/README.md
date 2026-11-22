# HumanX (arXiv:2602.02473, CoRL 2026 포스터) 분석 노트

*HumanX: Toward Agile and Generalizable Humanoid Interaction Skills from Human Videos* (arXiv v2, 2026-10-09)를 표 재계산, 구간 추정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2602.02473에서 받을 수 있습니다). 그래프는 모두 이 저장소 작성자가 그렸습니다.

## 구현 가능성 (요약)

- **실제 로봇 재현은 불가입니다.** Unitree G1과 광학 모캡 14대가 필요하고 배포 코드가 공개돼 있지 않습니다.
- **공식 학습·평가는 NVIDIA GPU와 Isaac Gym Preview 4(Linux)를 전제합니다.** Apple Silicon에서는 불가입니다.
- **표 재계산, 정수 분모 검사, XGen 예제 점검, 작은 환경에서의 부품 재구현은 가능합니다.** 앞의 세 가지는 이 저장소에 스크립트가 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, v1 ↔ v2 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, 공식 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산, 일관성 검사, XGen 예제 점검, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 구현 가능성, 라이선스 |
| `analysis/recompute.py` | 표 1–3 재계산과 검정 (표준 라이브러리, 1초 미만) |
| `analysis/feasibility.py` | 표 1 퍼센트의 정수 분모 가능성 (표준 라이브러리) |
| `analysis/xgen_example_check.py` | 공개 XGen 예제 npz 점검 (numpy) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (numpy, matplotlib) |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/recompute.py
python3 analysis/feasibility.py
python3 analysis/plot_figures.py
# 공식 저장소를 직접 받았다면 (읽기만 하는 스크립트):
python3 analysis/xgen_example_check.py <HumanX 저장소>/xgen/data/example
```

## 출처와 라이선스

- 논문: arXiv 비독점 배포 라이선스(CC BY 아님). 그래서 논문 그림을 넣지 않았고, 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다.
- 공식 코드는 `github.com/wyhuai/HumanX`에 있습니다(Apache-2.0 + Commons Clause, 판매·유료 서비스 제공 금지). 이 저장소에는 코드를 넣지 않았고, 정적으로만 읽었으며 실행하거나 설치하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다(공개된 정책 체크포인트도 없습니다).
- 그래프, 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다.

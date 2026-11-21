# UMI (arXiv:2402.10329, RSS 2024) 분석 노트

*Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots* (arXiv v3, 2024-03-06)을 보고 수치 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2402.10329에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 구현 가능성 (요약)

- **실로봇 실험, 현장 시연, 학습 재현은 이 맥에서 불가입니다.** 하드웨어, 사람, GPU가 필요합니다.
- **보고 수치 재계산과 공식 코드 정적 대조는 가능하고 끝냈습니다.** 코드는 실행하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, 공식 저장소 표 |
| `analysis/03_analysis.md` | 성공률 재계산, Fisher 검정, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 보고 수치 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 재계산 그래프 두 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비배타 배포 라이선스(non-exclusive distribution 1.0)입니다. CC BY 4.0이 아니므로 논문 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/real-stanford/universal_manipulation_interface`에 있습니다(MIT). 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 그래프와 분석 문서, 스크립트는 이 저장소 작성자가 만들었습니다. 논문 수치는 인용과 재계산을 위해 옮겼습니다.

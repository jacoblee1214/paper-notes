# Fast-JEPA (arXiv:2611.25927) 분석 노트

*Fast-JEPA: Advancing the Success–Latency Frontier for Latent World Models* (arXiv v1, 2026-11-19)을 표 재계산, 통계 검정, 일관성 검사로 분석한 자료입니다. 학회 표기는 없습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.25927에서 받을 수 있습니다). 논문 그림도 넣지 않았습니다.

## 판정 요약

- **지연 우위는 계산과 맞습니다.** 10.88 ms 대 INTACT 26.19 ms (2.41배), 채점 비용 4.8배 감소.
- **성공률 우위는 3 시드에서 구분되지 않습니다.** INTACT Guarded A와 0.17 pp, Welch p = 0.72.
- **DynaGoal Shift와 Deadline 우위는 시드 기준으로 확인됩니다.**
- **표와 그림이 맞지 않는 곳이 있습니다.** 0 ms 지연 결과와 Shift 표, 같은 컨트롤러의 두 표 값.
- **공식 구현은 없습니다.** 저장소에는 README와 그림만 있습니다.
- **구현 가능성:** 알고리즘은 구현할 수 있지만, 논문 수치는 코드, 체크포인트, 벤치마크 케이스가 없어 재현할 수 없습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표, 학회 표기 확인 |
| `analysis/03_analysis.md` | 재계산, 통계 검정, 표 사이 일관성 |
| `analysis/04_claim_verification.md` | 주장 검증, 권리, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install scipy matplotlib markdown-it-py
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스입니다(CC BY 아님). 그래서 논문 그림은 넣지 않았습니다. 텍스트 인용은 arXiv:2611.25927의 절과 표 번호를 붙였습니다.
- 공식 저장소 `DeepCybo-PhysAI/Fast-JEPA`에는 라이선스가 없고 구현 코드도 없습니다. 저장소 내용은 넣지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

# UniVTAC (arXiv:2602.10093, CoRL 2026 포스터) 분석 노트

*UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking* (arXiv v1, 2026-02-10)을 표 재계산과 검정, 공식 코드 정적 대조로 분석한 자료입니다. 학회는 공식 채택 목록에서 확인했습니다 (CoRL 2026 포스터).

## 구현 가능성 (요약)

- **시뮬레이션 벤치마크 재현은 이 맥에서 불가입니다.** Isaac Sim과 RTX GPU가 필요합니다.
- **인코더 사전학습은 조건부입니다.** 공식 코드가 CUDA로 고정되어 있고 데이터 받기가 먼저 필요합니다.
- **실로봇 재현은 불가입니다.**
- **표 재계산과 검정은 가능합니다** (`analysis/recompute.py`, 수 초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 자료 표 (코드·데이터·라이선스) |
| `analysis/03_analysis.md` | 재계산, 과제별 검정, 일관성 검사 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조 (D1–D23), 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `analysis/recompute_results.json` | 재계산 결과 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 1.0 라이선스입니다 (CC가 아님). 원문 PDF는 이 저장소에 넣지 않았습니다. arXiv:2602.10093에서 받을 수 있습니다.
- 논문 그림은 넣지 않았습니다. 그래프는 논문 표의 보고값을 옮겨 우리가 다시 그렸습니다.
- 공식 코드는 github.com/univtac/UniVTAC에 있습니다. LICENSE 파일은 Apache-2.0이고, README는 MIT라고 적습니다. 이 저장소에는 코드를 넣지 않았고, 설치하거나 실행하지 않았습니다.
- 표 값은 인용과 재계산을 위해 옮겼습니다. 분석 문서, 스크립트, 그래프는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

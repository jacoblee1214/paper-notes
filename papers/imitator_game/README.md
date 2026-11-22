# The Imitator Game (arXiv:2608.22301, CoRL 2026 포스터) 분석 노트

*The Imitator Game: Benchmarking Robot Imitative Ability Beyond Action Prediction* (arXiv v1, 2026-08-23)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2608.22301에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **벤치마크 시뮬 환경과 학습은 Apple 실리콘 맥에서 돌릴 수 없습니다.** 공식 코드가 SAPIEN(Linux/Windows 전용 의존성)과 CUDA 스택을 전제하고, 데이터는 약 880 GB입니다. 실제 로봇과 사람 평가자도 필요합니다.
- **정적 재계산과 통계, 공개 기록 재집계는 CPU로 수 초에 됩니다.** `analysis/recompute.py`.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 판본 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V12, 통계, 주장 판정 C1–C14 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D18, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 표 재계산과 검정 (numpy, scipy) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
mkdir -p data
curl -sL https://imitator-game.github.io/assets/js/figdata.js -o data/site_figdata.js
curl -sL https://imitator-game.github.io/assets/js/full-task-suite-data.js -o data/site_full_task_suite.js
curl -sL https://imitator-game.github.io/assets/leaderboard/all_master_records.csv -o data/site_all_master_records.csv
# data/site_sim_detail.json 은 https://imitator-game.github.io/assets/js/sim-detail-data.js 의
# window.SIM_DETAIL 객체 부분을 JSON으로 저장한 것입니다(텍스트로만 읽고 실행하지 않습니다).
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

`data/`가 없으면 `recompute.py`는 논문 표 수치만으로 가능한 검사(V1–V7, V10)를 돌리고 나머지를 건너뜁니다.

## 출처와 라이선스

- 논문 라이선스는 arXiv 비독점 배포(non-exclusive distribution, CC 아님)입니다. 그림을 넣지 않았고 우리가 그린 그래프만 씁니다.
- 공식 코드는 `github.com/imitator-game/The-Imitator-Game`(MIT)에 있습니다. 이 폴더에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 프로젝트 페이지(`imitator-game.github.io`)의 공개 정적 파일은 텍스트로만 읽었고 이 폴더에 복사하지 않았습니다.
- 데이터셋, 자산, 체크포인트는 받지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

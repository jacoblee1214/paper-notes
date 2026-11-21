# Auditing a Learned Simulator for Legged Robots (arXiv:2611.23118) 분석 노트

*Auditing a Learned Simulator for Legged Robots: The Robotic World Model on ANYmal D* (arXiv v1, 2026-11-17)을 저장소의 결과 파일과 대조해 재계산하고, 공개 코드를 정적으로 읽은 자료입니다. 학습과 정책 실행은 하지 않았습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.23118에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **정적 검사와 재계산은 완료했습니다.** 재계산 39개 중 35개가 논문과 맞았고, 논문 안의 불일치 네 곳을 찾았습니다.
- **CPU로 가능한 것 (설치 후):** 공개 체크포인트 추론, 단일 네트워크 학습(약 1시간), MuJoCo 정책 학습(약 2시간).
- **불가능한 것:** Isaac Lab 원 데이터 생성(GPU), 액추에이터 네트워크, 하드웨어.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 그래프는 우리가 그린 것만) |
| `analysis/01_paper_analysis.md` | 논문 구조, 주장, 방법, 결과, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 학회 상태, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 불일치, 저장소 대조, 데이터 구조 검사 |
| `analysis/04_code_verification.md` | 논문과 공개 코드 대조 항목, 구현 가능성 |
| `analysis/recompute.py` | 재계산과 검사 (표준 라이브러리, 수 초) |
| `analysis/data_checks.py` | 공개 데이터 구조 정적 검사. 원 CSV는 복제하지 않으므로 upstream 저장소에서 받아 `repo/upstream_rwm_lite/assets/data/`에 두거나 인자로 경로를 줍니다 |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `analysis/inputs/` | 저자 공개 결과 파일 네 개 (Apache-2.0 저장소의 `results/`) |
| `report/figures/` | 그래프 세 장 (우리가 그림) |

## 재실행

```bash
pip install matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF에 저작권 표시가 없습니다.
- 논문 그림은 넣지 않았습니다. 그래프는 논문 표와 저장소 결과 파일의 값을 옮겨 작성자가 그렸습니다.
- 공식 코드: `github.com/joyjeet-singh/rwm`, `github.com/joyjeet-singh/rwm-robot-level` (Apache-2.0). 원 RWM 코드 `leggedrobotics/robotic_world_model_lite`(Apache-2.0)와 `rsl_rl_rwm`(BSD 계열)은 읽기만 했고 재배포하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다.

# HumanoidToolBench (arXiv:2610.02089, 학회 표기 없음) 분석 노트

*HumanoidToolBench: Benchmarking Humanoid Tool Use from Selection to Mobile Execution* (arXiv v1, 2026-10-01)를 표 재계산과 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2610.02089에서 받을 수 있습니다). 논문은 arXiv 비독점 배포 라이선스(CC 아님)라서 논문 그림도 넣지 않았고, 그래프는 모두 이 분석에서 새로 그렸습니다.

## 구현 가능성 (요약)

- **평가와 학습은 GPU 없는 기기에서 재현할 수 없습니다.** 공식 평가기는 Linux x86_64 + NVIDIA GPU(EGL)를 전제하고, 강한 정책 5개의 체크포인트·학습 코드가 저장소에 없습니다.
- **공식 저장소는 평가 도구입니다.** ACT와 DP 체크포인트만 공개되어 있습니다.
- **표와 그림의 수치 재계산과 통계 검정은 CPU로 수 초에 끝납니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전, 학회 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소와 관련 저장소 표 |
| `analysis/03_analysis.md` | 재계산 K1–K16, 확인 필요 V1–V7, 주장 판정 C1–C10 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D20, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (표준 라이브러리, 수 초) |
| `analysis/page_table.json`, `analysis/page_events.json` | 프로젝트 페이지 표를 옮긴 입력 값 |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/figures/` | 그래프 세 장 (우리가 그림) |

## 재실행

```bash
pip install matplotlib
python3 analysis/recompute.py          # 표 재계산, recompute_results.json 생성
python3 analysis/plot_figures.py       # 그래프 세 장
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스로 공개되어 있습니다(arXiv abs 페이지의 license 링크 확인). 원문 PDF에서 저작권 고지는 찾지 못했습니다.
- 공식 코드는 `github.com/SNU-PI/HumanoidToolBench`(MIT)에 있습니다. 이 폴더에는 코드를 넣지 않았고 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서, 스크립트는 이 분석의 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

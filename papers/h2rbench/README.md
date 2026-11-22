# H2RBench (arXiv:2609.24778, CoRL 2026 포스터) 분석 노트

*H2RBench: A Real-to-Sim Benchmark for Evaluating Human-to-Robot Transfer* (arXiv v3, 2026-10-25)를 표 재계산, 그림 점 좌표 읽기, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2609.24778에서 받을 수 있습니다). 논문은 arXiv 비독점 배포 라이선스(CC 아님)라서 논문 그림도 넣지 않았고, 그래프는 모두 이 분석에서 새로 그렸습니다.

## 구현 가능성 (요약)

- **벤치마크의 학습과 평가는 GPU 없는 기기에서 재현할 수 없습니다.** Isaac Sim과 Isaac Lab 2.3(NVIDIA RTX, CUDA 13), 가우시안 스플랫 CUDA 확장, 네 방법의 별도 환경, 실세계 Franka + GELLO가 필요합니다.
- **공식 저장소는 평가 하네스입니다.** 학습 코드, 체크포인트, 평가 결과, 쌓기와 끼우기와 장난감 장면의 자산이 저장소에 없습니다.
- **표와 그림의 수치 재계산, 통계 검정, 잡음 시뮬레이션은 CPU로 수 초에 끝납니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소와 관련 저장소 표 |
| `analysis/03_analysis.md` | 재계산 K1–K20, 논문 내부 불일치 V1–V10, 주장 판정 C1–C16 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D16, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정, 잡음 시뮬레이션 (numpy, scipy, 수 초) |
| `analysis/digitize_figures.py` | 논문 그림의 점 좌표 읽기 (poppler, numpy, Pillow, scipy 필요, 원문 PDF를 직접 받아 경로를 넘김) |
| `analysis/digitized_points.json` | 읽은 점 좌표 (벡터 그림 오차 약 0.1, 래스터 그림 약 0.5 퍼센트포인트) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py          # digitized_points.json을 읽음
python3 analysis/plot_figures.py
# 점 좌표를 다시 읽으려면 (poppler와 Pillow 필요, 원문 PDF 경로를 인자로)
python3 analysis/digitize_figures.py path/to/paper.pdf
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스로 공개되어 있습니다(arXiv abs 페이지의 license 링크 확인). 원문 PDF에서 저작권 고지는 찾지 못했습니다.
- 공식 코드는 `github.com/xiaochy/H2RBench`(MIT, PolaRiS 저자의 저작권 표기)에 있습니다. 이 폴더에는 코드를 넣지 않았고 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼고, 그림에서 읽은 점 좌표는 사실 값입니다. 그래프, 분석 문서, 스크립트는 이 분석의 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

# Controller-Shaped Grasping (arXiv:2609.25887, CoRL 2026 포스터) 분석 노트

*What is the Better Curriculum: Controller-Shaped Grasping Behavior for Contact Force-Sensitive Manipulation* (arXiv v1, 2026-09-22)의 표와 본문 수치를 재계산하고, 수집에 쓰인 교사 제어기의 공개 저장소를 논문 부록과 정적으로 대조한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2609.25887에서 받을 수 있습니다). 이 논문의 학습·평가 코드는 공개되어 있지 않습니다.

## 구현 가능성 (요약)

- **논문 수치의 실물 재현은 공개 자료만으로 할 수 없습니다.** 로봇 팔, 그리퍼, 촉각 센서, 학습용 GPU가 필요하고 교사 제어기 저장소는 리눅스 전용이며 라이선스 파일이 없습니다.
- **중재기, 동역학 손실, 교사 제어 논리의 구현과 시뮬레이션 축소판은 일반 노트북급 CPU로 가능해 보입니다.** 이번 노트에서는 실행하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 저자 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 코드와 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V10, 일관성 K1–K18, 계산 실험 T1 |
| `analysis/04_code_verification.md` | 교사 제어기 저장소 대조 D1–D15, 주장 점검 C1–C12, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_delay_window.py` | 계산 실험 T1: 지연 × 속도 × 강성 경계 (numpy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_delay_window.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv non-exclusive 라이선스(arxiv.org/licenses/nonexclusive-distrib/1.0)이고 CC BY가 아니므로 논문 그림은 싣지 않았습니다. 이 저장소의 그래프는 작성자가 다시 그린 것입니다.
- 프로젝트 페이지는 shayfeng.github.io/better-curriculum 입니다. 교사 제어기 저장소(github.com/shayfeng/tactile-reflex-open-source)는 읽기만 했고 코드를 옮기지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 가중치와 데이터셋은 받지 않았습니다.

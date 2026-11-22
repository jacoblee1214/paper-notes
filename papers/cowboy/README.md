# COWBOY (arXiv:2610.28948, CoRL 2026 포스터) 분석 노트

*COWBOY: A Scalable Sim-to-Real Framework for Learning Contextual Whole-Body Manipulation* (arXiv v1, 2026-10-20)을 표 재계산, 일관성 검사, 평면 토이로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 비독점 배포 라이선스, arXiv:2610.28948에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **실로봇과 공식 학습은 이 맥(Apple M4 Pro, CUDA 없음)에서 불가입니다.** 공식 코드도 공개되지 않았습니다.
- **구조 검증의 축소판은 가능합니다.** `analysis/toy_gate_reaching.py`(3초, numpy)와 `analysis/student_size_check.py`(수 초, torch)로 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치 |
| `analysis/02_prior_work.md` | 선행 계보, GitHub 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1-V12, 불일치 목록, 토이 |
| `analysis/04_claim_verification.md` | 주장 검증 C1-C16, 공개 fork 정적 대조, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_gate_reaching.py` | 토이: 거리 게이트 WBC (numpy, 3초) |
| `analysis/student_size_check.py` | 학생 크기와 스텝 시간 (torch) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_gate_reaching.py
python3 analysis/student_size_check.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스로 공개되어 있어(abs 페이지 확인) 논문 그림은 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 공식 코드는 없습니다. 저자 개인 계정의 fork는 정적으로만 읽었고 이 저장소에 넣지 않았습니다. 가중치와 데이터는 받지 않았습니다.

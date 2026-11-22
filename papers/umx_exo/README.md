# UME (arXiv:2606.14218, CoRL 2026 포스터) 분석 노트

*Universal Manipulation Exoskeleton: Learning Compliant Whole-body Policies with Real-time Torque Feedback* (arXiv v1, 2026-06-12)를 성공 횟수 재계산, 통계 검정, 공식 코드 정적 대조, numpy 토이로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2606.14218에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **논문의 실물 실험과 성공률은 구현하기 어렵습니다.** UME 외골격, OpenArm 모바일 매니퓰레이터, 시연 데이터가 필요하고 데이터와 체크포인트는 공개되지 않았습니다. 공식 학습 코드는 CUDA를 하드코딩합니다.
- **재계산, 정적 대조, 토이는 CPU에서 됩니다.** `analysis/recompute.py`(수 초 ~ 20초)와 `analysis/toy_retarget_torque.py`(약 30초, numpy)로 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V, 일관성 K, 토이 T1, 주장 판정 C |
| `analysis/04_code_verification.md` | 공식 코드 대조 D, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy) |
| `analysis/toy_retarget_torque.py` | 토이: 되돌림 이득 사슬, 어깨 야코비안 사상 (numpy) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_retarget_torque.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스(arXiv 라이선스 페이지 확인, CC BY 아님)로 공개되어 있어 논문 그림과 원문 PDF를 넣지 않았습니다.
- 공식 코드는 `github.com/inclusionAI/Universal-Manipulation-Exoskeleton`(Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다. 시연 데이터와 체크포인트는 받지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

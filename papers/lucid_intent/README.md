# LUCID (arXiv:2606.11628, CoRL 2026 포스터) 분석 노트

*LUCID: Learning Embodiment-Agnostic Intent Models from Unstructured Human Videos for Scalable Dexterous Robot Skill Acquisition* (arXiv v1, 2026-06-10)을 그림의 시도 수 재계산, 통계 검정, 주장 대조, 폐루프 토이로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2606.11628에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **실로봇과 대규모 시뮬 학습의 재현은 일반 노트북에서 어렵습니다.** UR5e와 LEAP 손, Isaac Lab 대규모 PPO(RTX급 GPU), 영상 추출 모델 네 종이 필요합니다.
- **공식 코드가 확인되지 않아** 코드 대조 대신 주장 17개를 대조했습니다.
- **수치 재계산과 토이는 CPU에서 수 초입니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치 |
| `analysis/02_prior_work.md` | 선행 계보, 관련 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 |
| `analysis/04_claim_verification.md` | 주장 C1–C17 대조, 구현 가능성 |
| `analysis/recompute.py` | 그림 막대 라벨 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_closed_loop.py` | 토이: 폐루프 재질의 대 일회 질의 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_closed_loop.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스로 공개되어 있어(abs 페이지 확인) 논문 그림을 싣지 않았습니다.
- 그림 값은 "그림 판독 값"으로 표시했고, 분수 라벨(그림 12, 13)만 시도 수로 사용했습니다.
- 가중치와 데이터셋은 받지 않았습니다. 외부 저장소는 복제하거나 실행하지 않았습니다.
- 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

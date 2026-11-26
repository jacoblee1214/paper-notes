# WM-Craftnet (arXiv:2609.07002, CoRL 2026 스포트라이트) 분석 노트

*WM-Craftnet: World Synesthesia Model for Generalizable and Robust Dexterous In-Hand Manipulation* (arXiv v1, 2026-09-07)을 표 재계산, 통계 검정, 공식 코드 정적 대조, 4차원 토이로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(PDF에 저작권 고지가 있어 그래프는 직접 그렸습니다. 원문은 arXiv:2609.07002에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **논문의 시뮬 학습·평가와 실로봇 재현은 불가능합니다.** Isaac Gym Preview 4(Linux, NVIDIA GPU)와 22자유도 손, 촉각, 카메라, SDK가 필요합니다.
- **WSM(월드 모델 상태 추정기) 단독 구현은 가능합니다.** `analysis/toy_clean_target.py`(약 1분, torch)로 4차원 토이를 확인했습니다.
- 공식 코드(`sharpa-robotics/WM-Craftnet`, 커밋 a1e825a)는 정적으로 읽기만 했고 실행하지 않았습니다. 체크포인트와 데이터는 받지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프 세 장) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 기록 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 결과, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_clean_target.py` | 토이: 재구성 목표가 깨끗한 깊이일 때와 오염된 깊이일 때 (torch, 약 1분) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_clean_target.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에서 CC BY 4.0 링크로 공개되어 있으나(arXiv 라이선스 페이지 확인), PDF 첫 쪽에 "© 2026 Sharpa. All rights reserved." 표시가 있어 논문 그림은 쓰지 않았습니다.
- 공식 코드는 `github.com/sharpa-robotics/WM-Craftnet`에 있습니다(LICENSE 파일은 Apache-2.0, 번들된 SDK와 자산은 별도 조건). 이 폴더에는 코드를 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 노트 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

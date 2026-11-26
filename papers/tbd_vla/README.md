# TBD-VLA (arXiv:2606.07895, CoRL 2026 포스터) 분석 노트

*TBD-VLA: Temporal Block Diffusion Vision Language Action Model* (arXiv v1, 2026-06-05)을 표 재계산, 통계 검정, 공식 코드 정적 대조, 마스크 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.07895에서 받을 수 있습니다). 논문 그림은 도식과 그래프 다섯 장만 출처와 함께 넣었습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 개인 환경에서 어렵습니다.** 사전학습(약 1,600 A100 시간), LIBERO 미세조정(약 120 GPU 시간), 실로봇이 필요합니다.
- **공개 체크포인트 추론은 조건부입니다.** LeRobot 설치와 약 4.3 GB 체크포인트가 필요하고 Apple GPU(MPS) 동작은 확인하지 않았습니다.
- **구조의 소형 재구현은 가능합니다.** `analysis/toy_block_mask.py`로 이중 배치 마스크의 누수와 순차 추론 동등성을 확인했습니다(수 초, torch CPU).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 결과, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, statsmodels, 수 초) |
| `analysis/toy_block_mask.py` | 토이: 블록 확산 마스크 검사 (torch CPU, 수 초). 공식 저장소를 읽지 않음 |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 다섯 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy statsmodels matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_block_mask.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지의 라이선스 링크 확인). 원문 PDF에서 저작권 표시는 찾지 못했습니다.
- 논문 그림 다섯 장(그림 1, 2, 4, 5의 막대 그래프, 6)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2606.07895"). 사진과 시뮬레이션 화면이 든 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/TBD-VLA/lerobot`(Apache-2.0)에 있고 이 저장소에는 넣지 않았으며 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

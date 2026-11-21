# SONIC (arXiv:2511.07820) 분석 노트

*SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control* (arXiv v4, 2026-08-13)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 읽은 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2511.07820에서 받을 수 있습니다). 논문 그림은 넣지 않았고 그래프는 논문 수치로 우리가 그렸습니다.

## 구현 가능성 (요약)

- **학습(42M, 128 GPU × 7일)과 실로봇은 이 맥에서 불가능합니다.** CUDA와 G1 실물이 필요합니다.
- **공개 체크포인트 추론과 MuJoCo 소규모 평가는 조건부입니다.** 설치와 다운로드 승인이 필요합니다.
- **토큰 구조 토이와 계획기 식 검사는 가능합니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소와 자료 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 수치 재계산과 검정 (scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문 라이선스: arXiv abs 페이지의 "view license"는 CC BY-SA 4.0입니다. 논문 PDF 첫 쪽에는 "© 2026 NVIDIA. All rights reserved."가 있어 공개 고지가 충돌합니다. 그래서 논문 그림은 넣지 않았습니다.
- 공식 코드: `github.com/NVlabs/GR00T-WholeBodyControl`(커밋 1e1ec4f). 소스는 Apache-2.0, 모델 가중치는 NVIDIA Open Model License입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 체크포인트와 데이터(BONES-SEED)는 받지 않았습니다.

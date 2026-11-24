# Coupled Local and Global World Models (arXiv:2602.06219, CoRL 2026 포스터) 분석 노트

*Coupled Local and Global World Models for Efficient First Order RL* (arXiv v3, 2026-10-11)을 표 재계산, 그림 판독, 통계 검정, 분리 기울기 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2602.06219에서 받을 수 있습니다). 공식 코드는 공개되어 있지 않습니다(프로젝트 페이지 "Code (release soon)").

## 구현 가능성 (요약)

- **논문 전체의 재현은 일반 노트북에서 불가능합니다.** 실로봇 세 과제, 1.0–1.4B 전역 모델(24–64 H100/H200), 비공개 놀이 데이터가 필요합니다.
- **분리 기울기와 정책 최적화 루프의 축소 구현은 가능합니다.** 저차원 환경으로 3–5일(추정)입니다. 기울기 정렬 방향성은 `analysis/toy_decoupled_gradient.py`(torch CPU, 약 13초)로 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그래프 두 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 코드 자료 표 |
| `analysis/03_analysis.md` | 재계산 V1–V8, 그림 판독, 토이 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C16, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 1초 미만) |
| `analysis/digitize_fig8.py` | 그림 5, 8 곡선 끝점 판독 (numpy, Pillow) |
| `analysis/toy_decoupled_gradient.py` | 토이: 분리 기울기 대 국소 단독 (torch, 약 13초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `analysis/fig_crops/` | 판독 입력: 논문 그래프의 래스터 (© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib pillow torch
python3 analysis/recompute.py
python3 analysis/digitize_fig8.py
python3 analysis/toy_decoupled_gradient.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림은 그래프 두 장(그림 5, 그림 8)만 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2602.06219"). 사진, 렌더 프레임, 로봇 그림은 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 공개 코드, 가중치, 데이터셋은 받지 않았습니다.

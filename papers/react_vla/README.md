# REACT (arXiv:2610.12007, CoRL 2026 Spotlight) 분석 노트

*REACT: Rolling Denoising and Dual Decoupling for Reactive Robot Control with VLA Models* (arXiv v1, 2026-10-08)를 표 재계산, 통계 검정, 해석적 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2610.12007에서 받을 수 있습니다). 논문 그림은 세 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **실로봇, RoboTwin 평가, 공식 학습의 재현은 이 맥(Apple M4 Pro, CUDA 없음)에서 불가입니다.** 체크포인트와 평가 코드가 공개되어 있지 않고 학습은 NVIDIA GPU를 전제로 합니다.
- **정적 코드 대조, 표 재계산, 롤링 버퍼 토이는 가능하며 완료했습니다.** `analysis/recompute.py`(수 초), `analysis/toy_rolling_modes.py`(약 1초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치 |
| `analysis/02_prior_work.md` | 선행 계보, 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 통계 검정, 토이 결과, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy) |
| `analysis/toy_rolling_modes.py` | 토이: 롤링 버퍼와 모드 전환 (numpy) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장과 논문 그림 세 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_rolling_modes.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 세 장(그림 3, 5, 11)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2610.12007"). 사진과 시뮬 화면이 있는 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/houlong-xiong/REACT`(Apache-2.0)에 있습니다. 이 노트에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 노트 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

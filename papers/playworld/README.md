# PlayWorld (arXiv:2603.09030, CoRL 2026 포스터) 분석 노트

*PlayWorld: Learning Robot World Models from Autonomous Play* (arXiv v3, 2026-04-06)를 표·그림 수치 재계산, 시행 수 잡음 모의, 공식 저장소 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2603.09030에서 받을 수 있습니다). 논문 그림은 차트와 도식 일부만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥에서 불가능합니다.** 놀이 수집은 DROID 로봇, 세계모델 학습은 H200 8장 이틀, 정책 평가와 RL은 실로봇이 필요합니다.
- **공식 저장소에는 놀이 수집, 곡선 학습, LPIPS·SSIM 평가, DSRL에 해당하는 코드가 보이지 않습니다.**
- **표·그림 수치 재계산과 시행 수 잡음 모의는 CPU에서 수 초–10초입니다.**

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 판본 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1-V10, 일관성 검사 K1-K12, 주장 판정 C1-C10 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1-D16, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표·그림 수치 재계산과 검정 (numpy, scipy) |
| `analysis/toy_noise_ceiling.py` | 시행 수 잡음이 만드는 상관 천장과 RMSE 바닥 (numpy, 약 10초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 우리가 그린 그래프 네 장과 논문 그림 일부(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_noise_ceiling.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지의 라이선스 링크 확인). 원문 PDF에서 저작권 고지는 찾지 못했습니다.
- 논문 그림은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2603.09030"). 로봇 사진, 시뮬 프레임, 로고가 든 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/irom-princeton/open-world`에 있습니다(LICENSE 파일 없음). 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 모의, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

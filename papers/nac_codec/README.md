# NAC (arXiv:2606.21372, CoRL 2026 포스터) 분석 노트

*NAC: Neural Action Codec for Vision-Language-Action Models* (arXiv v2, 2026-09-25)를 표 재계산, 공식 코드 정적 대조, 손실 함수 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.21372에서 받을 수 있습니다). 논문 그림은 두 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **정책 성공률 재현(LIBERO-10, RoboMimic)은 개인 맥(CUDA 없음)에서 사실상 어렵습니다.** 정책 학습과 500 롤아웃 평가가 GPU를 전제한 규모입니다.
- **실로봇 결과는 재현할 수 없습니다.** 로봇 기종, 시연 수, 코드, 데이터가 논문과 저장소에 없습니다.
- **구조 재구현과 손실 토이는 CPU로 가능합니다.** `analysis/toy_loss_sensitivity.py`(약 1초, numpy, scipy)로 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, v1 → v2 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 행동 토크나이저 대안, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 결과, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_loss_sensitivity.py` | 토이: 재구성 손실의 부호·밀림 감도 (numpy, scipy, 약 1초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 두 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_loss_sensitivity.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF의 저작권 표시는 찾지 못했습니다.
- 논문 그림 두 장(그림 2, 그림 5)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2606.21372"). 시뮬레이션 화면과 실로봇 사진 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/ahadjawaid/nac`(Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다. 대조 기준 커밋은 `2e2d9db3d6abde0415f77500f41ff292262a1343`입니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

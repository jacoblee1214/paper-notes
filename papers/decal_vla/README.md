# DeCAL (arXiv:2609.09119, CoRL 2026 포스터) 분석 노트

*DeCAL: Towards Physically-Grounded Dexterous Vision-Language-Action Models via Contact-Aware Latent Co-Imagination* (arXiv v1, 2026-09-08)를 표 재계산, 통계 검정, 공식 코드 정적 대조, 합성 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2609.09119에서 받을 수 있습니다). 논문 그림은 두 장(그림 9, 그림 6의 그래프)만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **실로봇 재현과 공식 학습은 불가능합니다.** UR5 2대, SharpaWave 손, 8×H100, 약 2.2 TB 데이터가 필요하고 학습된 체크포인트는 공개되지 않았습니다.
- **표 수치 재계산과 구조의 합성 토이는 가능합니다.** `analysis/recompute.py`(수 초)와 `analysis/toy_gate.py`(torch CPU, 수 분).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V, 일관성 검사, 절제 검정, 토이, 주장 판정 C |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1–D18, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/code_budget.py` | 코드 설정값에서 파라미터 수, 토큰 수, 메모리, FLOPs 어림 (표준 라이브러리) |
| `analysis/toy_gate.py` | 토이: 스칼라 촉각 게이트가 필요한 조건 (torch CPU) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 두 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/code_budget.py
python3 analysis/toy_gate.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지의 라이선스 링크 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 두 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2609.09119"). 로봇 사진이나 카메라 프레임이 있는 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/AureleoPKU/DeCAL`에 있습니다(LICENSE는 Apache-2.0, `pyproject.toml`은 CC BY-NC-SA 4.0으로 표기가 다릅니다). 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

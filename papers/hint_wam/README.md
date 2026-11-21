# HINT-WAM (arXiv:2611.24789) 분석 노트

*HINT-WAM: Hindsight-Informed Training for World Action Models* (arXiv v1, 2026-11-18)을 표 재계산, 통계 검정, 공식 코드 정적 대조, 숨은 정보 토이로 분석한 자료입니다. 학회 표기는 없습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.24789에서 받을 수 있습니다). 논문 그림은 두 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **공식 설정 재현은 이 맥(CUDA 없음)에서 불가능합니다.** 8 × 24 GB GPU에서 약 72 h가 필요하고, 데이터와 시뮬레이터 평가기가 필요합니다.
- **축소 구현은 조건부로 가능합니다.** 설치와 데이터 승인 후 MPS에서 수 M 파라미터 정책을 비교할 수 있습니다.
- **재계산과 토이는 완료했습니다.** `analysis/recompute.py`(수 초)와 `analysis/toy_mode_energy.py`(CPU, 약 10초).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 검사, 비용, 토이, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy) |
| `analysis/toy_mode_energy.py` | 토이: 숨은 정보 아래의 증류 (torch CPU) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 두 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_mode_energy.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 두 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2611.24789"). 시뮬레이터 장면과 실로봇 사진은 넣지 않았습니다.
- 공식 코드는 `github.com/tuandattt/hintwam_code`입니다. LICENSE 파일은 MIT입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다.

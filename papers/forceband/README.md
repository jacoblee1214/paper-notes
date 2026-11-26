# ForceBand (arXiv:2606.26093, CoRL 2026 포스터) 분석 노트

*ForceBand: Learning Forceful Manipulation with sEMG* (arXiv v1, 2026-06-24)를 표 재계산, 통계 검정, 공식 코드 정적 대조, 합성 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.26093에서 받을 수 있습니다). 논문 그림은 도식과 그래프 세 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 전체(밴드 제작, 정책 학습, UR-5 실험)는 재현할 수 없습니다.** 정책, 로봇, 기준선 코드가 공식 저장소에 없고 하드웨어와 로봇이 필요합니다.
- **EMG2Force 체크포인트의 평가와 보정은 설치와 내려받기를 거치면 CPU나 MPS에서 가능할 것으로 보이지만 시도하지 않았습니다.** 속도는 측정 전입니다.
- **수치 재계산과 합성 토이는 끝냈습니다.** `analysis/recompute.py`와 `analysis/toy_calibration.py`(각 수 초)입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V, 일관성 검사 K, 토이, 주장 판정 C |
| `analysis/04_code_verification.md` | 공식 코드 대조 D, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_calibration.py` | 토이: 이득 불일치와 짧은 보정, 직전 값 유지 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 세 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_calibration.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지의 라이선스 링크 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 세 장(그림 2의 위쪽 패널, 그림 5, 그림 9)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2606.26093"). 사진, 사람, 로봇 화면이 들어간 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/Bottle101/ForceBand`(Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다. 체크포인트와 데이터셋은 받지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

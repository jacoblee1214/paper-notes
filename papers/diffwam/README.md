# DiffWAM (arXiv:2611.19904) 분석 노트

*DiffWAM: Learning World Action Model from Consecutive Visual Difference* (arXiv v1, 2026-11-15)을 표 재계산, 통계 검정, 주장 대조로 분석한 자료입니다. 학회 표기는 없습니다. 공식 코드는 아직 공개되지 않았습니다("Code coming soon."). 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.19904에서 받을 수 있습니다). 논문 그림은 두 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 수준의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 약 7B 파라미터, 논문 학습은 8장 A100 기준이고 코드가 없습니다.
- **재계산과 검정은 완료했습니다.** `analysis/recompute.py`(수 초, numpy와 scipy)와 `analysis/plot_figures.py`.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 학회 확인, 코드 표, 선행 계보, 기준선 값 출처 |
| `analysis/03_analysis.md` | 재계산, 과제별 검정, 일관성 검사 |
| `analysis/04_claim_verification.md` | 주장 대조, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 두 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib pymupdf
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 논문 그림 두 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2611.19904"). 실로봇 사진, 데이터셋 프레임, 시뮬 장면 그림은 넣지 않았습니다.
- 공식 저장소(github.com/ZGCA-HMI-Lab/DiffWAM)에는 코드가 없고 라이선스 파일도 없습니다. 이 저장소에는 코드를 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

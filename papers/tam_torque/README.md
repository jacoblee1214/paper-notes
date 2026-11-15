# TAM (arXiv:2606.06218, CoRL 2026 Spotlight) 분석 노트

*TAM: Torque Adaptation Module for Robust Motion Transfer in Manipulation* (arXiv v3, 2026-10-12)을 표 재계산, 통계 검정, 2-링크 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.06218에서 받을 수 있습니다). 논문 그림은 두 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **실로봇 재현(추종, 과제 성공률)은 이 맥에서 불가입니다.** Franka 로봇, 토크 명령 인터페이스, 컨트롤러 브리지가 필요합니다.
- **공식 코드의 학습은 CUDA를 전제합니다.** 사실상 이 맥에서 어렵습니다.
- **구조 재구현과 토이는 가능합니다.** `analysis/toy_tam_2link.py`(약 3.5분, numpy)와 `analysis/toy_inverse_roundtrip.py`(수 초)로 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 결과, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_tam_2link.py` | 토이: 2-링크 팔의 토크 잔차 보정 (numpy, 약 3.5분) |
| `analysis/toy_inverse_roundtrip.py` | 토이: 역변환 왕복 검사 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 두 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_inverse_roundtrip.py
python3 analysis/toy_tam_2link.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 두 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2606.06218"). 실로봇 사진 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/Dongwon-Son/TAM`에 있습니다. `pyproject.toml`은 MIT라고 적지만 LICENSE 파일이 없습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

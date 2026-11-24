# CompVLA (arXiv:2609.23614, CoRL 2026 포스터) 분석 노트

*CompVLA: A Variable Compliance Vision-Language-Action Model for Contact-rich Manipulation* (arXiv v2, 2026-11-06)를 표 재계산, 통계 검정, 공식 코드 정적 대조, 1축 접촉 toy로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2609.23614에서 받을 수 있습니다). 논문 그림은 두 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **실로봇 재현(표 1–4)은 이 맥에서 불가입니다.** Franka FR3 두 대, F/T 센서가 필요하고 임피던스 제어기가 공개되지 않았습니다.
- **공식 코드의 학습은 CUDA와 큰 GPU 메모리를 전제합니다.** 전체 미세조정은 약 33 GB(추정)입니다.
- **라벨 규칙, 제어 수식 검산, 1축 접촉 toy는 가능합니다.** `analysis/toy_contact_1d.py`와 `analysis/toy_gic_checks.py`(각각 수 초)로 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, toy 결과, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_contact_1d.py` | toy: 1축 준정적 접촉 (numpy, 수 초) |
| `analysis/toy_gic_checks.py` | toy: 식 (5)와 분리 오차 검산 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |
| `report/figures/` | 그래프 두 장(우리가 그림)과 논문 그림 두 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_contact_1d.py
python3 analysis/toy_gic_checks.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 두 장(그림 1의 도식, 그림 6(b)의 측정 그래프)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2609.23614"). 로봇 사진과 화면 캡처는 넣지 않았습니다.
- 공식 코드는 `github.com/compvla/CompVLA`(Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다. 원자료(CC BY 4.0, 5.8 GB)와 가중치는 받지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, toy, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

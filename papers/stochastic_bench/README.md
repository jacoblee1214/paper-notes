# Stochastic Challenges (arXiv:2611.12804, CoRL 2026 포스터) 분석 노트

*Benchmarking Real-World Robotic Manipulation through Stochastic Challenges* (arXiv v1, 2026-11-09)를 표 재계산, 통계 검정, 소표본 토이, 주장 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.12804에서 받을 수 있습니다). 논문 그림은 그림 1 한 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **통계 틀(필요 시드 수, 시드 공유 설계, 판정 규칙)은 일반 노트북 CPU에서 구현하고 시험할 수 있습니다.** `analysis/recompute.py`(수 초)와 `analysis/toy_rule_check.py`(약 2초)로 확인했습니다.
- **논문의 실험(Spot과 Stretch, 두 연구실, 568회)은 재현할 수 없습니다.** 로봇, 연구실, 평가 스택이 필요합니다.
- **벤치마크 생성기와 서버는 공개되어 있지 않습니다.** 코드 대조 대신 논문의 주장을 재계산과 공개 자료에 맞춰 확인했습니다(`analysis/04_claim_verification.md`).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 1만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 통계 틀, 실험 수치, 버전, 프로젝트 페이지와의 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, GitHub 표 |
| `analysis/03_analysis.md` | 재계산 V1–V10, 토이, 검사 요약 |
| `analysis/04_claim_verification.md` | 주장 C1–C19 대조, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_rule_check.py` | 토이: 판정 규칙의 소표본 거동 (numpy, scipy, 약 2초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_rule_check.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 논문 그림은 그림 1 삽화 한 장만 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2611.12804"). 실로봇 사진이 들어간 그림은 넣지 않았습니다.
- 프로젝트 페이지(rpl-bonn.github.io/stochastic-challenges)와 KLX Benchmark 사이트는 읽기만 했고, 사이트 번들은 정적으로 읽었으며 실행하거나 로그인하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

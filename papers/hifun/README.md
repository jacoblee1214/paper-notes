# HiFun (arXiv:2610.19518, CoRL 2026 포스터) 분석 노트

*HiFun: a Hierarchical Framework for Efficient Functional Dexterous Manipulation Learning* (arXiv v1, 2026-10-14)을 표 재계산, 통계 검정, 작은 numpy 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2610.19518에서 받을 수 있습니다). 논문 그림은 차트 그림 네 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 표의 실로봇 재현은 이 맥에서 불가입니다.** Franka 로봇, Sharpa 또는 XHand 손, 손끝 힘 센서, Gello가 필요합니다.
- **공식 코드는 확인하지 못했습니다.** 프로젝트 페이지가 가리키는 저장소를 2026-11-26에 열 수 없었습니다. 코드와의 대조는 하지 못했습니다.
- **구조의 소형 시뮬 재구현과 토이는 가능합니다.** `analysis/toy_iaw.py`(수 초, numpy)로 개입 이점 가중(IAW)의 거동을 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 결과 |
| `analysis/04_claim_verification.md` | 주장 판정, 구현에 빠진 정보, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_iaw.py` | 토이: 개입 품질이 섞일 때 IAW (numpy, 약 16초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 네 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_iaw.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 표시는 찾지 못했습니다.
- 논문 그림 네 장(그림 4, 5, 7, 8)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2610.19518"). 로봇 사진이 들어간 그림은 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 코드와 가중치, 데이터셋은 받지 않았습니다.

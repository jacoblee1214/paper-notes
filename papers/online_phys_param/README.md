# Beyond Semantic Classes (arXiv:2611.10107 v1, CoRL 2026 포스터) 분석 노트

*Beyond Semantic Classes: Online Physical Parameter Estimation via Embedding-Conditioned Gaussian Mixtures* (Borca-Tasciuc, Sharma, Djeumou)을 표 재계산, 그림 주석 역산, 주장과 설정 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2611.10107에서 받을 수 있습니다). 논문 그림은 한 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **수식과 시뮬레이션 실험 1·2는 CPU로 구현할 수 있습니다.** numpy만으로 혼합 사후분포와 지지 기반 혼합을 구현할 수 있습니다.
- **레이싱 폐루프(BeamNG)와 실차 재현은 이 맥에서 하지 않습니다.** 실행 환경과 실차가 없습니다.
- **임베딩 정제는 조건부입니다.** BeamNG 이미지와 DINOv3 가중치가 필요합니다. 가중치 라이선스는 "Other"로 표시되어 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 한 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 실험의 틈 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소 출처, 공식 코드 확인 결과 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 부록 E 구간, 그림 10 역산, 커널 민감도 |
| `analysis/04_claim_verification.md` | 라이선스 판정, 주장별 판정, 설정 대조, 구현 가능성 |
| `analysis/recompute.py` | 본문과 표 수치 재계산 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/crop_figures.py` | 논문 그림 한 장을 원문 PDF에서 잘라냄 (원문 PDF 필요) |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 한 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 논문 그림 한 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2611.10107"). 제3자 소프트웨어 화면과 실물 사진이 있는 그림은 넣지 않았습니다.
- 공식 코드는 찾지 못했습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

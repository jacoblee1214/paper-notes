# BiDexGrasp (arXiv:2604.06589, CoRL 2026 Spotlight) 분석 노트

*BiDexGrasp: Coordinated Bimanual Dexterous Grasps across Object Geometries and Sizes* (arXiv v2, 2026-09-08)를 표 재계산과 앵커 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2604.06589에서 받을 수 있습니다). 논문 그림은 그림 3 한 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥에서 불가입니다.** 공식 코드와 데이터가 없고, 합성은 GPU, 실험은 양손 로봇이 필요합니다.
- **토이와 소형 재구현은 가능합니다.** `analysis/toy_anchor.py`(수 초, numpy)로 앵커 이동 분포를 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 한 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, v1과 v2 차이 |
| `analysis/02_prior_work.md` | 선행 계보, GitHub 자료 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 |
| `analysis/04_claim_verification.md` | 주장 대조, 저자 질문, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_anchor.py` | 토이: 앵커 이동 분포 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 한 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_anchor.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 3은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2604.06589"). 사진과 데이터셋 렌더링이 들어간 그림은 넣지 않았습니다.
- 공식 코드는 아직 공개되지 않았습니다(프로젝트 페이지: Coming soon).
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.

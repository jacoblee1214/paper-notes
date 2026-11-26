# DAM-VLA (arXiv:2606.12105, CoRL 2026 포스터) 분석 노트

*DAM-VLA: Decoupled Asynchronous Multimodal Vision Language Action model* (arXiv v1, 2026-06-10)을 표 재계산, 통계 검정, 작은 토이 두 개로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.12105에서 받을 수 있습니다). 논문 그림은 두 장만 넣었고 출처를 적었습니다. 2026-11-26 기준 공식 코드는 공개되지 않았습니다.

## 구현 가능성 (요약)

- **실로봇 재현과 X-VLA 미세조정은 일반 노트북에서 할 수 없는 규모입니다.** Franka Panda, 과제 장치, 시연 데이터, GPU가 필요합니다.
- **구조의 축소판(비동기 버퍼, 게이트 교차어텐션)은 작은 정책과 시뮬레이션으로 구현할 수 있습니다.** 노력 1–2주(추정, 측정 전).
- **수치 재계산과 토이 두 개는 끝냈습니다.** 각각 수 초입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 결과 |
| `analysis/04_claim_verification.md` | 주장별 검증, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_gca_init.py` | 토이: 새 모달리티 통합 방식별 초기 출력 변화 (torch CPU, 수 초) |
| `analysis/toy_zoh_lag_sparc.py` | 토이: 0차 유지가 지연과 SPARC에 미치는 영향 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 두 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_gca_init.py
python3 analysis/toy_zoh_lag_sparc.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 두 장(그림 2의 오른쪽 패널, 그림 4)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2606.12105"). 카메라 영상이나 실로봇 사진이 들어간 그림은 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 프로젝트 페이지는 https://intuitive-robots.github.io/DAM-VLA/ 입니다. 가중치와 데이터셋은 받지 않았습니다.

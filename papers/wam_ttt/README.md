# WAM-TTT (arXiv:2607.06988, CoRL 2026 포스터) 분석 노트

*WAM-TTT: Steering World-Action Models by Watching Human Play at Test Time* (arXiv v3, 2026-10-23)을 표 재계산, 주장 대조, 소형 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2607.06988에서 받을 수 있습니다). 논문은 arXiv 비독점 배포 라이선스라서 논문 그림도 넣지 않았고, 그림은 모두 이 자료의 작성자가 그린 그래프입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 일반 노트북에서 어렵습니다.** 로봇 세 기체, 짝 사람-로봇 데이터 2,286쌍, 8×H800 메타 학습이 필요하고 공식 코드는 확인하지 못했습니다.
- **재계산과 소형 토이는 가능합니다.** `analysis/recompute.py`(수 초)와 `analysis/toy_fast_weight.py`(torch CPU, 약 6초)로 확인했습니다.
- **메커니즘의 소형 재구현**(비디오 쪽 고속 가중치 + inner 한 단계 + 바깥 역전파)은 새 대형 자원 없이 시작할 수 있는 규모입니다(노력 2–3주 추정, 측정 전).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 판본 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공개 자료 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 결과 |
| `analysis/04_claim_verification.md` | 주장 검증, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_fast_weight.py` | 토이: 선형 기억의 용량과 inner 한 단계 (torch CPU, 약 6초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 analysis/recompute.py
python3 analysis/toy_fast_weight.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문의 arXiv 라이선스 링크는 비독점 배포 라이선스(arxiv.org/licenses/nonexclusive-distrib/1.0)입니다. 원문 PDF에서 저작권 표시는 찾지 못했습니다. 논문 그림은 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 자료의 작성자가 만들었습니다.
- 공식 코드는 확인하지 못했습니다(2026-11-26 기준). 백본 LDA-1B와 TTT 참고 구현 Spatial-TTT는 각 저장소에서 확인할 수 있고, 이 자료에는 코드를 넣지 않았습니다.
- 프로젝트 페이지: https://wam-ttt.github.io/
- 가중치와 데이터셋은 받지 않았습니다.

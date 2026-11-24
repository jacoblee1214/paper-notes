# RoboTTT: Context Scaling for Robot Policies (arXiv:2607.15275, CoRL 2026) 분석 노트

*RoboTTT* (arXiv v1, 2026-07-16)를 표·그림 수치 재계산과 주장 검증으로 분석한 자료입니다. 공식 코드가 없어서 코드 대조는 하지 않았습니다. 논문 원문과 논문 그림은 넣지 않았습니다(arXiv abs 페이지는 CC BY 4.0 링크를 표시하지만 PDF에 별도 저작권 표기가 있어 그림은 쓰지 않았습니다). 이 폴더의 그래프 네 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 사전학습은 GB200 16개 x 30K 스텝이고, 평가는 YAM 양팔 로봇 세 과제이며, 데이터와 체크포인트는 공개되지 않았습니다.
- **방법의 작동 원리(TTT 층, tanh 게이트, TBPTT, 시퀀스 행동 강제, 마스킹)는 합성 과제에서 소형 모델로 구현할 수 있습니다**(노력 1–2주 추정). 로봇 성능 수치와는 관계가 없습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 읽기 쉬운 틈 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표·그림 재계산(Fisher, Wilson, 점수 가능 범위, 스케일링 회귀, 배치 교란, 비용 모델) |
| `analysis/04_claim_verification.md` | 주장 C1–C21 검증, 본문에 없는 구현 세부, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 재계산과 일관성 검사 (numpy, scipy, 1초 미만) |
| `analysis/recompute_results.json` | 위 스크립트의 결과 |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 공개되어 있고 abs 페이지(2026-11-24 확인)는 CC BY 4.0 링크를 표시합니다. PDF 첫 쪽에 "© 2026 NVIDIA. All rights reserved." 표기가 있어서 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 그림 7, 8, 10, 12, A.1의 값은 인용과 재계산을 위해 PDF의 벡터 좌표에서 읽은 값입니다(오차 약 ±0.005).
- 공식 코드는 찾지 못했습니다. 기반 모델 GR00T N1.7(Apache-2.0 코드)의 설정 파일은 정적으로 읽었고 실행하지 않았습니다. 모델 가중치와 데이터셋은 받지 않았습니다.
- 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.

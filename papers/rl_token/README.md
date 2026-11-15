# RL Token: Bootstrapping Online RL with Vision-Language-Action Models (arXiv:2604.23073, CoRL 2026) 분석 노트

*RL Token* (CoRL 2026 구두 발표, arXiv v2 2026-04-30)을 보고 수치 재계산, 내부 일관성 검사, 그림 읽은 값 대조, 1차원이 아닌 numpy 토이 한 개로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **알고리즘:** RL token 인코더-디코더, 청크 actor-critic, 기준 행동 묶음은 PyTorch로 약 350–500줄, 3–5일 정도(추정)로 구현할 수 있습니다.
- **재현:** 실로봇 4과제, 원 VLA 가중치, GPU 학습은 이 맥에서 불가능합니다. 공식 코드는 확인하지 못했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법(식 (1)–(5)), 실험 수치, 한계, v1과 v2 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, 코드와 자료 표 |
| `analysis/03_analysis.md` | 보고 수치 재계산, 그림 읽은 값 대조, 샘플 산술, 토이 결과 |
| `analysis/04_claim_verification.md` | 주장 검증, 일관성 검사, 재현에 빠진 세부, 구현 가능성 |
| `analysis/table_check.py` | 보고 수치 재계산과 Wilson, Fisher 검정 (scipy, 수 초) |
| `analysis/toy_readout.py` | 저차원 readout과 묶음의 토이 (numpy, 약 1–2분) |
| `analysis/plot_figures.py` | 우리가 그린 그래프 세 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 -I analysis/table_check.py
python3 -I analysis/toy_readout.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv non-exclusive distribution 라이선스입니다(CC 아님, abs 페이지 확인 2026-11-15). 논문 그림과 원문 PDF는 이 저장소에 넣지 않았습니다. 원문은 arXiv:2604.23073에서 받을 수 있습니다.
- 논문의 표 수치는 인용과 재계산을 위해 옮겼습니다. 문장과 그림은 옮기지 않았습니다. 그림 9의 중앙값 등 그림에서 읽은 값은 ±1%p 정도의 오차가 있습니다.
- 공식 코드는 확인하지 못했습니다. 제3자 구현은 복제하지 않았습니다.
- 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다. 토이의 파라미터는 설계 가정이며 논문 값에 맞추지 않았습니다. 토이는 critic을 학습하지 않고 실제 보상의 행동 기울기를 쓰므로(오라클), 논문의 critic 학습을 검증하지 않습니다.

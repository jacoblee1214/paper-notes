# ContactGuard: Pre-Contact Execution Monitoring with Action-Conditioned Latent World Models (arXiv:2608.13438, CoRL 2026) 분석 노트

*ContactGuard* (arXiv v2, 2026-11-14)를 표 재계산, 두 arXiv 판 비교, 작은 토이와 구조 측정으로 분석한 자료입니다. 논문과 arXiv 페이지에서 코드나 프로젝트 링크를 찾지 못해(2026-11-26 기준) 코드 대조 대신 주장 대조를 썼습니다. 논문 원문과 논문 그림은 넣지 않았습니다. 이 폴더의 그래프 네 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 불가능합니다.** 14자유도 AgileX Piper 로봇, 카메라 3대, 과제별 약 250개 라벨 시도가 필요하고 공개 링크를 확인하지 못했습니다.
- **모델 구조는 Apple M4 Pro에서 구현해 돌릴 수 있습니다.** 서술대로 만든 랜덤 가중치 구조는 학습 13.01 M, 전체 34.78 M(논문 13.2 / 35.0 M)이고, MPS에서 한 번 통과가 46 ms(인코딩 9장을 매번 다시 할 때), 새 프레임 3장만 인코딩하면 약 23 ms(추정)입니다. CPU는 215 ms로 어렵습니다.
- 합성 환경의 축소 재현은 1–2주(추정)입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 표 1–7 수치, 주장과 증거의 틈, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, 자료 표 |
| `analysis/03_analysis.md` | 수치 재계산(V1–V16), 롤아웃 프로브 토이, 구조·지연 측정 |
| `analysis/04_claim_verification.md` | 주장 C1–C16 대조, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, scipy, 1초 미만) |
| `analysis/toy_rollout_probe.py` | 토이: 롤아웃 + 선형 프로브 대 직접 선형 프로브 (torch CPU, 약 70초) |
| `analysis/bench_monitor.py` | 논문 부록 A 구조의 파라미터 수와 지연 측정 (torch, 랜덤 가중치, MPS가 있으면 약 30초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
pip install torch                       # 토이와 구조 측정에만 필요
python3 analysis/toy_rollout_probe.py
python3 analysis/bench_monitor.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(abs 페이지 확인, 2026-11-26). 그림 1–4에는 실로봇 카메라 프레임이 들어 있어 이 저장소에는 논문 그림을 넣지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.

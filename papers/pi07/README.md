# π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities (arXiv:2604.15483, CoRL 2026) 분석 노트

*π0.7* (CoRL 2026 구두 발표, arXiv v2 2026-04-24)을 막대 그림 판독과 시행 수 복원, Fisher·Wilson 재계산, 내부 일관성 검사, 메타데이터 조건 토이 한 개로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다.

## 구현 가능성 (요약)

- **소규모 재구현:** 메타데이터 조건화, 구성요소별 드롭아웃, 블록 인과 마스크, 선형 상태 사영, 학습 시점 RTC, CFG를 작은 백본에 넣는 PyTorch 코드는 약 600–900줄, 1–2주(추정)로 가능합니다.
- **재현:** 풀 π0.7(5B), 세계 모델(BAGEL 14B), 실로봇은 가중치·코드·데이터 비공개와 하드웨어 때문에 이 맥(M4 Pro, CUDA 없음)에서 불가능합니다. 공식 코드는 확인하지 못했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 구성, 프롬프트와 드롭아웃, 데이터, 실행 구조, 그림 판독 결과, 한계, 버전·블로그 대조 |
| `analysis/02_prior_work.md` | 구성 요소별 출처, 코드와 자료 표 |
| `analysis/03_analysis.md` | 그림 판독과 시행 수 복원, 재계산(V1–V16), 산술 점검, 토이 결과 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C14, 일관성 K1–K9, 빠진 값 M1–M9, 구현 가능성 |
| `analysis/recompute.py` | 읽은 값 재계산, 시행 수 복원, Fisher·Wilson (scipy, 수 초) |
| `analysis/toy_metadata.py` | 메타데이터 조건 토이 (torch CPU, 약 20초) |
| `analysis/plot_figures.py` | 우리가 그린 그래프 세 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib torch
python3 -I analysis/recompute.py
python3 -I analysis/toy_metadata.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv non-exclusive distribution 라이선스입니다(CC 아님, abs 페이지 확인 2026-11-24). 논문 그림과 원문 PDF는 이 저장소에 넣지 않았습니다. 원문은 arXiv:2604.15483에서 받을 수 있습니다.
- 그림에서 읽은 값(막대 높이, 오차 막대)은 ±0.5–3%p 정도의 오차가 있고, 시행 수는 막대와 오차 막대에서 복원한 가정입니다. 문장과 그림은 옮기지 않았습니다.
- 공식 코드는 확인하지 못했습니다. 제3자 구현은 복제하지 않았습니다.
- 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다. 토이의 파라미터는 설계 가정이며 논문 값에 맞추지 않았습니다. 1차원 회귀라서 논문이 보고한 데이터 다양성 이득은 다루지 않습니다.

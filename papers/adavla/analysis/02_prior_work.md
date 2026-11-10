# 02. 선행연구와 관계

## 1. 계보
- **흐름 매칭과 빠른 적분:** Lipman et al. [9] 흐름 매칭 → Liu et al. Rectified Flow(arXiv:2209.03003, 직선 경로를 목표로 한 재흐름) → DPM-Solver(arXiv:2206.00927, 확산 ODE의 고차 적분) → Consistency Models(arXiv:2303.01469), Shortcut Models(arXiv:2410.12557, 적은 단계 학습).
- **VLA에서의 흐름 행동 머리:** π0 [10], π0.5 [11], SmolVLA [12], X-VLA [13]. 논문은 VLM이 KV 캐시로 행동 전문가에 맥락을 넘긴다는 구조(Sec. III-A)를 공통 전제로 씁니다.
- **VLA 가속:** DeeR-VLA [5](조기 종료), EfficientVLA [6](토큰·층 가지치기, 재학습 또는 교정 필요), MoLeVLA [23](층 건너뛰기), OpenVLA-OFT [22](병렬 디코딩), TinyVLA [4](경량 설계), RoboMamba [20](SSM).
- **VLM 토큰 가속(기준선):** SparseVLM [7], FastV [8]. 둘 다 행동 전문가를 건드리지 않습니다.
- **희소화·랭크 지표:** CATS [24](활성화 임계 희소화), 실효 랭크 [28], RankMe [29](표현 랭크와 하류 성능).
- **수치 적분:** RK2(Runge, Kutta) [26][27].

## 2. 가까운 대안

### 2.1 논문이 인용한 것 (Sec. II)
Sec. II는 VLA 일반 가속과 학습이 필요한 가지치기(EfficientVLA, MoLeVLA)를 다룹니다. 흐름 적분 단계 자체를 줄이는 방법은 논문에서 다루지 않습니다.

### 2.2 논문이 인용하지 않은 가까운 연구 (이 문서 작성자의 판단)
| 연구 | 관련 이유 | AdaVLA와 차이 |
|---|---|---|
| Rectified Flow (arXiv:2209.03003) | 직선성(straightness)을 흐름 품질의 지표로 씀. AdaVLA의 "직선이면 신뢰" 가정과 같은 뿌리 | 재흐름 학습이 필요. AdaVLA는 학습 없음 |
| DPM-Solver (arXiv:2206.00927) | 적은 단계 ODE 적분. 고차 해법의 오차 추정 | 확산 모델용이고 학습 필요 없는 해법을 씀. 적응 단계 선택은 없음 |
| Shortcut Models (arXiv:2410.12557), Consistency (arXiv:2303.01469) | 적은 단계 생성 | 학습된 모델이 필요함. 추론 시 곡률 판단 없음 |
| Real-Time Execution of Action Chunking Flow Policies (arXiv:2506.07339) | 흐름 정책의 청크 실행과 지연 문제 | 청크 재사용에 초점. 적분 단계 적응은 다루지 않음 |
| Embedded RK error estimate (Hairer–Nørsett–Wanner, Solving ODEs I) | Euler와 중점 해의 차이를 국소 오차 추정으로 씀 | AdaVLA의 C는 같은 종류의 추정이지만 비율로 정규화하고 중점 규칙과 맞추지 않음(`03` §4) |
| IMLE-VLA (arXiv:2609.10915, IROS 2026, 같은 학회) | 같은 π0.5 지연 문제를 단일 단계 생성기(cIMLE)로 풂 | §3.2 참고 |

## 3. 이 저장소의 다른 논문과의 관계

| 폴더 | 관계 | 비고 |
|---|---|---|
| `pacevla/` (arXiv:2610.25961) | 행동 청크의 **실행 속도** 조절. 시간 해상도가 다른 디코더 4개와 판정기(judge)·라우터를 학습으로 둠 | PaceVLA는 출력의 시간 해상도를, AdaVLA는 적분 단계 수를 바꿈. PaceVLA의 라우터는 학습이 필요하고, AdaVLA의 분기는 추론 시 곡률로 온라인에 정함 |
| `openwam/` (arXiv:2609.07398) | 흐름의 잡음 평면과 시간 재매개화(Wan의 shift 5 timestep warp). `noise_toy.py` | 흐름 스케줄 설계 비교의 관점. AdaVLA는 스케줄을 고정하지 않고 적응 |
| `chronospolicy/` (arXiv:2610.13929) | 관측·행동 지연과 시각 조건. 지연을 입력으로 준다는 방향 | AdaVLA의 지연 측정 조건(평균 방식, 워밍업, 장치)은 Chronos가 강조하는 시간 정의 문제와 연결됨 |
| `thinkworld/` (arXiv:2610.37666) | VLM이 손목 움직임 토큰을 생성하고 예측 일치로 RL. 추론 지연 문제 언급 | 가속 직접 관계는 없음 |
| `agentic_wam/` (arXiv:2610.30136) | 영상·코드·행동을 한 Transformer에서 함께 denoise. 추론 시 생성 단계 공유 | AdaVLA의 적응 단계 아이디어를 WAM의 joint 루프에 옮길 수 있는지는 열린 질문 |

### IMLE-VLA (arXiv:2609.10915): 초록만 읽음
- 제목: *IMLE-VLA: Fast Single-Step Action Generation for Vision-Language-Action Policies* (IROS 2026 채택, 프로젝트 페이지 있음).
- 초록 주장: 흐름·확산 행동 머리의 다단계 샘플링을 cIMLE로 학습한 단일 단계 생성기로 바꿈. π0.5에서 추론 빈도 3.67배(55 Hz 대 15 Hz), LIBERO 40과제 평균 98.0%, LIBERO-plus 교란에서도 유지, 실물 4과제에서 부드러움(jerk 2.2–3.0배 감소).
- 비교: AdaVLA는 학습이 없고 적분을 줄이며, IMLE-VLA는 학습이 필요하고 단계를 1로 고정합니다. 두 논문은 Hz 표기가 달라서 숫자를 직접 비교할 수 없습니다. 본문과 실험 절은 읽지 않았습니다.

## 4. GitHub (2026-11-10 확인)

**AdaVLA 공식 코드:** 찾지 못했습니다. 논문 PDF와 arXiv abs 페이지에 저장소 링크, 프로젝트 페이지가 없습니다. `gh search repos "AdaVLA"`는 무관한 저장소 한 개만 반환했고, `gh search code "AdaVLA"`도 관련 코드가 없었습니다. 따라서 clone하지 않았습니다.

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | 별 | 최근 커밋 |
|---|---|---|---|---|---|
| (AdaVLA) | 찾지 못함 | — | — | — | — |
| Physical-Intelligence/openpi | π0/π0.5 원본(JAX와 PyTorch). `models_pytorch/pi0_pytorch.py`의 `sample_actions`가 KV 캐시 한 번 + Euler 10단계 | 바탕 모델 공식 | Apache-2.0 | 14,589 | `215abfb` (2026-08-24) |
| huggingface/lerobot | SmolVLA, π0.5 PyTorch 포트, LIBERO 체크포인트 배포처 | 체크포인트 공식 | Apache-2.0 | 28,772 | `d490e47` (2026-11-05) |
| lerobot/pi05_libero_finetuned (HF) | 논문 [30] 체크포인트. 메타데이터만 확인, 다운로드 안 함 | 공식 | 메타데이터에 없음 | — | v044로 리다이렉트 |
| lerobot/xvla-libero (HF) | 논문 [31] 체크포인트. 메타데이터만 확인 | 공식 | — | 좋아요 8 | 2026-11-04, sha `8e6aeef` |
| 2toinf/X-VLA | X-VLA 공식 구현 | 공식 | Apache-2.0 | 761 | `6c4ecfb` (2026-10-20) |
| Lifelong-Robot-Learning/LIBERO | 벤치마크 | 공식 | MIT | 2,523 | `8f1084e` (2025-03-15) |
| TheRobotStudio/SO-ARM100 | SO-100/101 하드웨어·설계 | 공식 | Apache-2.0 | 8,058 | `a758567` (2026-10-06) |
| pkunlp-icler/FastV | FastV 기준선(논문 [8]과 이름이 같은 저장소, 저자 본인 저장소인지는 확인 안 함) | 확인 안 함 | 없음 | 600 | `d165972` (2025-01-04) |
| Gumpest/SparseVLMs | SparseVLM(ICML'25)과 SparseVLM+ 공식 구현 | 공식(README로 확인) | 확인 안 함 | 289 | `a9e7142` (2026-07-30) |

- 이 표의 "공식 여부"는 README 또는 논문 인용으로 확인한 범위입니다. HF 체크포인트 두 개와 SparseVLMs의 라이선스는 확인하지 않았습니다.
- openpi 샘플러를 읽은 결과(`03` §5): 접두사 KV는 한 번만 계산하고, 행동 전문가만 10번 돌립니다. 따라서 논문 Table I의 "단일 단계 53.75 ms"는 행동 전문가 한 번의 비용으로 해석하는 것이 맞습니다. 시간 변수 방향은 openpi가 t=1 노이즈 → t=0 데이터이고, 논문은 τ=0 노이즈 → τ=1 데이터로 반대입니다. 기능은 같습니다.

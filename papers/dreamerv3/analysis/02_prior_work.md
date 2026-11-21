# DreamerV3 선행연구와 관련 연구 (02)

분석일 2026-11-21. 논문 번호(`[n]`)는 arXiv v2의 참고문헌 번호입니다.

## 1. 계보

| 세대 | 논문 | 핵심 | DreamerV3와의 관계 |
|---|---|---|---|
| 잠재 동역학 계획 | PlaNet [24], Learning latent dynamics for planning from pixels | 잠재 상태에서 계획 | RSSM 구조의 뿌리 |
| Dreamer 1 | Dream to Control [21] | 세계 모델 안에서 상상 궤적으로 행위자·비평자 학습 | 학습 루프의 뿌리 |
| Dreamer 2 | Mastering Atari with Discrete World Models [22] | 범주형 잠재 상태 | 이산 잠재와 KL 설정의 직접 전신. 공식 코드 `danijar/dreamerv2`를 DreamerV3 코드가 기반으로 함(README "Disclaimer") |
| Dreamer 3 | 이 논문 | 고정 하이퍼파라미터, 로버스트 기법 | 본 분석 대상 |

- 논문은 Dreamer를 "세 번째 세대"라고 부릅니다(§Learning algorithm). v1 초록은 "DreamerV2보다 높다"를 강조했고, v2는 "전문가 알고리즘보다 높다"로 바꿨습니다.

## 2. 구성 요소별 출처

| 구성 요소 | 본 논문의 식·절 | 출처 (본문이 인용) | 새로운 점 |
|---|---|---|---|
| 범주형 잠재 상태와 1% 혼합 | 식 1, §World model learning | 범주 분포, straight-through [25] | 1% 균등 혼합으로 KL 폭주를 막음 |
| free bits (1 nat) | 식 3 | Kingma 외, IAF [26] | 동역학·표현 KL 모두에 적용하고 값을 1로 고정 |
| symlog 예측 | 식 8–9 | 쌍대칭 로그 변환 [34] | 입력과 재구성 목표에 씀 |
| symexp twohot 분포 예측 | 식 10–12 | 분포형 RL [28], MuZero 계열 [8] | 지수 간격 bin에 twohot 목표, 기댓값 읽기 |
| λ-반환, 목표 평활화 | 식 5 | 람다 반환 [29] | 비평자 EMA 정규화(감쇠 0.98) |
| 반환 정규화 (5–95 백분위) | 식 6–7 | REINFORCE [31], 엔트로피 정규화 [30], 제약 엔트로피 목표 [32, 33]과 비교 | 분산 정규화 대신 분위수 범위, 작은 반환은 그대로 둠 |
| 출력층 0 초기화 | §Critic learning | — | 초기 환각 보상·가치를 막음 |
| 값 정규화 대안 (PopArt) | §Robust predictions | PopArt [36] | 가중치 조정 없이 목표 크기를 다루려고 피함 |
| AGC 기울기 클리핑 | §Optimizer | Brock 외 [59] | 손실 스케일과 분리된 임계값 |
| LaProp | §Optimizer | Ziyin 외 [60] | ε를 10⁻²⁰으로 낮춤. Adam 대비 불안정을 줄임 |
| 블록 대각 GRU | §Networks | 블록 대각 순환 가중치 [64], GRU [63] | 8블록 |
| 온라인 큐 재생 | §Implementation | 온라인 재생 [57] | 균일 재생, 우선순위 재생 미사용 |

## 4. 공식 코드와 주변 저장소

| 저장소 | 관계 | 라이선스 | 최근 push | 별 | 상태 |
|---|---|---|---|---|---|
| [danijar/dreamerv3](https://github.com/danijar/dreamerv3) | **공식 구현** (JAX, README: "reimplementation ... based on the open source DreamerV2 code base") | MIT (LICENSE: Copyright (c) 2024 Danijar Hafner) | 2026-10-04 (커밋 `e01491f`, "Pin Bazel 7 for the DMLab install script (#223)") | 4,001 | 얕은 복제 `repo/` (정적 읽기 전용, 실행·설치 안 함). 공식 점수 파일 `scores/`를 재계산에 씀 |
| [danijar/dreamerv2](https://github.com/danijar/dreamerv2) | 직전 세대 공식 구현 (DreamerV3 코드의 기반) | MIT | 2023-01-21 | 1,062 | 읽지 않음 |
| [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) | 커뮤니티 PyTorch 구현 | MIT | 2026-03-08 | 888 | 읽지 않음. 맥 MPS 경로의 후보이나 호환성은 확인 안 함 |
| [burchim/DreamerV3-PyTorch](https://github.com/burchim/DreamerV3-PyTorch) | 커뮤니티 PyTorch 구현 | Apache-2.0 | 2024-02-16 | 17 | 읽지 않음 |
| [naivoder/dreamerv3](https://github.com/naivoder/dreamerv3) | 커뮤니티 PyTorch 구현 | 라이선스 없음 | 2025-08-08 | 18 | 읽지 않음 |
| [symoon11/dreamerv3-flax](https://github.com/symoon11/dreamerv3-flax) | Flax 구현, Crafter | 라이선스 없음 | 2025-11-29 | 19 | 읽지 않음 |
| [danijar.com/project/dreamerv3](https://danijar.com/project/dreamerv3/) | 프로젝트 페이지 | — | — | — | 읽음. 저널판 링크(Nature), 코드 링크 |
| [Nature 640, 647–653 (2025)](https://www.nature.com/articles/s41586-025-08744-2) | 저널판 (CC BY 4.0, 오픈 액세스) | CC BY 4.0 | 2025-04-02 온라인 공개 | — | 메타데이터만 읽음 |

- **공식 저장소의 `scores/` 폴더:** 실행별 곡선 파일(`*.json.gz`, 27개, 약 6 MB)이 들어 있습니다. Dreamer와 기준선(PPO, IMPALA, Rainbow, MuZero, DrQ-v2, CURL, D4PG, DMPO 등)의 과제별·seed별 점수를 담고 있습니다. 이 분석의 재계산(`analysis/recompute.py`)이 이 파일을 씁니다. 파일 목록과 실행 수는 `03` §V1에 있습니다.
- **`baselines.yaml`:** Atari 게임 기준값(랜덤, 사람), DMLab-30 기준값, ProcGen 기준값(랜덤, 최대)을 담습니다. 논문 표 6·7·8의 정규화 정의를 이 파일로 재현했습니다(`03` §V2).
- **다른 주요 공개 구현:** 이 분석에서 찾은 Dreamer 계열의 PyTorch 구현은 여럿입니다. 공식 구현은 JAX입니다. 맥(MPS)에서 쓸 때는 공식 코드보다 PyTorch 커뮤니티 구현이 현실적인 출발점일 수 있습니다. 그 구현이 논문 결과를 재현했는지는 확인하지 않았습니다.
- **응용 예:** DreamerV3를 드론 제어에 쓴 저장소(MuJoCo, Isaac Gym 기반)도 있습니다. 이 분석에서는 확인하지 않았습니다.

## 5. 선행연구와 비교 대상 (논문이 쓴 기준선의 출처)

| 기준선 | 논문에서의 역할 | 출처 |
|---|---|---|
| PPO | 고정 설정으로 모든 영역에 쓴 기준선. Acme 구현 | [5], [55] (구현 세부), [56] (Acme) |
| IMPALA, R2D2+ | DMLab, Minecraft 기준선 (문헌값 또는 직접 실행) | R2D2 [35], IMPALA [66], Acme 구현 [56] |
| Rainbow, IQN | Atari 기준선 | [40], [41] |
| MuZero | Atari 기준선 (문헌값) | [8] |
| EfficientZero, SPR, SimPLe, IRIS, TWM | Atari 100k 기준선 | EfficientZero [44], IRIS [45], SimPLe [17]. SPR, TWM은 표 9 문헌값 |
| D4PG, DMPO, MPO, DDPG | DMC 고유감각 기준선 | MPO 계열 [33]. D4PG, DDPG는 표 11 문헌값 |
| DrQ-v2, CURL, SAC | DMC 시각 기준선 | DrQ-v2 [47], SAC [32]. CURL은 표 12 문헌값 |
| PPG | ProcGen 기준선 | [37] |
| VPT, Voyager, Gato | Minecraft·범용 에이전트 비교 대상 (직접 비교 없음) | [20], [54], [52] |

## 6. 이 분석에서 확인하지 않은 연결

- 구현 계열(PyTorch 커뮤니티)과 공식 구현의 수치 차이는 확인하지 않았습니다.

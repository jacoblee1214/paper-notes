# 02. 선행연구와 관련 저장소

## 1. 계보

RPD는 두 흐름이 만나는 자리에 있습니다. 하나는 **정책 증류**(RL 교사를 작은 학생에 옮기기), 다른 하나는 **VLA 파인튜닝과 RL 개선**입니다.

| 흐름 | 대표 연구 (논문 참고문헌 번호) | RPD와의 관계 |
|---|---|---|
| 정책 증류 (DQN) | Rusu 외 2016 [20] | 교사가 RL 정책이고 손실은 Q값 기반 |
| 정책 증류 (PPO, KL) | Green 외 2019 [21] | DQN 교사를 PPO 학생에 KL로 증류. 교사는 RL 정책 |
| 정책 증류 (재사용) | Agarwal 외 2022 [22] | 다른 RL 알고리즘의 교사를 지도학습 방식으로 옮김 |
| 정책 증류 개요 | Czarnecki 외 2019 [19] | 정책 증류의 일반 틀 |
| 근접 정책 증류 (PPD) | Spigler 2024 [23] | 가장 가까운 대안. 학생 PPO에 KL 항과 클리핑 추가. 교사는 PPO. RPD 논문에서 비교 기준 |
| BC와 RL 결합 | Rajeswaran 외 2018 [25], Libardi 외 2021 [26] | 고정된 시연을 탐색 안내에 씀. RPD는 생성 교사를 온라인으로 씀 |
| 오프라인 RL 결합 | Levine 외 2020 [27], Ball 외 2023 [28] | 데이터셋 기반. RPD는 온-폴리시 |
| VLA 데이터 생성 | Xu 외 2024 (RLDG) [10] | RL 전문가로 VLA를 파인튜닝. 전문가가 이미 있어야 함. RPD는 전문가를 만들지 않고 증류 |
| 교사 VLA | Octo [4], OpenVLA [7] | RPD의 교사 |
| 시뮬레이션 | ManiSkill3 [31], CleanRL [32] | RPD의 환경과 PPO 기준선 |

RPD의 차별점은 논문이 주장하는 것처럼 "교사 VLA에서 온라인 on-policy RL로 바로 가는 증류"입니다(§II). 다만 PPD와 "교사 질의 비용"에서 직접 비교하지 않았습니다. 논문은 PPD를 KL 샘플 비용 때문에 피한다고 씁니다(§II, 150–154행). 우리 `03` §4.6은 RPD도 같은 교사 질의를 에폭마다 하므로 비용이 가볍다고 말할 수 없음을 보입니다.

## 2. 공식 저장소와 의존성 (GitHub 및 HF 확인)

확인 날짜: 2026-11-11. `gh api`와 HF API로 확인했습니다. 별 수는 확인 시점 기준입니다.

| 저장소 | 무엇 | 공식? | 라이선스 | 별 | 마지막 push | 확인한 커밋 |
|---|---|---|---|---|---|---|
| [Refined-Policy-Distillation/RPD](https://github.com/Refined-Policy-Distillation/RPD) | RPD PPO 구현, `train.py`, 설정 | **공식** (저자) | Apache-2.0 | 24 | 2025-07-18 | `10becdf` (shallow clone, HEAD) |
| RobotControlStack/vlagents (구 juelg/agents) | 교사 VLA 서버와 `RemoteAgent`, `octodist`, `openvladist` | 저자 저장소 (이름 변경, 리다이렉트) | Apache-2.0 | 22 | 2026-10-21 | HEAD `1b98af4`. **2025 상태 확인은 `f51338c` (2025-07-18)** |
| mani-skill/ManiSkill (구 haosulab/ManiSkill) | 시뮬레이션, 과제, 인간 카메라 | 원 저장소 | Apache-2.0 | 3504 | 2026-08-04 | RPD는 `v3.0.0b15`로 고정 |
| octo-models/octo | Octo 모델 코드 | 원 저장소 | MIT | 1821 | 2024-07-31 | 확인 안 함 |
| openvla/openvla | OpenVLA 코드 | 원 저장소 | MIT | 7288 | 2025-03-23 | 확인 안 함 |
| HF Juelg/octo-base-1.5-finetuned-maniskill | 파인튜닝된 Octo | 저자 공개 | MIT (태그) | 다운로드 19 | 2026-01-24 | 가중치는 받지 않음 |
| HF Juelg/openvla-7b-finetuned-maniskill | 파인튜닝된 OpenVLA | 저자 공개 | MIT (태그) | 다운로드 145 | 2026-01-24 | 가중치는 받지 않음 |
| HF Juelg/RPD-maniskill | RLDS 데이터셋 | 저자 공개 | Apache-2.0 (태그) | 다운로드 1898 | 2026-01-24 | 데이터는 받지 않음 |

주의:
- **`agents` 의존성은 고정되지 않았습니다.** `pyproject.toml`은 `agents @ git+https://github.com/juelg/agents.git`로만 씁니다. 커밋을 고정하지 않았습니다.
- **현재 vlagents HEAD에는 RPD 코드가 쓰는 경로가 없습니다.** `agents.client`, `agents.policies`, `agents.evaluator_envs`, `agents.wrappers`는 HEAD의 `vlagents` 패키지에 없습니다. `octodist`, `openvladist`, `means`, `stds`도 HEAD tarball에서 찾지 못했습니다(`04` D16). 2025-08-04 이전 커밋 `f51338c`에는 모두 있습니다.
- 저장소의 `repo/`는 gitignore되어 있습니다. 실행하지 않았습니다.

## 3. 이 저장소의 다른 분석과의 관계

| 폴더 | 한 줄 | RPD와의 관계 |
|---|---|---|
| `foresight_residual_rl/` (IROS 2026) | 고정된 π0 위에서 잔차 RL, 단계 성공 예측기로 보상을 곱해 나사 조립 성공률 54.5% → 85.6% | VLA를 고정하고 RL을 얹는다는 점에서 RPD와 같은 문제. RPD는 VLA를 학생이 떠나는 교사로 쓰고, 잔차 RL은 행동을 더한다 |
| `dexpolicy/` | ViViDex 상태 정책 RL에서 탐색 잡음 σ를 0.20 → 0.04로 줄이는 스케줄 | RL 탐색을 어떻게 안내하느냐는 같은 질문. RPD는 교사 행동으로 안내, dexpolicy는 잡음 크기로 안내 |
| `thinkworld/` | VLA의 예측-관찰 일치 보상으로 GRPO 강화 | RPD와 비교하면 교사를 손실 항으로 쓰느냐 보상으로 쓰느냐의 차이 |
| `pacevla/` | 시연의 느린 속도를 물려받는 VLA를 이웃 행동 합치기로 가속 | VLA 쪽 개선. RPD와 직접 관계는 약함 |
| `adavla/` | 흐름 매칭 VLA의 적분 가속 (학습 없음) | VLA 추론 비용 관점. RPD의 교사 질의 비용(`03` §4.6)과 관련 |
| `insertion_wm/` | 잠재 세계 모델로 조립 제로샷 | RPD와 직접 관계 없음. 시뮬 기반 증류 비교 대상 정도 |
| `ddpg/` | DDPG 재현과 numpy 토이 | 기준선 구현과 토이 설계 참고. 교사가 없는 RL 기준선 |

## 4. 가까운 대안: 이 논문이 다루지 않은 것

- **PPD 대 MSE:** 논문의 PPD는 자체 구현이고 공식 코드가 없습니다. 교사 분포를 10개 샘플로 맞추므로 질의가 10배입니다. RPD는 1개를 쓰지만 에폭마다 다시 질의합니다(`03` §4.6). 두 방법의 비용을 같은 질의 수에서 비교한 결과는 없습니다.
- **BC 정규화 대 RL 정규화:** 고정 σ 가우시안 BC와 같다는 것은 `03` §4.4의 유도에 있습니다. 논문은 이 연결을 쓰지 않습니다.
- **교사 혼합 탐색:** 교사 행동을 확률 β로 실행하는 방식(DAgger, ε-greedy형 혼합)은 논문에 없습니다. RPD와 비교할 자연스러운 대안입니다.
- **무관한 교사 대조:** 논문에 없습니다(`03` §5.4).

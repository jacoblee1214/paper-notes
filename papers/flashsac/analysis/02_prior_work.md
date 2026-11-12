# FlashSAC 선행연구와 계보

## 1. 계보 한눈에

```
DPG (결정적 정책 경사)
 └ DDPG (재생 버퍼, 표적망, OU 잡음)            [1509.02971]
    └ TD3 (이중 비판자, 지연 갱신, 목표 잡음)   [1802.09477 계열]
       └ SAC (최대 엔트로피, 자동 온도)          [1801.01290]
          ├ REDQ·DroQ 계열 (앙상블, 드롭아웃)
          ├ CrossQ (BN + 표적망 없이 교차 배치)  [ICLR 2024, 논문 [8]]
          ├ SimBa, SimbaV2 (단순성 편향, 초구면 정규화) [논문 [44], [45]]
          ├ XQC (BN 기반 비판자, 높은 표본 효율)   [논문 [72]]
          ├ 병렬 시뮬레이션 계열: FastTD3 [88], FastSAC [87] (약 0.2M 파라미터)
          └ FlashSAC (이 논문): 위 요소를 한 알고리즘에 합친다
on-policy 기준선: PPO [85, 86] (RSL-RL 구현으로 비교)
모델 기반·표현 계열: TD-MPC2 [24], MR.Q [19], DreamerV3 [23]
시각 기준선: DrQ-v2 [111] (DDPG 계열 + 이미지 증강), DrM
```

논문 저자 가운데 Hojoon Lee, Jaegul Choo, Takuma Seno, Donghu Kim는 SimBa(논문 [44])와 초구면 정규화(논문 [45])의 저자이기도 하다. 그래서 가중치 투영과 BN 선택은 그 계열의 연장선으로 읽을 수 있다. 이 분석은 저자 소속을 다루지 않는다.

## 2. 가까운 대안과 차이

| 방법 | 무엇을 바꾸나 | FlashSAC와 다른 점 | 논문에서의 위치 |
|---|---|---|---|
| SAC (기준) | 최대 엔트로피, 이중 비판자 | 얕은 MLP, 배치 256, UTD 1, 1M 버퍼 | §3.2 |
| CrossQ [8] | BN과 교차 배치(논문 기술 기준) | FlashSAC는 표적망(τ 0.01)을 유지하고, 가중치 투영과 RMSNorm, 분포형 비판자를 추가 | §4.2 "교차 배치"는 CrossQ를 따른다고 명시 |
| SimbaV2 [45] | 초구면 정규화(가중치와 특징을 단위 노름으로 제한) | 같은 계열. 세부 구조(BN 사용 여부, 액터 적용 범위)는 이 분석에서 확인하지 않았다 | §2.2 |
| XQC [72] | BN 기반 비판자, 높은 표본 효율 | 같은 계열. 비교는 CPU 설정(배치 512, UTD 1)에서만 한다 | §5.2 |
| FastTD3 [88] | 병렬 환경, 작은 네트워크, 빠른 벽시계 | 같은 병렬 환경 계열이지만 큰 모델·큰 배치·적은 업데이트를 쓴다 | §5.1 (GPU 기준선) |
| FastSAC [87] | 병렬 환경에서 SAC 고속화 | 같은 계열. 공개 CSV는 없다 | §2.2 |
| PPO [86] (RSL-RL) | on-policy, 큰 병렬 샘플 | 200M step, 공개 CSV에 비교 곡선 있음 | §5.1, §5.4 |
| TD-MPC2 [24] | 잠재 세계 모델과 MPC | 세계 모델 비용 때문에 벽시계 효율이 낮다고 주장 | §5.2 |
| MR.Q [19] | 모델 기반 보조 목표로 표현 학습 | FlashSAC는 보조 목표를 쓰지 않는다. 논문은 둘이 결합 가능하다고 말함 | §5.3 |
| DrQ-v2 [111] | DDPG 계열 + 이미지 증강 | 시각 과제에서 FlashSAC가 더 안정하다고 주장 | §5.3 |
| DDPG [51] | 결정적 정책, 표적망 | 기준선 아님. 이 저장소의 DDPG 분석에서 은닉층 전체 BN이 LQR 토이에서 성능을 떨어뜨렸다(§3 참조) | §2.2 |

## 3. 이 저장소의 다른 논문 분석과의 관계

아래 연결은 해당 논문 분석 노트의 결론을 인용한 것이다. 각 노트는 독립적으로 읽을 수 있다.

- **DDPG 분석 (arXiv:1509.02971).** 같은 off-policy actor-critic 계열이다. 그 분석의 LQR 토이에서 은닉층 전체 BN은 표적망의 유무와 관계없이 성능을 떨어뜨렸다. FlashSAC는 은닉층마다 BN을 넣고 교차 배치 통계를 쓴다. 두 결과가 맞지 않아 보인다. 차이는 설정(DDPG 토이는 배치 64, 64-64 망, 결정적 정책)에서 올 수 있다. 이 분석은 SAC 토이에서 BN 설정을 따로 시험했다(03 §9).
- **DexPolicy 분석 (arXiv:2610.00360).** PPO에서 행동 잡음 σ를 학습 중에 0.20에서 0.04로 줄이는 스케줄을 다룬다. FlashSAC는 잡음 크기를 엔트로피 목표(σ_tgt 0.15)로 고정하고, 대신 잡음의 시간 상관을 반복으로 만든다. 탐색 설계를 비교할 때 한 축은 "크기 스케줄"이고 다른 축은 "시간 상관"이다.
- **Foresight Residual RL 분석 (arXiv:2607.16506).** 보상에 확률 예측기를 곱하는 설계가 가치 오차에 민감하다는 결론이다. FlashSAC의 분포형 비판자는 예측 값을 지원 [-5, 5] 안에 묶는다. 값이 범위를 벗어나지 못하게 하는 방식이라 같은 문제를 다른 방식으로 다룬다.
- **ThinkWorld 분석 (arXiv:2610.37666).** 보조 예측 손실과 강화학습을 함께 쓰는 VLA 연구다. 그 분석은 보고된 이득의 대부분이 보조 예측 손실에서 나온다고 해석했다. FlashSAC도 보조 목표 없이 순수 RL로 비교하므로, MR.Q·TD-MPC2와의 비교는 "보조 목표 대 구조 정규화"의 대조로 읽을 수 있다.
- **MPM 기울기 신뢰성 분석 (arXiv:2609.34666).** 원자 덧셈 순서와 긴 롤아웃에서 기울기가 흔들린다는 결론이다. FlashSAC의 가중치 투영과 RMSNorm은 기울기와 특징 노름을 제한하는 수단이라, 수치 안정성 문제에 대한 학습 측의 대응으로 볼 수 있다. 같은 현상을 직접 측정한 것은 아니다.

## 4. 공개 구현 (GitHub)

2026-11-12, HTTP HEAD 요청으로 저장소 주소가 살아 있는지만 확인했다. 별 개수, 라이선스, 최근 push 날짜는 API를 쓰지 않아 확인하지 않았다. 라이선스는 공식 저장소의 LICENSE 파일만 읽었다.

| 저장소 | 성격 | 라이선스 | 주소 확인 | 대조 |
|---|---|---|---|---|
| [Holiday-Robot/FlashSAC](https://github.com/Holiday-Robot/FlashSAC) | 공식 (커밋 `87edc90`, 2026-04-09) | MIT (LICENSE 파일 확인) | 200 | 04에서 정적 대조 |
| [younggyoseo/FastTD3](https://github.com/younggyoseo/FastTD3) | FastTD3 (논문 [88]으로 보이나 저자 확인 안 함) | 미확인 | 200 | 기준선 설명만 |
| FastSAC (논문 [87]) | 저장소 주소를 찾지 못함 (`younggyoseo/FastSAC` 404) | 미확인 | 미확인 | — |
| [dojeon-ai/SimbaV2](https://github.com/dojeon-ai/SimbaV2) | SimbaV2 (논문 [45]와 이름 일치) | 미확인 | 200 | 이름으로 연결 |
| [facebookresearch/mrq](https://github.com/facebookresearch/mrq) | MR.Q (논문 [19]와 이름 일치) | 미확인 | 200 | 이름으로 연결 |
| XQC (논문 [72]) | 저장소 주소 미확인 | 미확인 | 미확인 | — |
| [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2) | TD-MPC2 (논문 [24]) | 미확인 | 200 | 이름으로 연결 |
| [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) | PPO 구현 RSL-RL (논문 [86]) | 미확인 | 200 | 기준선 구현 |
| [facebookresearch/drqv2](https://github.com/facebookresearch/drqv2) | DrQ-v2 (논문 [111]) | 미확인 | 200 | 시각 기준선 |
| [adityab/CrossQ](https://github.com/adityab/CrossQ) | CrossQ (논문 [8]) | 미확인 | 200 | 교차 배치 원형 |
| [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | 시뮬레이터 (논문 [61]) | 미확인 | 200 | 환경 |
| [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground) | 시뮬레이터 (논문 [112]) | 미확인 | 200 | 환경 |
| [haosulab/ManiSkill](https://github.com/haosulab/ManiSkill) | 시뮬레이터 (논문 [95]) | 미확인 | 200 | 환경 |
| [Genesis-Embodied-AI/Genesis](https://github.com/Genesis-Embodied-AI/Genesis) | 시뮬레이터 (논문 [4]) | 미확인 | 200 | 환경 |
| [MyoHub/myosuite](https://github.com/MyoHub/myosuite) | 시뮬레이터 (논문 [9]) | 미확인 | 200 | 환경 |
| HumanoidBench (논문 [89]) | 저장소 주소 미확인 | 미확인 | 미확인 | — |
| [google-deepmind/dm_control](https://github.com/google-deepmind/dm_control) | DMC (논문 [96]) | 미확인 | 200 | 환경 |
| [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | Gym 계열 (논문 [99]) | 미확인 | 200 | 환경 |

**고르는 기준:** 이 논문을 실제로 돌리려면 공식 저장소 하나면 된다. 기준선 비교는 PPO는 RSL-RL, 시각은 DrQ-v2 공식 저장소로 맞출 수 있다. FastSAC와 XQC, HumanoidBench의 저장소는 이 분석에서 찾지 못했으므로, 그 기준선은 논문 수치나 공식 CSV에 의존한다.

## 5. 이 분석에서 중요한 선행연구 질문

1. "작은 배치와 많은 업데이트"는 off-policy의 기본값이다. FlashSAC는 반대 방향(큰 배치, 적은 업데이트)을 택했다. 이 방향이 언제 이득인지는 벽시계 기준으로 따져야 한다. 샘플 기준으로는 큰 배치가 대체로 낫다(03 §7).
2. 가중치 노름 제약은 CrossQ, SimbaV2, FlashSAC가 공통으로 쓴다. 차이는 적용 위치(비판자만, 액터까지)와 BN과의 조합이다.
3. 분포형 비판자(C51 계열)는 값 범위를 제한하는 장점이 있지만, 지원 범위를 잘못 잡으면 값이 끝에 몰린다. 이 논문은 [-5, 5]를 정규화된 보상에 맞춰 골랐다(04 D17).

# 02. 선행연구와 자료 지도: RL-Co (arXiv:2602.12628)

## 1. 연구 계보 (논문이 스스로 쓰는 위치)

논문은 세 흐름을 잇습니다(§1, §2.2, §2.3).

1. **VLA의 SFT 후 RL.** VLA는 대규모 로봇 데이터와 시각-언어 사전 지식으로 SFT를 하고, 그 뒤 RL로 성공률과 일반화를 올립니다. 논문은 FLaRe [49], SimpleVLA-RL [38], Liu 외 [12] "What can RL bring to VLA generalization?"를 씁니다. 흐름 정책 쪽은 ReinFlow [39], SAC Flow [40], πRL [50]입니다.
2. **시뮬 공동 학습(sim-real co-training).** 실데이터와 시뮬 데이터를 섞어 SFT하는 방법입니다. 논문은 Maddukuri 외 [27]의 틀(α로 섞기, 식 (1))을 쓰고, Cheng 외 [23], Yang 외 [24], Yu 외 [25], Lei 외 [26], 그리고 MimicGen 기반 공동 학습 [61–66]을 이 흐름으로 봅니다.
3. **실데이터로 시뮬 RL을 정규화.** 시뮬 RL에 행동 손실이나 행동 사전 정보를 더해 안정화합니다. 논문은 RialTo [19] 계열의 실-시뮬-실 증류, PLD [52](잔차 RL 전문가로 VLA를 SFT), 잔차 정책 학습 [51]을 가까운 대안으로 씁니다.

RL-Co의 새로움은 **(a) 시뮬 RL의 상호작용을 공동 학습에 넣은 것, (b) 실데이터 SFT 손실을 RL 손실에 더해 잊기를 막는 것, (c) 두 단계(SFT 초기화 후 RL)를 쓰는 것**입니다(§1, §4.2). 논문 스스로 "RL 알고리즘에 독립적"이라고 씁니다. 실험은 PPO(OpenVLA)와 ReinFlow(π0.5)로 했습니다.

## 2. 구성 요소별 출처 (주장에 대한 근거)

| 구성 요소 | 출처 (논문 번호) | 절 |
|---|---|---|
| VLA 모델 | OpenVLA [3], π0 [4], π0.5 [5] | §5.1 |
| 시뮬 데이터 생성 | MimicGen [61] (ManiSkill에 구현) | §5.1, B.2 |
| 시뮬 환경 | ManiSkill3 [15] | §5.1, E.1 |
| SFT 기본 절차 | Kim 외 [10] (VLA 미세조정), 각 모델의 공식 SFT 구현 | §3.1, §5.1, §4.1 |
| OpenVLA RL 구현 | Liu 외 [12] (PPO, RL의 VLA 일반화) | §5.1, App. I |
| π0.5 RL | ReinFlow [39], RLinf [69] | §5.1, App. I |
| PPO | Schulman 외 [67] | §5.1, App. I |
| LoRA | Hu 외 [68] | App. I |
| 옵티마이저 | AdamW [72] | App. I |
| 볼록 분해 충돌 형상 | CoACD [71] | App. E.1 |
| 영상 증강 기준선 | Cosmos-Transfer [70] | §5.2.3 |
| 강한 DR | 도메인 무작위화 [16, 17, 18, 59, 60] | §5.2.3, §2.3 |
| 실-시뮬-실 증류 기준 | RialTo [19] | H.2 |
| 시뮬 공동 학습 (정렬) | Maddukuri 외 [27] (α 혼합 틀, 시점 정렬) | §3.2, E.1 |
| 시뮬 물리 정렬 | 평면 밀기 [28] (물리 대응) | E.1 |
| 흐름 정책 RL | ReinFlow [39], SAC Flow [40] | §2.2 |
| 잔차 RL과 PLD | PLD [52], 잔차 정책 [51] | §2.2 |
| RL Token (RLT) | Xu 외 [48] (VLA 표현 위 경량 액터-크리틱) | §2.2 |

## 4. GitHub와 코드 자료 (확인 2026-11-15)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| 프로젝트 페이지 | https://rl-co-training.github.io/ | - | - | JS 앱이라 본문 없음. 번들에 arXiv, Hugging Face 컬렉션(`RLinf/rl-co`), RLinf 문서 링크 |
| 공식 코드 (RLinf, π0.5 공동 학습 예시) | https://github.com/RLinf/RLinf | Apache-2.0 | 별 5,652, 최근 push 2026-11-13 (커밋 `6a7dcbd`) | `repo/`에 얕은 sparse clone, 정적 읽기만. 설정 `maniskill_ppo_co_training_openpi_pi05.yaml`, 손실은 `embodied_fsdp_actor_worker.py` |
| 공식 코드 (RLinf 문서) | https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/co_training.html | - | - | 설치, 설정, 실행 절차 |
| OpenVLA 공동 학습 구현 | 저장소에 없음. 논문은 Liu 외 [12] 구현을 씀 | - | - | 확인하지 않음 (04 D4) |
| 모델 가중치 (SFT 체크포인트) | https://huggingface.co/RLinf/RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT | 미확인 | - | 받지 않음. 1 GB 이상 추정, 크기 미확인 |
| 데이터 (실 50 궤적, 혼합) | https://huggingface.co/datasets/RLinf/RLCo-Example-Real-Data , `RLCo-Example-Mix-Data` | 미확인 | - | 받지 않음. 크기 미확인 |
| 시뮬 자산 | https://huggingface.co/datasets/RLinf/RLCo-maniskill-assets | 미확인 | - | 받지 않음 |
| 공식 코드 정적 대조 | `analysis/04_code_verification.md` | - | - | 식 (2), 보상, 동결, 설정 대조 |

- 프로젝트 페이지 번들에 있는 Hugging Face 컬렉션 `RLinf/rl-co`는 위 모델과 데이터를 묶습니다. 컬렉션 페이지 자체는 열지 않았습니다.
- 공식 코드의 **라이선스(Apache-2.0)**는 저장소 LICENSE 파일에서 확인했습니다. 모델과 데이터의 라이선스는 Hugging Face 페이지에서 확인하지 않았습니다.

## 5. 이 논문이 기대는 외부 결과

- **시뮬 RL의 상대적 안정성.** 논문은 PPO와 ReinFlow가 VLA 미세조정에 쓸 수 있다는 기존 결과 [12, 39, 50]에 기댑니다. RL-Co는 이 방법들을 그대로 쓰고 실데이터 SFT만 더합니다.
- **시뮬 데이터 생성의 품질.** MimicGen [61]이 실시연을 시드로 시뮬 궤적을 만든다는 가정입니다. 시드가 실시연이라 시뮬이 실데이터의 변형에 가깝습니다(01 §5.9).
- **VLA 분포 이동 강건성.** 처음 물체, 처음 자세 이동을 강건성 지표로 씁니다. 이 지표의 기준은 기존 연구에서 가져오지 않았습니다(논문은 자체 정의).
- **실로봇 평가의 기준.** 성공 판정은 App. D.6에 있고 자세한 기준은 없습니다(04 D6).

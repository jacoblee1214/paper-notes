# 02. 선행연구와 계보

번호 [n]은 논문의 참고문헌 번호이고, arXiv ID는 논문 참고문헌 목록에 적힌 값을 그대로 옮겼습니다. SPIDER의 성공률 스크립트(§2)와 ManipTrans의 지표 정의(§2)는 원문과 코드를 직접 확인했습니다. 나머지 논문은 따로 열어 검증하지 않았습니다.

## 1. 계보도

```
[물리 기반 리타겟팅: 시연마다 따로 최적화]
  ├─ sampling 계열   CEM [27] · MPPI [28] · Predictive Sampling [29]
  │                    → DROP [30], ExoStart [31]
  │                    → SPIDER [6] arXiv:2511.09484   가상 접촉력 + annealed sampling
  │                    → Do as I Do [5] arXiv:2606.19333   영상 복원 + warmup·교란·전이 보상  (이 저장소 do_as_i_do/)
  └─ RL 계열 (모션마다 정책 하나)
                       DexMV [32] → ManipTrans [8] (사전학습 imitator + residual)
                       → DexMachina [9] arXiv:2505.24853 (물체 보조 힘을 점차 줄이는 커리큘럼)
                       → CHORD [7] arXiv:2607.00033 (contact-wrench guidance, PPO)

[휴머노이드 전신 추적: 정책 하나로 많은 모션]
  SONIC [10] (multi-motion tracking 규모 확장) · KungfuBot2 [38], PHUMA [39] (짧은 미래 프레임 조건)

[대규모 off-policy RL]
  FlashSAC [11] (같은 그룹, RSS 2026)

          ▼ 세 갈래를 합침
  ★ FlashDexRetarget: 손-물체 리타겟팅을 multi-reference tracking으로, FlashSAC로 학습
```

**차별점 요약:** 손 리타겟팅 쪽 선행연구는 sampling이든 RL이든 모두 **시연 하나에 최적화 하나**입니다. 휴머노이드 쪽은 이미 정책 하나로 수천 개 모션을 추적합니다(SONIC). 이 논문은 그 정식화를 손-물체 상호작용으로 옮기면서 생기는 문제를 짚습니다. 물체마다 모양이 다르고, 복원된 접촉이 부정확하고, 상태 분포가 넓어지고, 두 손의 역할이 비대칭입니다. 각 문제에 장치를 하나씩 붙였습니다(01 §2).

## 2. 구성 요소별 출처

| 역할 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| multi-reference tracking 정식화 | SONIC [10] (휴머노이드) | 손-물체 시연 모음 전체에 정책 하나 |
| 미래 K 프레임 조건 | KungfuBot2 [38] arXiv:2509.16638, PHUMA [39] | K = 10, Conv1D 인코더 → 128차원 |
| 시작 상태 무작위화 | 레퍼런스 위 무작위 프레임에서 시작 (코드 `sampling.mode: uniform`) | DexMachina 기준선도 같은 방식으로 수정함 |
| 손 추적 + 관절 BC 보상 | ManipTrans [8], DexMachina [9] | 손목·손끝 지수 보상, 운동학 리타겟과의 BC |
| 물체 보조 힘 커리큘럼 (코드에만 있음) | DexMachina [9]의 가상 물체 제어기 감쇠 | xfrc PD, kp 80에서 시작해 0.9배씩 감쇠 (`04` D1) |
| 성공 기준 SR_MT | ManipTrans [8] arXiv:2503.21860 | 3 cm / 30° / 6 cm / 8 cm. ManipTrans Sec. 4.1도 궤적 평균 오차에 임계값을 적용하고, 양손이면 한 손이라도 실패하면 실패로 셈 |
| 성공 기준 SR_SPIDER | SPIDER [6], `spider/postprocess/get_success_rate.py` | 10 cm / 0.5 rad. 양손 모드에서 위치를 궤적 평균으로 중심화하고 정지한 쪽 물체를 뺌 (`04` D3) |
| off-policy 학습기 | FlashSAC [11] (`Holiday-Robot/rsl_rl_flashsac` @ `9822b1b`) | 101-bin 범주형 critic, twin-Q 앙상블, 버퍼 50M, critic 1024 |
| 데이터 | TACO [2], OakInk2 [3], HOT3D [4] | CHORD [7]의 전처리 절차 |
| 시뮬레이터·환경 틀 | Isaac Sim 5.1 / Isaac Lab 2.3, mjlab의 manager 구조(코드에 vendored) | 관측·보상·종료·커리큘럼 manager |
| 운동학 리타겟 | 코드 `src/retarget/robot_hand/ik.py` (MuJoCo equality 제약: 손바닥·물체는 weld, 손끝은 connect) | 논문 본문에는 설명 없음 |

## 3. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| **Do as I Do** (`do_as_i_do/`) | **직접 기준선입니다.** Table I에서 가장 싼 기준선(66 / 62 GPU-h)이지만 SR_MT-Obj는 0.04 / 0.10입니다. 두 방법은 경쟁이면서 보완 관계이기도 합니다. Do as I Do의 앞단(단안 영상 → 4D 손-물체 궤적)은 FlashDexRetarget이 향후 과제로 남긴 "사람 영상 레퍼런스의 강건한 처리"와 정확히 맞물립니다. 우리 `do_as_i_do` 분석에서 본 SPIDER 기준의 관대함(평균 10 cm 통과, 높이 차이를 못 잡음)이 이 논문에서는 SR_MT-Obj와 SR_SPIDER의 큰 격차로 드러납니다. |
| **DexScrew** (`dexscrew/`) | 시뮬레이션 RL로 손가락 기술을 배우는 PPO·IsaacGym 계열입니다. 이 논문의 Fig. 4(d)는 multi-reference 설정에서 같은 스텝 예산이면 PPO가 거의 0이라는 결과입니다. 한 과제만 학습하는 DexScrew 설정에 그대로 적용되지는 않지만, 상태 분포가 넓은 RL에서 off-policy와 큰 critic을 먼저 검토할 근거가 됩니다. |
| **GPT-as-Tactile-Policy** (`gpt6_astra_tactile/`) | FlashDexRetarget의 정책은 링크별 접촉 wrench를 관측으로 씁니다(`04` D2). 시뮬레이션의 privileged 촉각에 해당합니다. 실제 배포로 가는 teacher-student 단계에서 촉각 센서 입력이 그 자리를 대신할 수 있는지가 두 연구를 잇는 질문입니다. |
| **Hydra-0** (`hydra0/`) | 둘 다 로봇 데이터 부족을 다루지만 경로가 다릅니다. Hydra-0은 생성 world model로, FlashDexRetarget은 사람 시연을 물리 시뮬레이션으로 검증된 로봇 궤적으로 바꿔서 접근합니다. |

## 4. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| **PHC / PULSE** (Luo et al., 휴머노이드 다중 모션 모방) | SONIC 이전의 "정책 하나로 대규모 모션 데이터셋 추적" 계보입니다. 실패한 모션에 전문가를 덧붙이는 방식 같은 운영 요령이 손 리타겟팅에도 바로 적용될 수 있습니다. | 내 사전 지식. 서지 확인 필요 |
| **DeepMimic** (Peng et al., 2018)의 reference state initialization | 코드의 무작위 시작 프레임 샘플링이 같은 기법입니다. | 내 사전 지식. 서지 확인 필요 |
| **ADD / ADD-S** (6-DoF 자세 추정 지표) | Eq. 6은 대응점 거리의 worst-k 평균이라 ADD의 변형입니다. 대칭 물체에는 ADD-S(가장 가까운 점 거리)가 쓰입니다. 대칭 도구가 많은 데이터에서는 ADD-S식 보상이 회전 대칭 문제(`03` §2)를 자연스럽게 다룹니다. SPIDER 스크립트도 ADD-AUC10을 함께 계산합니다. | 내 해석. SPIDER 코드에서 ADD-AUC10 확인 |
| **다중 과제 RL의 간섭** (예: PCGrad 계열) | 레퍼런스가 많아질수록 변환 비율이 떨어지는 Fig. 6 결과(82% → 61%)는 과제 간 간섭이나 용량 한계로 읽을 수 있습니다. | 내 해석 |

**한계:** 4절의 서지 정보는 내 사전 지식에 근거합니다. 블로그나 git에 올리기 전에 원문을 직접 확인해야 합니다.

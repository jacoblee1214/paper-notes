# 02. 선행연구와 관련 연구

출처 표기: [n]은 CHORD v2 참고문헌 번호입니다. "내 판단"이라고 쓴 곳은 논문에 없는 해석입니다.

## 1. 계보도

```
사람 손-물체 시연 → 로봇손 정책 (물리 시뮬레이션에서 추적)
  ├─ 운동학 리타겟 + 잔차 RL: ManipTrans [2] (CVPR 2025, 사전학습 imitator + 잔차)
  ├─ 물체 보조 커리큘럼 RL: DexMachina [1] (ICML 2026, VOC·위치 기반 접촉 보상·ADD-AUC)
  ├─ sampling 최적화: SPIDER [3] (MPPI식 annealing, SP-SR 지표) → Do as I Do (단안 영상 + SPIDER, 이 저장소 do_as_i_do/)
  ├─ 사람 시연 1개 + sim-to-real RL: Human2Sim2Robot [43]
  └─ 접촉 wrench로 지도: CHORD (이 논문, 2026-06)
        └─ 같은 데이터·전처리로 다중 레퍼런스 off-policy: FlashDexRetarget (arXiv:2610.01849, 이 저장소 flashdexretarget/)

접촉 wrench 공간·파지 품질
  Ferrari–Canny [32] → wrench 기반 파지 보상 RL [33–36] (정적 안정성 최적화)
  → CHORD: 정적 안정성이 아니라 "사람 접촉과 같은 물체 운동을 만드는 능력"을 비교 (Sec. 2)

전신 확장 (NVIDIA 같은 그룹)
  SONIC [20] (전신 추적 제어기) + MotionBricks [37] (모션 생성) + CARI4D [38] (3인칭 손-물체 복원)
```

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 이 논문에서의 역할 |
|---|---|---|
| VOC(가상 물체 제어) 커리큘럼, 위치 기반 접촉 보상, ADD-AUC | DexMachina [1] | 학습 보조와 "Position Only" ablation, AUC 지표 |
| 잔차 행동 공간, MP-SR 지표 | ManipTrans [2] | 리타겟 동작을 사전으로 쓰는 잔차 정책 |
| SP-SR 지표 | SPIDER [3], `spider/postprocess/get_success_rate.py` | 시간 평균 위치 오차 ≤ 10 cm, 자세 ≤ 0.5 rad. SPIDER 원 스크립트는 양손 모드에서 궤적 평균을 빼지만(`flashdexretarget/04` D3), CHORD 공개 평가 코드와 V2D 챌린지 키트는 빼지 않습니다(`04` D13, 2026-11-12 수정) |
| 시뮬레이터 | Isaac Lab [42] (PPO 경로), Newton + Warp + JAX (FlashCHORD 경로) | 코드에 두 경로가 함께 있음(`04` §1) |
| PPO 구현 | rsl_rl (`scripts/rsl_rl/train.py`) | 공개 첫 학습 경로 |
| FlashSAC | FlashSAC (Holiday Robotics·KAIST) | v2 App. B가 말하는 학습기. FlashCHORD에 JAX로 들어 있음 |
| 데이터 | ARCTIC, TACO, HOT3D, OakInk2, DexYCB, GRAB, H2O [13–16, 39–41] | 4,739개 과제의 원천 |
| 영상 복원 | MoGe, Grounding DINO, SAM2, SAM3D, FoundationPose, ViPE, Dyn-HaMR, Gaussian splat, DROID-SLAM [45–53] | App. C 붓 과제 한 건 |
| 전신 | SONIC [20], MotionBricks [37], CARI4D [38] | 잔차를 얹을 전신 제어기와 손 → 전신 inpainting |
| 파지 품질 | Ferrari–Canny ε [32] | 벤치마크 통계(Fig. 3), r_fc의 근거 |

## 3. FlashDexRetarget과의 정합: CHORD 자신의 논문은 FDR이 돌린 CHORD와 맞는가

`flashdexretarget/`에서 CHORD는 주 기준선이었습니다. XHand 50개 모션에서 SR_SPIDER 0.46, 2,847 GPU-h(RTX 3090), Sharpa에서 0.50, 3,314 GPU-h였고, FDR은 이를 근거로 "약 100배 적은 계산량, 44%p 높은 성공률"을 주장했습니다(FDR Table I, `flashdexretarget/01` §3.2). CHORD 논문과 코드로 이 비교를 다시 보면 다음과 같습니다.

### 3.1 FDR이 돌린 것은 CHORD의 PPO 공개 경로입니다

- FDR Fig. 1 캡션과 Sec. IV-A는 "CHORD 결과는 공개 구현으로 얻었고, PPO로 학습한다"고 적었습니다. FDR v1은 2026-10-01에 올라왔고, video_to_data의 첫 학습 코드 공개(v0.2.0)는 2026-08-14, FlashSAC 경로(FlashCHORD, v0.4.0)는 2026-09-23입니다. FDR의 기준선 실험은 PPO 경로만 있던 시기에 돌았을 가능성이 큽니다(실험 날짜는 FDR에 없음).
- PPO 공개 설정은 환경 4,096개 × 24스텝 × 최대 20,000 iteration = **약 19.7억 전이**이고, VOC는 iteration 12,500(약 12.3억 전이)부터 0입니다(`04` C7, `curriculum.py`의 `bisect_right` 의미로 계산).
- FDR이 잰 모션당 비용은 2,847/50 = 56.9 GPU-h(XHand), 3,314/50 = 66.3 GPU-h(Sharpa)입니다. 20,000 iteration을 다 돌렸다고 가정하면 RTX 3090에서 초당 약 9.6k·8.2k 전이입니다. Isaac Lab에서 양손 접촉 센서와 wrench 계산을 매 스텝 하는 설정으로는 그럴듯한 값입니다. 다만 이 처리량은 GPU-h에서 거꾸로 계산한 값이라 독립적인 확인은 아닙니다.

### 3.2 CHORD v2가 주장하는 레시피는 FlashSAC이고, 10배 짧습니다

- App. B: FlashSAC, 환경 2,048개, L40S 한 장에서 과제당 약 2시간. FlashCHORD 설정은 2.5억 전이이고(`sharpa_flash_sac.yaml`), VOC 임계값 9개가 PPO 일정을 정확히 1/10로 줄인 값입니다. 설정 파일 주석도 "검증된 10배 VOC 일정을 유지한다"입니다. 2시간에 2.5억이면 초당 약 34.7k 전이입니다.
- FDR이 잰 모션당 57–66 GPU-h는 논문의 2시간보다 **28–33배** 큽니다. 전이 수 비율(19.7억 / 2.5억 = 7.9배)과 처리량 차이(3090 + PhysX vs L40S + Newton/Warp, 약 3.6–4.2배)를 곱하면 이 간극이 설명됩니다.
- 가정으로 CHORD-FlashSAC을 50개 모션에 돌리면 약 100 L40S GPU-h입니다. FDR의 29(XHand)·44(Sharpa) RTX 3090 GPU-h와 비교하면 **2.3–3.4배**입니다. 하드웨어가 달라 정확한 비율은 아니지만, FDR의 "약 100배"는 주로 PPO 대 off-policy, 그리고 PPO 경로의 긴 기본 일정에서 나온 것입니다. 같은 FlashSAC을 쓰는 CHORD와 비교했다면 격차는 한 자릿수였을 것입니다. FDR 저자는 FlashSAC 저자와 같은 그룹이라, FlashCHORD가 FlashSAC을 쓴다는 점이 이 해석과 맞물립니다.

### 3.3 그런데 CHORD의 대규모 수치 자체가 어느 레시피에서 나왔는지는 불분명합니다

- 82.12%(1,831개), 90.77%(전신), Fig. 5–9의 그림 파일은 v1(2026-06-22)과 v2가 **바이트 단위로 같습니다**. FlashSAC 학습 단락은 v2에서 처음 나옵니다. 1,831개 평가가 PPO로 학습되었다면, FDR이 돌린 PPO 경로는 오히려 CHORD 자신의 대규모 결과를 만든 레시피에 가깝습니다. FlashSAC으로 학습되었다면 FDR의 비교는 CHORD의 최신 레시피를 반영하지 못한 것입니다. 논문은 어느 쪽인지 밝히지 않습니다.
- FlashCHORD의 `reference_curriculum()`은 "W&B 실행 `sizirwre`에서 복사"했다고 적혀 있고, 그 값이 PPO 설정과 같습니다. PPO 일정이 먼저 검증되고 FlashSAC으로 옮겨졌다는 정황입니다(내 판단).

### 3.4 성공률 차이: 과제·로봇·지표가 모두 다릅니다

| 항목 | CHORD 논문 Table 1 | FDR Table I의 CHORD |
|---|---|---|
| 과제 | 자체 60개(강체 17, 관절 24, 다물체 19) | FDR 50개(HOT3D 25, TACO 10, OakInk2 15) |
| 로봇 | Sharpa Wave(CHORD 기본 손) | XHand(CHORD 기본 손이 아님), Sharpa Wave |
| 학습 | 논문은 FlashSAC이라고 함 | PPO 공개 경로 |
| SPIDER 지표 | SP-SR **0.926** | SR_SPIDER **0.46**(XHand), **0.50**(Sharpa) |
| 엄격한 지표 | MP-SR 0.894 (ManipTrans식) | SR_MT 0.00, SR_MT-Obj 0.02(XHand)·0.22(Sharpa) |
| 판정 | 완료율 > 0.7 (rollout 여러 개의 비율) | FDR 쪽 판정(첫 성공에서 멈추고 실패 모션은 전체 비용 합산) |

- 같은 Sharpa 손, 같은 이름의 SPIDER 지표에서도 0.926 vs 0.50입니다. 과제 묶음, 학습 레시피, 성공 판정(완료율 > 0.7 vs FDR의 rollout 판정)이 모두 달라서 둘 중 하나가 틀렸다고 할 수는 없습니다.
- 위치 오차 정의도 다릅니다(2026-11-12 추가). FDR의 SR_SPIDER는 양손에서 궤적 평균을 빼고(`flashdexretarget/04` D3), CHORD 공개 평가 코드는 빼지 않습니다(`04` D13). 평균을 빼면 오차가 줄어드는 쪽이므로, 이 정의 차이로는 FDR 쪽 값이 더 낮은 것을 설명할 수 없습니다.
- 가장 큰 불일치는 엄격한 지표입니다. CHORD 자신은 MP-SR 0.894를 보고했는데, FDR에서는 SR_MT가 0입니다. FDR의 SR_MT는 손끝 6 cm, 관절 8 cm 조건까지 요구하고(`flashdexretarget/01` §3.1), CHORD의 MP-SR이 이 손 조건을 포함하는지는 논문에 적혀 있지 않습니다. 같은 이름의 지표라도 구현이 다를 수 있습니다.
- FDR 쪽의 불리한 요인: XHand는 CHORD의 기본 손이 아니므로 접촉 링크 대응, 마찰원뿔(μ = 0.1), 보상 가중치를 다시 맞춰야 했을 수 있습니다. FDR이 XHand용으로 튜닝했는지는 알 수 없습니다(`flashdexretarget/01` §3.2 "기준선이 모두 공개 구현 그대로").
- CHORD 쪽의 유리한 요인: Table 1의 60개 과제는 1,831개에서 고른 것이고 선택 기준이 없습니다. 기준선은 모두 CHORD 팀이 Sharpa로 옮겨 다시 구현했습니다.

**결론:** 두 논문은 모순되지 않지만, 서로의 수치를 옮겨 쓸 수 없습니다. FDR의 "100배"는 "CHORD의 PPO 공개 경로 대비"로 읽어야 합니다. CHORD의 "과제당 2시간"은 FlashCHORD 경로의 주장이고, 그 경로로 1,831개 결과를 만들었다는 증거는 논문에 없습니다. 공정한 비교는 FlashCHORD 경로로 FDR의 50개 모션을 Sharpa에서 돌리는 것입니다.

## 4. 접촉 보상 세 가지 비교: CHORD wrench, FDR 거리·점군, DexForce 힘 목표

| | CHORD (이 논문) | FlashDexRetarget | DexForce |
|---|---|---|---|
| 접촉 정보의 출처 | 사람 MANO 접촉점·법선(시연에서 추출) | 사람 손끝-물체 최근접 꼭짓점 거리 | 손끝 6축 F/T 센서로 잰 실제 힘 |
| 무엇을 맞추나 | 512방향 wrench 지지값(접촉 위치·법선이 만드는 힘·토크 방향 능력) | 손끝-물체 거리의 한쪽 상한(Eq. 7) + 물체 점군 worst-k 정합(Eq. 6) | 위치 목표 x_f = x_o + k_f f (힘을 위치 오프셋으로 환산) |
| 쓰는 곳 | RL 보상(시뮬레이션) | RL 보상·관측(시뮬레이션) | 모방학습 행동 목표 + 임피던스 제어(실물) |
| 힘의 크기 | **보지 않음.** 접촉력 방향만 쓰고 모서리를 단위 길이로 만듦(`04` D7) | 보지 않음(거리만) | **핵심.** 힘 크기가 목표 오프셋을 정함 |
| 법선 방향 | 핵심. 법선이 다르면 보상 하락 | 보지 않음(SDF 법선은 관측으로만) | 센서가 잰 힘 방향으로 간접 반영 |
| 위치 오차 | 둔감(2 cm 이동에도 0.94, `03` §2) | 직접 벌함(거리 커널 σ = 1 cm) | 관측 위치 그대로 |
| 사람 시연 잡음 | 약함. 그래서 r_fc 대안을 둠 | 한쪽 거리 상한이라 비교적 관대 | 시연 자체가 로봇손 kinesthetic이라 잡음 작음 |

내 판단:

- 세 방법은 같은 축의 강약이 아니라 **다른 축**을 봅니다. CHORD는 "어느 방향으로 밀 수 있는가", FDR은 "어디에 닿아 있는가와 물체가 맞게 움직였는가", DexForce는 "얼마나 세게 미는가"입니다.
- CHORD 보상이 힘 크기를 보지 않는다는 점이 DexForce와의 가장 큰 차이입니다. 시뮬레이션에서는 크기를 물리 엔진과 물체 추적 보상이 정해 주므로 문제가 덜하지만, 실물에서 쥐는 힘이 약해 떨어뜨리는 실패(Sec. 4.7 원격조종의 "fragile grasp")를 wrench 지지값만으로는 막지 못합니다.
- 결합 가능성: CHORD의 지지값 비교에 DexForce식 크기 항(사람 쪽 힘 추정이나 센서 측정치)을 곱하면 "방향 + 크기" 보상이 됩니다. FDR의 점군 정합은 물체 결과를, CHORD의 wrench는 접촉 원인을 맞추므로 둘은 보완 관계입니다. FDR ablation에서 상호작용 보상을 빼면 −7개였는데, 그 자리에 wrench 보상을 넣는 실험은 두 논문 모두 하지 않았습니다.

## 5. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| **FlashDexRetarget** (`flashdexretarget/`) | §3에서 정리했습니다. FDR은 CHORD의 전처리(TACO·OakInk2·HOT3D)를 그대로 쓰고, CHORD를 주 기준선으로 삼았습니다. CHORD가 "시퀀스 하나에 정책 하나"로 남긴 비용 문제를 다중 레퍼런스 정책 하나로 풀었습니다. 두 방법 모두 FlashSAC을 쓰게 되면서 남는 차이는 보상(wrench vs 거리·점군)과 정책 구조(시퀀스별 vs 공동)입니다. |
| **Do as I Do** (`do_as_i_do/`) | 같은 Sharpa Wave 손, 같은 "사람 영상 → 로봇 궤적" 문제입니다. Do as I Do는 SPIDER식 sampling MPC에 warmup·교란·전이 보상을 더했고, CHORD는 RL에 wrench 보상과 VOC를 썼습니다. 두 논문 모두 "레퍼런스가 잡음일 때 접촉 사건을 어떻게 유도하나"를 다루는데, Do as I Do는 접촉 유무의 상수 벌칙, CHORD는 r_miss·r_unintend와 r_fc로 답했습니다. CHORD Table 1에서 SPIDER가 관절 오차 41°였다는 결과는 sampling 계열이 사람 손 모양을 보존하지 않는다는 점을 보여 주고, 이는 Do as I Do에도 해당할 수 있습니다(내 판단). |
| **DexForce** (`dexforce/`) | §4에서 비교했습니다. 접촉의 "방향"(CHORD)과 "크기"(DexForce)가 서로 빠진 부분입니다. DexForce에서 force-informed 목표 대신 관측 위치를 그대로 행동으로 쓰면 성공률이 거의 0%였던 결과(`dexforce/01` §1)와 CHORD 원격조종 실험의 "쥐는 힘이 약해 실패"(Sec. 4.7)는 같은 현상을 다른 쪽에서 본 것입니다. |
| **Contact-Grounded Policy** (`contact_grounded_policy/`) | CGP는 촉각과 실제 상태를 예측해 compliance 제어기 목표로 바꾸고, CHORD는 시뮬레이션에서 접촉 wrench를 맞춥니다. CHORD 실물 배포는 Vicon 상태만 쓰고 촉각이 없습니다(저자 한계 1). CHORD 정책 관측에 이미 "손목 좌표계 접촉 위치·힘 방향"이 있으므로(App. B), 실물에서 이 관측을 촉각 센서로 대신하는 것이 CGP와 이어지는 자연스러운 다음 단계입니다. |
| **GNR** (`gnr/`) | GNR은 HOT3D 성공 기준으로 위치 150 mm를 "CHORD 관례"로 썼습니다(`gnr/04` C12). 그 값은 CHORD 논문의 15 cm이고, CHORD 코드는 PPO 경로 0.2 m, FlashCHORD 경로 0.10 m라서 코드 기준으로는 같은 관례가 아닙니다(`04` D3). GNR(샘플링 MPC 궤적으로 학습한 생성 잔차)과 CHORD(시퀀스별 RL)는 같은 HOT3D 데이터에서 "새 시연마다 얼마나 계산하나"의 반대쪽 끝입니다. |
| **Hydra-0** (`hydra0/`) | 같은 NVIDIA의 "로봇 데이터 부족" 해법이지만 경로가 다릅니다. Hydra-0은 생성 world model로, CHORD는 물리 시뮬레이션으로 검증된 궤적으로 접근합니다. CHORD 산출물은 Hugging Face에 LeRobot 형식 로봇 데이터로 공개되어 있고, 코드에 GR00T 후속 학습 경로(`groot_finetune/`)가 있습니다. 이 데이터가 Hydra-0 같은 world model의 학습 데이터가 될 수 있습니다(내 판단). |

## 6. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 연구 | 관련성 | 근거 |
|---|---|---|
| **Task wrench space (Li & Sastry 1988), 과제 지향 파지 품질** | CHORD의 "사람 wrench 집합과 같은 집합"은 과제 wrench 공간을 시연에서 얻는 방식입니다. 과제 wrench를 포함하는지(⊇)만 보는 고전 지표와 달리 CHORD는 위로도 벌합니다(too_large). | 내 사전 지식. 서지 확인 필요 |
| **Soft-finger·비틀림 마찰 모델** | 코드는 점접촉 마찰원뿔만 씁니다. 두 점 집기의 축 둘레 토크(`03` §4)는 soft finger 모델이면 생깁니다. 사람 손가락 패드는 soft finger에 가깝습니다. | 내 사전 지식 |
| **준균일 방향 집합(Fibonacci·Sobol 구면 점)** | 512개 가우시안 무작위 방향 대신 준균일 점을 쓰면 같은 개수로 덮는 각도가 줄어듭니다. 6차원에서 512개 무작위 방향은 가장 가까운 방향까지 중앙값 22°입니다(`03` §4). | 내 판단 |
| **DexTrack, PHC류 다중 모션 추적** | 시퀀스별 정책을 증류하거나 하나로 합치는 계보입니다. CHORD 4,739개를 데이터로 쓰려면 결국 이 단계가 필요합니다. | GNR 계보도(`gnr/02` §1)에도 있음 |

## 7. GitHub 정리 (2026-11-05 `gh api` 확인)

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | ★ | 마지막 push | 확인 커밋 |
|---|---|---|---|---|---|---|
| nvidia-isaac/video_to_data | CHORD 코드(`robotic_grounding/`: Isaac Lab PPO 경로 + `flash_chord/` FlashSAC 경로), 영상 복원, GR00T 후속 학습 | **공식** | 코드 Apache-2.0 + 문서 CC BY 4.0 (GitHub 표시는 NOASSERTION) | 943 | 2026-11-02 | `9cad120` (`robotic_grounding/`만 sparse clone, 미디어 제외) |
| Hugging Face `nvidia/video-to-data-robot-dexterity-task-library-and-dataset` | 4,739개 과제 라이브러리와 LeRobot 형식 로봇 데이터 | 공식 데이터 | CC BY 4.0 | — | 2026-08-21 | `2376c2e` (메타데이터만, 파일 목록 API 상한 10만 개) |
| MandiZhao/dexmachina | DexMachina [1] 기준선 | 기준선 공식 | MIT | 280 | 2026-03-23 | `adae5bf` (메타데이터만) |
| ManipTrans/ManipTrans | ManipTrans [2] 기준선 | 기준선 공식 | GPL-3.0 | 349 | 2025-10-10 | `a3d08cf` (메타데이터만) |
| facebookresearch/spider | SPIDER [3] 기준선, SP-SR 스크립트 | 기준선 공식 | CC BY-NC (LICENSE 본문, GitHub 표시 NOASSERTION) | 595 | 2026-09-22 | `4471700` (`flashdexretarget/`에서 지표 스크립트 확인) |
| tylerlum/human2sim2robot | Human2Sim2Robot [43] 기준선 | 기준선 공식 | MIT | 161 | 2026-08-27 | `894eae2` (메타데이터만) |
| Holiday-Robot/FlashSAC | FlashSAC 학습기 (FlashCHORD가 쓰는 알고리즘) | 구성 요소 | MIT | 505 | 2026-04-09 | `87edc90` (메타데이터만) |
| Holiday-Robot/FlashDexRetarget | CHORD를 기준선으로 쓴 후속 연구 | 관련 연구 공식 | MIT | 70 | 2026-10-20 | `6f95e13` (`flashdexretarget/`에서 대조) |
| leggedrobotics/rsl_rl | PPO 구현 (Isaac Lab 경로) | 구성 요소 | BSD-3 계열 (LICENSE 본문, GitHub 표시 NOASSERTION) | 3,119 | 2026-10-28 | `1156331` (메타데이터만) |
| isaac-sim/IsaacLab | 시뮬레이션 프레임워크 [42] | 구성 요소 | BSD-3-Clause (LICENSE 본문, GitHub 표시 NOASSERTION) | 8,569 | 2026-11-05 | `f644e1f` (메타데이터만) |
| newton-physics/newton | FlashCHORD의 물리 엔진 | 구성 요소 | Apache-2.0 | 5,887 | 2026-11-05 | `1abe073` (메타데이터만) |
| NVlabs/GR00T-WholeBodyControl | SONIC [20] 전신 제어기 | 구성 요소(같은 그룹) | 코드 Apache-2.0 + 가중치 NVIDIA Open Model License | 3,883 | 2026-11-03 | `1e1ec4f` (메타데이터만) |
| NVlabs/motionbricks | MotionBricks [37] inpainting 구조 | 구성 요소(같은 그룹) | 코드 Apache-2.0 + 문서·자산 CC BY 4.0 | 905 | 2026-10-26 | `54fc563` (메타데이터만) |

- video_to_data의 LICENSE 첫 줄은 "Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."이고, 이어서 "문서·skill은 CC BY 4.0, 소스 코드는 Apache-2.0의 이중 라이선스"라고 적었습니다. 코드 자체는 쓸 수 있습니다. MANO와 원천 데이터셋(ARCTIC, HOT3D 등)은 각자의 라이선스를 따로 받아야 합니다(`robotic_grounding/README.md`).
- 저장소 이력: 첫 커밋 2026-06-23(arXiv v1 다음 날), 학습 코드 공개 v0.2.0 `cbee067`(2026-08-14, arXiv v2와 같은 날), v0.3.0 `dd395a7`(2026-09-17), FlashCHORD가 처음 들어온 v0.4.0 `33129dd`(2026-09-23), 그 뒤 `aee95f2`(09-28), `8c778f9`(09-29)에서 FlashCHORD를 다듬었습니다.

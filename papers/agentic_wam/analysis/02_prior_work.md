# 02. 선행연구: AW-0는 어디에서 왔는가

## 1. 두 갈래의 계보

AW-0는 두 갈래를 하나의 생성 모델로 묶습니다. 하나는 **"미래를 예측하며 행동하는 정책"**(World Action Model)이고, 다른 하나는 **"코드로 계획하고 기술을 조합하는 에이전트"**(agentic robotics)입니다.

```
[영상 예측 → 정책]                                 [언어·코드 → 로봇]
 Finn et al. 2016 (action-conditioned video)        SayCan (Ahn et al. 2022)
 V-JEPA 2 (Assran et al. 2025)                      Code as Policies (Liang et al. 2023)
        │                                           VoxPoser (Huang et al. 2023)
 Cosmos-Predict2.5 (NVIDIA 2025)                    CodeAct (Wang et al. 2024): 코드가 텍스트보다 나은 행동 공간
        │                                                 │
 WAM: DreamZero (Ye et al. 2026)                    agentic coding: CaP-X (Fu et al. 2026), ASPIRE (Lu et al. 2026)
      Motus (Bi et al. 2025)                        frontier model 제어: Hi Robot (Shi et al. 2025), Gemini Robotics
      WLA (Yang et al. 2026): 텍스트 subtask + subgoal 영상 + 행동
        └──────────────────────────┬──────────────────────────┘
                     AW-0: 코드 · 미래 영상 · 행동을 MoT 하나에서 joint denoise
```

- **WAM의 문제(저자 관점, §1):** 영상과 행동 chunk만으로는 긴 과제를 어떤 기술로 어떻게 나눌지 정해지지 않습니다.
- **agentic robotics의 문제:** 코드는 정책 바깥에서 로봇을 조종하고, 부를 수 있는 primitive의 성능에 묶입니다. 물리 동역학을 예측하지도 않습니다.
- **AW-0의 답:** 코드를 정책 **안**의 계획 표현으로 넣습니다. 생성된 코드는 필요할 때만 밖에서 실행합니다(하이브리드).

## 2. 구성 요소별 출처

| 구성 요소 | 가져온 곳 | AW-0에서 바뀐 점 |
|---|---|---|
| Mixture-of-Transformers | Liang et al. 2025 (arXiv:2411.04996) | 모달리티별 Q/K/V/O·LN·FFN을 두고 층마다 joint attention. C → V → A blockwise causal mask |
| 코드 전문가 본체 | SDAR-1.7B (Cheng et al. 2025, arXiv:2510.06303), block diffusion LM | 임베딩·FFN·출력층을 동결하고 attention 투영만 학습. 학습 목표는 LLaDA식 전체 마스크 확산(`joint`), block diffusion과 AR은 대안 |
| 이산 확산 디코딩 | LLaDA (Nie et al. 2025, arXiv:2502.09992) | 신뢰도 상위 k개 확정을 video·action의 flow step과 같은 루프에서 수행 |
| 코드 전문가 시각 입력 | Qwen3-VL-2B vision tower (동결) + 학습 merger | VAE 특징 대신 쓰면 71.1 vs 55.6 (Table 3) |
| video 전문가 | Cosmos-Predict2.5-2B, VAE는 Wan2.2(48채널)로 교체 | 입력·출력 투영 재초기화, 다시점 3D RoPE |
| action 전문가 | DiT를 처음부터 학습 | video와 code를 attention으로 읽음 |
| 언어 조건(video·action) | Cosmos-Reason1-7B 임베딩(오프라인 계산) | |
| 코드 라벨 | Gemini 3.1 Pro(annotator) + Claude Code/Opus 4.8(coordinator) | 스킬 50개 동결, 사건 일치 기반 캐시 재사용 (부록 K) |
| 하이브리드 실행 primitive | CaP-X 스타일 API + Molmo, SAM 3, Contact-GraspNet, RoboTwin 모션 플래너 | 시뮬레이션 시연 수집은 Claude Opus 4.8이 ASPIRE 방식으로 쓴 프로그램 실행 |

## 3. 가장 가까운 대안과의 차이

| 방법 | 계획 표현 | 계획과 행동의 결합 | AW-0와의 차이 |
|---|---|---|---|
| WLA (Yang et al. 2026) | 텍스트 subtask + subgoal 영상 | 자기회귀 | AW-0는 코드를 joint denoise. Table 4의 AR 변형(54.5)이 WLA식 직렬 생성에 가까운 대조군 |
| Tang et al. 2026 | 동적으로 갱신하는 subtask 지시문 | 지시문이 video·action 조건 | 자연어 계획. Fig. 7의 Subgoal 변형(63.6)이 가장 가까운 대조군 |
| DreamZero (Ye et al. 2026) | 없음 | video + action joint diffusion | AW-0의 "w/o code"와 구조가 가장 가까움. 다만 AW-0 w/o code(32.9)는 DreamZero(46.0)보다 낮음 |
| Code as Policies / CaP-X / ASPIRE | 실행 코드 | 코드가 곧 정책(primitive 호출) | AW-0 하이브리드 모드가 이 방식을 학습된 action과 섞음 |
| Hi Robot, Gemini Robotics | 상위 VLM이 지시·도구 호출 | 상위 모델이 하위 정책을 부름 | 상위 모델이 별도. AW-0는 한 네트워크 안에서 계획 |
| π0.5 (Black et al. 2025) | VLM 백본이 고수준 subtask 텍스트 예측 가능 | flow-matching action expert | 영상 예측 없음. RoboTwin 리더보드 결과(70.66/46.0)가 논문 재현값(54.9/27.7)보다 높음 (`03` §2) |

## 4. 이 논문이 주장하는 새로움과 실제로 새로운 것

- **새롭다고 볼 만한 것:**
  - 코드(이산 마스크 확산), 영상, 행동(연속 flow)을 **같은 denoising 루프에서 함께** 만드는 구조입니다.
  - 자동 라벨 파이프라인으로 3,000시간 규모의 데이터에 프로그램을 붙인 것도 새롭습니다. 50개 스킬 라이브러리와, 사건 일치 기반 캐시 재사용이 여기에 들어갑니다.
- **조합에 가까운 것:**
  - 각 구성 요소(MoT, LLaDA식 디코딩, Cosmos 백본, CaP primitive)는 모두 기존 것입니다.
  - 하이브리드 arbitration은 규칙 기반 게이트입니다. 학습된 선택기가 아닙니다.
- **ablation이 실제로 보여 주는 것:** 계획 스트림(자연어든 코드든)과 의미 수준 시각 입력(ViT)이 있으면 성능이 크게 오릅니다(32.9 → 63.6). 코드라는 형식이 더하는 몫은 그보다 작습니다(+7.5, `03` §3).

## 5. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| Hydra-0 (`hydra0/`) | 둘 다 NVIDIA 계열 video world model(Cosmos 계열)을 정책에 쓰는 WAM입니다. Hydra-0는 행동을 이미지 평면의 점 궤적(action flow)으로 표현하고, AW-0는 64차원 말단 자세·관절 행동을 쓰면서 코드 계획을 더합니다. 두 논문 모두 "영상 예측이 행동을 돕는다"는 주장을 같은 구조의 대조군 없이 내세운다는 점에서 비슷합니다. |
| ZeroDex (`zerodex/`) | 학습된 정책 없이 VLM과 고정 primitive로 손을 움직이는 agentic 쪽 극단입니다. AW-0의 하이브리드 모드가 실패하는 이유(primitive의 실패를 그대로 물려받음, 부록 J.2)는 ZeroDex의 단계별 실패 귀속 분석과 같은 문제입니다. |
| GPT-as-Tactile-Policy (`gpt6_astra_tactile/`) | 대형 언어 모델이 느린 계획을 맡고 빠른 정책이 실행하는 구조입니다. AW-0는 계획과 실행을 한 forward에 넣어 지연을 0.04 s로 줄였다고 주장합니다. 반면 AR 코드 생성은 0.55 s가 걸립니다(Fig. 6). |
| Chronos Policy (`chronospolicy/`) | AW-0는 32 step chunk 중 24 step마다 다시 계획합니다(공개 서버 기본값). 추론 지연 0.23 s 동안의 관측 지연은 다루지 않습니다. Chronos의 시간 조건화가 보완할 수 있는 부분입니다. |
| FlashDexRetarget (`flashdexretarget/`) | 같은 Sharpa Wave 손을 씁니다. AW-0는 사람 1인칭 영상의 손 운동을 22-DoF 관절 목표로 리타겟해서 사전학습에 넣습니다(부록 B.1). 리타겟 품질 평가는 논문에 없습니다. |
| DexScrew, Contact-Grounded Policy | 접촉이 많은 다지 손 과제입니다. AW-0의 실패 사례(카드 미끄러짐, 손끝 접촉 상실, 부록 H)는 촉각 없이 시각만 쓰는 정책의 한계와 같습니다. AW-0는 촉각 입력을 쓰지 않습니다. |

## 6. 논문에 없는 관련 연구 (비교해 볼 만한 것)

- **RoboTwin 리더보드의 WAM 계열:** UniWAM, PatchWAM, 4D-WAM, PrismWAM 등이 같은 프로토콜에서 clean과 random 모두 AW-0보다 높거나 비슷합니다(`03` §2). 이 중 상당수는 AW-0 공개(2026-10-21) 전에 등록되었습니다.
- **Motus (Bi et al. 2025):** AW-0가 "전문가 기반 WAM"으로 인용하는, 구조가 가장 가까운 선행연구입니다. 하지만 Table 1이나 Table 2의 비교 대상에는 없습니다.
- **기준선 저자의 중복:** 기준선 EgoScale(Zheng et al. 2026)의 저자 일부(Ruijie Zheng, Dantong Niu, Jing Wang, Linxi Fan)가 AW-0 저자이기도 합니다. EgoScale을 RoboTwin random에서 2.9%로 보고한 것이 그 모델의 최선 설정인지는 논문만으로 판단할 수 없습니다.

# 02. 선행연구와 계보

번호 [n]은 논문 참고문헌 번호입니다. GitHub 표(§5)는 2026-10-29에 `gh api`로 직접 확인했습니다. 그 밖의 논문은 따로 열어 검증하지 않았습니다.

## 1. 계보도

```
[손 시연 데이터 수집]
  텔레옵·리타겟 [5]–[14] (Holo-Dex, Open-TeleVision, DexPilot, AnyTeleop, Bunny-VisionPro …)
  사람 영상·mocap [15, 16] (DexCap, ARCap)                    → 리타겟 문제 + 촉각 피드백 없음
  puppeteering·잔차 리타겟 [17, 22] (Tilde, ResPilot)          → 리타겟은 피하지만 촉각 피드백 없음
  hand-over-hand·외골격 [23]–[25]                              → 파지 위주, 힘 센서 없음(일부)

[시연에서 힘을 행동으로]
  팔: DMP [28, 29], 가변 강성 학습 [30], hybrid force-position(ForceMimic) [31]
      compliance를 정책이 함께 예측 [32](Adaptive Compliance Policy), 힘 일치 모방 [33]
  손: 물체 수준 임피던스·고정 파지 가정 [26, 27]
          ▼
  ★ DexForce: 손끝 F/T + kinesthetic 시연 → x_f = x_o + k_f f → 임피던스 추종 → Diffusion Policy
```

**차별점 요약:** 팔에서는 시연의 힘을 위치 목표나 강성으로 바꾸는 연구가 이미 많습니다. 손에서는 고정 파지를 가정한 연구[26, 27]가 있었습니다. DexForce는 접촉이 바뀌는 손가락 과제(잡았다 놓기, 밀어 돌리기)에서도, 별도 휴리스틱 없이 식 3 하나로 힘을 행동에 넣는다는 점을 내세웁니다. 방법 자체는 간단합니다. 기여의 무게는 "kinesthetic 시연 + 손끝 F/T + 재생으로 사람 손 없는 영상 얻기"라는 수집 절차와, 그 데이터로 한 절제 실험에 있습니다.

## 2. 구성 요소별 출처

| 역할 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| 임피던스 제어 | Hogan [40], Springer Handbook [41] | 식 1–2, k_p = 310, k_v = 2 |
| force-informed 목표 | 식 1 첫 항의 스프링 모델 (이 논문) | 식 3, k_f = 0.0045 (v1: K_f = I₃/220) |
| 손끝 6축 힘·토크 센서 | CoinFT [43] (같은 그룹의 Choi, Cutkosky) | 엄지·검지 손끝 밑 2개 |
| 정책 | Diffusion Policy [3] (CNN, ResNet18) | 관측 = 영상 + 힘·모멘트, 출력 = x_f |
| 손목 어안 카메라 | TidyBot++ [44]의 마운트를 변형 | RealSense D435 + 어안렌즈 |
| 이진 접촉 기준 | "Rotating without seeing" [45] | RGB, 0/1 절제(0.55 N 문턱) |
| 열린 루프 전략 | 펜 돌리기 [38], in-hand 실증 [39] | Stage 2 재생의 근거 |
| proprioception과 causal confusion (v1) | v1 참고문헌 [41, 42] | v1 Fig. 9 절제 후 관측에서 제외 |

## 3. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| **Contact-Grounded Policy** (`contact_grounded_policy/`) | 가장 가까운 아이디어입니다. CGP는 정책이 실제 상태와 촉각을 예측하고, **학습한** 사상으로 compliance 목표 a = x + Δ를 만듭니다. 우리 CGP 분석에서 그 사상은 준정적 강성 역행렬(Δ = K_p⁻¹ Jᵀ f)로 정확히 풀렸습니다. DexForce의 식 3은 바로 그 해석식을 데이터 수집 시점에 데카르트 공간에서 쓴 것입니다. 다만 DexForce의 k_f는 1/k_p가 아니라 1.4배 큰 값입니다(`03` §1). 두 논문을 함께 보면 "해석식으로 시작하고, 모르는 부분만 학습한다"는 설계가 자연스럽습니다. |
| **Reactive Grasp** (`reactive_grasp/`) | 둘 다 손가락 힘을 다루지만 층이 다릅니다. Reactive Grasp는 매 주기 QP로 접촉력을 분배하는 반응형 제어기입니다. DexForce는 고정 이득 임피던스로 시연 힘을 위치 목표에 담아 열린 루프로 추종하고, 피드백은 정책이 영상과 힘 관측으로 맡습니다. |
| **DexScrew** (`dexscrew/`) | 둘 다 실물 촉각·힘 데이터를 사람 시연으로 모읍니다. DexScrew는 RL로 배운 기술을 텔레옵 버튼으로 불러 쓰고, DexForce는 kinesthetic 시연으로 리타겟을 피합니다. 수집 방식은 다르지만 "힘 정보가 들어 있는 시연"이라는 목표가 같습니다. |
| **GPT-as-Tactile-Policy** (`gpt6_astra_tactile/`) | 촉각 관측이 정책에 얼마나 쓸모 있는지 묻는다는 점이 같습니다. DexForce는 6축 힘이 이진 접촉보다 낫다고 보고하고, 우리 GPT-tactile 재구현 실험은 촉각이 결정 변수(방향)를 담을 때만 이득이 컸습니다. |
| **FlashDexRetarget** (`flashdexretarget/`) | 사람 손 시연을 로봇 행동으로 바꾸는 반대쪽 경로입니다. 리타겟 데이터에는 힘이 없으므로, DexForce식 force-informed 목표를 붙이려면 시뮬레이션의 접촉력을 쓰는 방법이 필요합니다. |
| DexFIT (IROS 2026, 사용자가 함께 언급) | 운동학적으로 동형인 촉각 장갑으로 힘을 전달하는 후속 계열로 보입니다. 논문이 아직 공개되지 않아(프로젝트 페이지 "Paper: To be released") 이 문서에서는 다루지 않았습니다. |

## 4. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| 가변 임피던스·강성 학습(예: 정책이 강성까지 출력) | DexForce는 k_f를 상수로 둡니다. 인용한 [32](Adaptive Compliance Policy)처럼 강성을 함께 예측하면, k_f 민감도 문제를 정책이 흡수할 수 있습니다. | [32]는 논문 인용. 이 방향의 손 적용은 내 판단 |
| admittance 기반 kinesthetic 교시 | 시연 중 로봇을 0 강성으로 두는 대신 admittance로 두면 x_o 자체에 힘 정보가 일부 담깁니다. | 내 사전 지식. 서지 확인 필요 |
| 손끝 F/T의 모멘트로 접촉점 추정 | 손끝 밑 6축 센서는 접촉점 추정(intrinsic contact sensing)이 가능합니다. 식 3에 접촉점을 넣으면 손끝 기하 오차를 줄일 수 있습니다. | 내 사전 지식. 서지 확인 필요 |

**한계:** 4절의 서지는 내 사전 지식에 근거합니다. 공개 전에 원문을 확인해야 합니다.

## 5. GitHub 정리 (2026-10-29 `gh api` 확인)

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | ★ | 마지막 push | 확인 커밋 |
|---|---|---|---|---|---|---|
| (없음) | DexForce 코드·데이터 | — | — | — | — | 공개 저장소 없음. 웹페이지 저장소도 공개 목록에 없음 |
| Google Drive (프로젝트 페이지 링크) | 손목 카메라·어안렌즈 마운트 부품 | 공식 | 표기 없음 | — | — | 목록만 확인 |
| coin-ft/coin-ft | CoinFT 센서 제작 자료 [43] | 구성 요소 공식(공저자) | NOASSERTION | 43 | 2026-01-23 | `8166d49` |
| real-stanford/diffusion_policy | Diffusion Policy [3] | 구성 요소 공식 | MIT | 4,693 | 2024-12-24 | `5ba07ac` |
| jimmyyhwu/tidybot2 | TidyBot++ [44] (어안 카메라 마운트의 원형) | 구성 요소 | MIT | 622 | 2025-12-18 | `c8dd238` |
| yifan-hou/adaptive_compliance_policy | Adaptive Compliance Policy [32] | 관련 연구 공식 | MIT | 144 | 2026-04-18 | `74d68ec` |
| ForceMimic/hybridil | ForceMimic의 HybridIL [31] | 관련 연구 공식 | MIT | 48 | 2025-03-05 | `1defa54` |
| Stanford-TML/SpringGrasp_release | 같은 1저자의 compliant 파지 합성 (SpringGrasp) | 관련(같은 1저자) | 표기 없음 | 45 | 2025-10-15 | `07c1131` |
| simlabrobotics/allegro_hand_ros_v4 | Allegro Hand ROS 드라이버 | 하드웨어 공식 | BSD-2-Clause | 35 | 2019-06-26 | 메타데이터만 |

"DexForce"로 검색되는 다른 저장소(`DexForce/dexforce.github.io` 등)는 이름만 같은 다른 조직의 저장소로, 이 논문과 무관합니다.

# 02. 선행연구와 계보

번호 [n]은 논문의 참고문헌 번호입니다. 각 참고문헌을 따로 열어 검증하지는 않았습니다. "내 사전 지식"이라고 적은 항목은 논문 밖의 정보라 확인이 더 필요합니다.

## 1. 계보도

```
[저비용 양팔 원격조작 + action chunk 모방학습]
  ACT·ALOHA [1] (RSS 2023) → Mobile ALOHA [15] (CoRL 2024) → ALOHA Unleashed [16] (CoRL 2024)
  ↳ leader 관절각을 행동으로 기록, follower는 P 위치 서보로 추종 → δ = q_leader − q_follower가 데이터에 자연히 생김
        │
[힘 정보를 모방학습에 넣는 두 갈래]
  (a) 전용 센서: F/T few-shot IL [6] (RA-L 2025), CoinFT [12], UMI-FT [13] (ICRA 2026)
  (b) 대형 모델에 힘·토크 융합: TA-VLA [7] (CoRL 2025), ForceVLA [8] (NeurIPS 2025)
        │
[원격조작 방식 자체로 힘을 다루기]
  양방향(bilateral) 제어 + ACT: Bi-ACT [9] (AIM 2024), Bi-LAT [10] (RO-MAN 2025)
  햅틱 피드백 원격조작 + 가변 컴플라이언스 학습 [11] (IROS 2024)
        │
  ★ Beyond Implicit Force: 비양방향 leader-follower 데이터에서
      (1) 목표를 follower 상태로 바꿔 δ를 지우고(ACT-o),
      (2) 서보 전류 토크를 proprio 입력에 붙여(+τ) 회복되는지를 실물 2×2로 비교
```

**차별점 요약:** [6]–[8], [12]–[13]은 전용 센서나 대규모 모델을 전제하고, [9]–[11]은 원격조작 장치를 바꿉니다. 이 논문은 장치를 바꾸지 않고, 이미 있는 데이터의 **학습 목표**와 **입력 채널**만 바꿔서 암묵적 단서와 명시적 대리값을 분리합니다. 실험 설계는 목표(leader/follower) × 입력(q/q + τ)의 2×2 요인 설계입니다.

## 2. 구성 요소별 출처

| 역할 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| 정책 구조 | ACT [1] (CVAE + Transformer, chunk) | 그대로. 목표와 입력 차원만 바꿈(Eq. 1, 3, 5) |
| 하드웨어 | ALOHA Solo, Interbotix ViperX-300s / WidowX-250s [14] | 한 팔 leader-follower, 카메라 3대, 50 Hz |
| 토크 대리값 | Dynamixel 서보 전류 × 토크 상수(τ = k_t I) | 보상 없이 입력. Foam Stop 분석에만 J(θ)로 말단 렌치 환산(Eq. 2, 4) |
| 평가 지표 | 과제 성공률 [9]–[11] | Table IV, Foam Stop만 F_x 곡선 |
| 비교 기준선 | Diffusion Policy [5]는 언급만, VLA [2]–[4]는 동기 | 실험 비교는 ACT 변형끼리만 |

## 3. 이 저장소의 다른 논문과의 관계

| 논문 | 연결점 | 이 논문을 읽는 데 주는 것 |
|---|---|---|
| **DexForce** (`dexforce/`, arXiv:2501.10356) | 관측 위치에 측정 힘을 더한 force-informed 목표 x_f = x_o + k_f f, 재생 힘 배율 G(k_f, k_e) | leader-follower 원격조작은 k_f = 1/K_p인 force-informed 목표를 **센서 없이** 만든다. Base ACT = G ≈ 1, ACT-o = G(0) ≈ 0(+ 중력). ACT-o의 붕괴는 DexForce가 "관측 위치만 쓰면 누르지 못한다"고 보인 것과 같은 현상(`03` §1.2) |
| **Contact-Grounded Policy** (`contact_grounded_policy/`, arXiv:2603.05687) | 미래의 **실제 로봇 상태**와 촉각을 예측하고, 학습된 contact-consistency mapping M_φ(x, u)로 compliance 제어기 목표(â = x̂ + Δ)를 만든다 | ACT-o + τ는 CGP에서 M_φ를 뺀 형태다. follower 상태를 예측하고 힘 신호를 보지만, 그것을 명령 여유로 바꾸는 단계가 없다. `03` §5의 "ACT-o + τ는 어떻게 누르는가" 질문에 CGP가 구조적 답을 준다 |
| **NestDex** (`nestdex/`, arXiv:2608.13362) | leader-follower 팔 + ACT식 chunk·temporal ensembling. inner 손 정책 입력이 **30스텝 × [관절 위치, 관절 effort]** | NestDex는 effort를 **이력과 함께** 넣었다. 이 논문은 한 프레임만 넣었다(`03` §2의 qpos 이력 대조군과 같은 축). NestDex 공개 데이터에서 유지 effort가 잡기 성공을 AUC 0.91로 갈랐던 것은 전류 대리값이 수 N 수준 사건에는 충분하다는 이 논문의 결과와 맞다 |
| **Real-Time Force Regulation** (`reactive_grasp/`, arXiv:2609.30082) | 촉각 없이 관절 토크로 접촉력을 분배하는 QP. 명령 토크 τ = Jᵀ f는 Jᵀ의 null space 성분을 담지 못한다 | 반대 방향(측정 토크 → 말단 힘)에도 같은 사영 문제가 있다. Foam Stop의 F_x는 J(θ)⁻ᵀ 환산이라 특이 자세 근처에서 잡음이 커지고, 중력 G(q)를 빼지 않으면 자세 차이가 힘 차이로 보인다(`03` §7) |
| **GPT-as-Tactile-Policy** (`gpt6_astra_tactile/`, arXiv:2610.28457) | 지식 × 관측 2×2 요인 설계와 반사실 입력 개입으로 "정말 촉각을 쓰는가"를 가림 | 이 논문도 목표 × 입력 2×2다. 상호작용을 계산하면 τ의 효과는 δ가 없을 때 훨씬 크다(Wipe: leader 목표에서 +30%p, follower 목표에서 +80%p, `03` §6). 반사실 개입(무른 병에 단단한 병의 토크를 넣기)은 하지 않아서, 정책이 τ를 "따르는지"는 아직 모른다 |

## 4. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 연구 | 관계 |
|---|---|
| DexForce (arXiv:2501.10356) | §3. 힘을 입력이 아니라 **목표**에 넣는 대안. 논문은 인용하지 않음 |
| Contact-Grounded Policy (arXiv:2603.05687) | §3. 상태 예측 + 명령 변환 |
| Adaptive Compliance Policy (Hou et al., 2024, 내 사전 지식) | 6축 F/T로 목표 자세와 함께 강성을 예측. 위치 서보의 고정 K_p 대신 컴플라이언스를 정책이 고르는 방향 |
| FACTR (Liu et al., 2025, 내 사전 지식) | leader-follower에 힘 피드백을 주고 외부 관절 토크를 정책 입력으로 쓰는 커리큘럼. 이 논문의 ACT + τ와 가장 가까운데 인용되지 않음 |
| Reactive Diffusion Policy (Xue et al., RSS 2025, 내 사전 지식) | 느린 정책 + 빠른 촉각·힘 반응 루프. 이 논문의 chunk 2 s 열린 루프 문제(`03` §8)에 대한 구조적 해법 |
| LeRobot ACT 구현 (`huggingface/lerobot`) | 공개 ALOHA 데이터셋과 ACT 구현을 함께 제공. 이 논문 변형을 재구현할 가장 쉬운 기반(`04` §4) |

## 5. GitHub와 공개 자료 (2026-11-09 확인)

논문의 공식 코드, 데이터, 프로젝트 페이지는 찾지 못했습니다(arXiv abs 페이지에 링크 없음, 본문에 링크 없음, `gh search repos`로 제목·저자 검색). 아래는 기반·관련 저장소입니다. 별 수와 마지막 push는 GitHub API 값입니다.

| 저장소 | 관계 | 라이선스 | ★ | 마지막 push | 대조 commit |
|---|---|---|---|---|---|
| `tonyzhaozh/act` | ACT 원조 구현 [1]. 논문의 기반 | MIT | 2,330 | 2024-07-23 | `742c753` (2024-01-28) |
| `tonyzhaozh/aloha` | ALOHA 원조 수집·제어 코드(서보 게인, effort 기록) | MIT | 2,362 | 2024-04-19 | `06369f0` (2024-04-19) |
| `MarkFzp/act-plus-plus` | Mobile ALOHA [15]의 ACT·Diffusion Policy 구현 | MIT | 3,674 | 2024-05-15 | 대조 안 함 |
| `huggingface/lerobot` | ACT 구현 + 공개 ALOHA 데이터셋 | Apache-2.0 | 28,764 | 2026-11-08 | 대조 안 함 |
| `Interbotix/interbotix_ros_manipulators` | ViperX·WidowX ROS 드라이버 [14] | BSD-3-Clause | 183 | 2025-08-08 | 대조 안 함 |
| `ft-robotic/ForceVLA` | ForceVLA [8] 공식 코드 | Apache-2.0 | 118 | 2026-11-05 | 대조 안 함 |
| `ZZongzheng0918/Torque-Aware-VLA.github.io` | TA-VLA [7] 프로젝트 페이지(코드 아님) | 표기 없음 | 1 | — | — |

- Bi-ACT [9], Bi-LAT [10]의 공개 저장소는 찾지 못했습니다.
- Trossen Robotics의 ALOHA Solo 전용 저장소(`TrossenRobotics/aloha`)는 API에서 404였습니다. 논문의 서보 게인이 원조 ALOHA(D1)와 같은지는 확인하지 못했습니다.

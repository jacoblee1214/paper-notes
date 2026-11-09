# 02. 선행연구와 관계

## 1. 계보

| 갈래 | 대표 연구 | ResTacVLA가 가져온 것 |
|---|---|---|
| Predictive coding, 감각 약화 | Rao & Ballard 1999 [13], Friston 2010 [14], Wolpert 1998(간지럼) [15], Kilteni & Ehrsson 2022 [12] | "예측 가능한 입력은 약화하고 예측 오차에 주의를 준다"는 동기. [12]는 예측적 약화와 촉각 gating이 **서로 다른** 현상이라고 보고합니다. 논문은 둘을 한 장치(σ 게이트)로 묶습니다 |
| 이분산 불확실성 | Kendall & Gal 2017 [28] | Eq. 1의 NLL, 스칼라 σ_t. 원 논문에서 이 σ는 "입력에 내재한 잡음"(aleatoric)입니다 |
| 이산 표현 학습 | VQ-VAE [29], ViT-VQGAN [30], VQ-VAE-2 [31], SoundStream [32] | 코드북, 저차원 코사인 코드, EMA, 죽은 코드 재초기화 |
| 조건부 변조 | FiLM [33] | q_t로 예측 사전 ẑ_t를 변조해 촉각을 복원 |
| VLA 바탕 | π0 [1], π0.5 [2], PaliGemma [27], flow matching [26] | 정책 본체와 action expert 주입 위치 |
| 촉각 표현 | UniT [21](VQVAE 기반, RA-L 2025), Sparsh [20] | 촉각 인코더. 기준선 T-UniT도 같은 인코더 |
| 시각→촉각 교차 예측 | Li et al. 2019 "Connecting Touch and Vision via Cross-Modal Prediction"(arXiv:1906.06322, **인용 안 됨**) | 영상에서 촉각을 예측하는 CMP의 직접적 선행 |

## 2. 가까운 대안

### 2.1 촉각·힘을 넣은 VLA·정책 (논문 Sec. II-C)
- **촉각을 토큰으로 넣는 VLA:** VLA-Touch [7], VTLA [8], TLA [19], Tactile-VLA [25]. 논문은 이들이 촉각을 "보조 시각 특징"으로 다뤄 modality collapse를 겪는다고 봅니다. 하지만 이 논문 표에 이들과의 직접 비교는 없습니다. 비교한 촉각 기준선은 π0.5 + ResNet·UniT 인코더 두 개입니다.
- **언어로 정렬:** Beyond Sight [6]은 이종 센서를 언어 grounding으로 미세조정합니다. VITaL·"Touch begins where vision ends" [22]은 대조 학습으로 촉각을 시각·언어 공간에 맞춥니다. 논문은 정렬이 촉각 고유의 잔차 정보를 지운다고 비판합니다(Sec. II-B).
- **힘·토크:** ForceVLA [23](힘 인지 MoE), TA-VLA [24](토크 인지 설계 공간). 손목 F/T는 접촉 기하를 잃는다는 이유로 촉각과 구분합니다.
- **모달리티 불균형:** Chaudhuri et al. [9](표현 붕괴), multi-modal policy consensus [10], ImplicitRDP [11].

### 2.2 논문이 인용하지 않은 가까운 연구
- **FoAR** (He et al., arXiv:2411.15753, RA-L 2025)는 미래 접촉 예측기가 힘 특징의 주입을 조절합니다. "접촉 국면에서만 비시각 신호를 쓰는 gating"이라는 점에서 SAG와 가장 가까운 선행입니다. 다만 FoAR의 게이트는 시각과 힘 신호 둘 다에서 접촉을 예측합니다.
- **Reactive Diffusion Policy** (Xue et al., arXiv:2503.02881)는 촉각·힘으로 빠른 루프를 따로 돌리는 slow-fast 구조입니다. ImplicitRDP [11]의 전작입니다.
- **Connecting Touch and Vision** (Li et al., arXiv:1906.06322)는 위 §1 표에 적었습니다. 이 연구가 영상에서 촉각을 예측한 지 7년이 지났는데, 예측 오차 자체를 정책 입력으로 쓴 것이 ResTacVLA의 새 점입니다.

### 2.3 구성 요소별 출처

| 구성 요소 | 출처 | 이 논문에서 새로운 점 |
|---|---|---|
| 영상 → 촉각 잠재 예측 | [1906.06322], [28] | 정책 입력을 예측 오차(잔차)로 바꿈 |
| 잔차 VQ + FiLM 복원 | [29–33] | 예측 사전을 조건으로 잔차 코드만 학습 |
| σ 게이트 + 학습되는 기본 토큰 | [28], FoAR와 유사 | 게이트 입력이 σ 하나뿐 |
| action expert 토큰 주입 | π0.5 [2] | VLM 쪽을 건드리지 않음 |

## 3. 이 저장소의 다른 논문과의 관계

| 폴더 | 공통점 | 차이와 시사점 |
|---|---|---|
| `gpt6_astra_tactile/` | 삽입 과제에서 촉각의 몫을 따짐 | 그 폴더에서 반사실 촉각(거울 반전)으로 "정책이 촉각을 실제로 쓰는가"를 시험했습니다. ResTacVLA에는 이런 시험(촉각 섞기·제거, 게이트 고정)이 없어서, 이득이 촉각 정보에서 오는지 확인할 수 없습니다 |
| `contact_grounded_policy/` | 촉각을 **예측**한다 | CGP는 미래 촉각을 예측해 제어 목표로 쓰고, ResTacVLA는 현재 촉각을 영상으로 예측해 빼는 기준선으로 씁니다. CGP 분석에서 "잡음 강건성 과장"을 찾았는데, 여기 Table III도 교란 하락폭이 방법 사이에 같아 같은 패턴입니다. 두 논문 모두 CC BY가 아닙니다 |
| `dexforce/` | 접촉 과제, 소수 시연 | DexForce는 측정한 힘을 행동 목표(x_f = x_o + k_f f)에 넣습니다. ResTacVLA는 촉각을 조건으로만 쓰고, Plate Wiping의 "힘 조절"을 위치 행동만으로 합니다. 높이 −2 cm에서 세 방법이 모두 −20pp 떨어진 것은, 촉각이 있어도 힘 목표가 없으면 높이 오차를 보정하지 못한다는 해석과 어긋나지 않습니다(확인은 안 됨) |
| `reactive_grasp/` | 영상·기하로 접촉을 예상 | 그 논문은 메쉬와 추적 자세로 접점을 **해석적으로** 계산합니다(촉각 없음). CMP의 μ_t는 같은 "시각적 접촉 기대"를 학습으로 근사한 것이고, 잔차는 그 기대가 틀린 부분입니다 |
| `nestdex/` | 소표본 성공률 비교 | 같은 통계 도구(Fisher, Holm, MH)를 썼습니다. NestDex의 effort, 여기의 잔차 크기 모두 접촉 사건 신호로 쓸 수 있습니다 |
| `pacevla/` | VLA의 action expert만 바꿈 | PaceVLA는 GR00T action expert에 decoder를 여러 개 달고, 라우터가 국면에 따라 고릅니다. ResTacVLA의 국면 의존 게이트와 같은 발상입니다. 둘 다 국면 판단을 영상에서 합니다 |
| `thinkworld/` | 예측 → 비교 | ThinkWorld는 1초 뒤 DINO 특징을 예측하고, 예측-관찰 일치를 RL 보상으로 씁니다. ResTacVLA는 예측-관찰 차이(잔차)를 입력 특징으로 쓰지만 게이트는 오차가 아니라 σ로 엽니다. predictive coding을 표방하는 두 방식 중 ThinkWorld가 실제로 오차를 쓰는 쪽입니다 |

## 4. GitHub (2026-11-09 확인)

| 저장소 | 무엇인가 | 공식 여부 | 커밋 | 라이선스 |
|---|---|---|---|---|
| `Awilekong/ResTacVLA` | 프로젝트 웹페이지(`docs/index.html`, JS, CSS)만 있음. 코드 없음("Code coming soon", "Data coming soon") | 공식 | `09cdbcb` (2026-08-06, 커밋 5개) | 표기 없음 |
| `ZhengtongXu/UniT` | 촉각 인코더 [21]. VQVAE 표현 학습 코드와 데이터(Google Drive) | 구성 요소 공식 | `52a2865` (2025-04-16) | MIT |
| `Physical-Intelligence/openpi` | π0.5 [2] JAX·PyTorch 구현. LoRA 미세조정은 JAX에서만 | 구성 요소 공식 | `215abfb` (2026-08-24) | Apache-2.0 |
| `real-stanford/diffusion_policy` | DP 기준선 [34] | 구성 요소 공식 | `5ba07ac` (2024-12-24) | MIT |
| `gelsightinc/gsrobotics` | GelSight Mini SDK | 하드웨어 공식 | `321d6a2` (2025-06-25) | GPL-3.0 |
| `facebookresearch/sparsh` | 대안 촉각 SSL [20] | 관련 | `fee6a05` (2025-02-27) | 저장소 표기 NOASSERTION |

`ResTacVLA` 이름으로 GitHub를 검색하면 웹페이지 저장소 하나만 나옵니다. 공식 코드가 없으므로 clone하지 않았고, 04는 주장 검증(`04_claim_verification.md`)으로 썼습니다.

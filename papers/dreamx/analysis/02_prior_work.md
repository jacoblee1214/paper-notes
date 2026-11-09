# 02. 선행연구와 위치

## 1. 계보

```
latent action model (LAM)                         영상 생성 모델의 표현
LAPO (Schmidt & Jiang 2024), Genie (Bruce 2024)    대응·추적·flow가 창발 (Nam 2026, Vélez 2025)
   이산 latent, 인접 프레임 쌍의 IDM/FDM 병목        텍스트 조건 모델의 의미 grounding (Wiedemer 2025)
        │                                                  │
LAPA (Ye 2025) VLA 사전학습으로 확장                       │
Moto (Chen 2025c) 움직임 토큰                               │
UniVLA (Bu 2025b) 의미 특징으로 grounding                  │
villa-X (Chen 2025b) 언어·로봇 상태 동역학                   │
CoMo (Yang 2026), DiLA (Zhang 2026) 움직임/외관 분리         │
LAOM (Nikulin 2025): 방해 요소가 있으면 감독이 필요           │
        │                                                  │
        └───────────── 영상 생성 prior를 쓰는 LAM ──────────────┘
          VPP (Hu 2024): 미세조정한 영상 diffusion 특징 + 행동 레이블 정책
          AdaWorld (Gao 2025): latent action 조건 적응형 world model
          VideoWorld 2 (Ren 2026): 사전학습 영상 diffusion + 별도 latent 동역학 모델
                         │
                       DreamX: 백본 내부 표현에서 query로 latent action을 직접 읽고,
                               같은 백본을 decoder로 재구성 감독 (Cosmos-Predict2.5 2B)
```

저자들의 위치 설정(§2)은 이렇습니다. 기존 LAM은 프레임 한 쌍 같은 국소 전이에서 행동 공간을 처음부터 학습하므로, 움직임이 한 step 너머로 어떻게 이어지는지 보지 못하고 외관과 행동이 얽힌다고 봅니다. 영상 prior를 쓰는 LAM(AdaWorld, VideoWorld 2)도 영상 모델은 픽셀이나 rollout을 공급할 뿐이고, 행동 표현은 처음부터 학습한 외부 모듈이 만든다고 봅니다. DreamX는 백본 내부 표현을 직접 읽고, embodiment별 행동 encoder를 쓰지 않는다는 점을 차이로 듭니다.

**이 위치 설정에서 빠진 것.** 저자들이 인용한 LAOM(Nikulin et al. 2025)의 결론은 "방해 요소가 있으면 latent action 학습에 감독이 필요하다"입니다. DreamX는 이 문제를 "생성 prior가 외관을 맡는다"로 우회한다고 주장하지만, 사람 쪽에는 감독이 전혀 없고 로봇 쪽 행동 손실만 있습니다. 감독 없이 사람·로봇 latent가 같은 좌표에 놓이는 이유는 설명하지 않습니다(`03` §1).

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | DreamX에서 바꾼 점 |
|---|---|---|
| 영상 diffusion 백본 | Cosmos-Predict2.5 2B (Ali et al. 2025) | encoder와 decoder 양쪽으로 재사용, 각각 rank-16 LoRA, 백본 고정 |
| query readout | learnable query로 특징을 모으는 Perceiver·Q-Former 계열 방식 | 영상 토큰 열에 붙여 백본 시공간 attention을 함께 통과, latent frame당 4개, query → 영상은 프레임 causal |
| latent action + 미래 재구성 | LAPO, Genie, LAPA의 IDM/FDM 병목 | decoder를 사전학습 영상 diffusion으로 바꾸고, 텍스트 조건 자리를 latent action이 대신함 |
| 행동 보조 branch | 영상·행동 결합 생성(WAM 계열) | 레이블 있는 클립만 λ > 0, 영상과 행동 스트림이 서로를 보지 않음 |
| 압축 코드(FAE) | Gao et al. 2026 "One layer is enough"의 feature autoencoder | 토큰 배치를 유지하며 2,048 → 32차원, β = 0.01 |
| 정책 생성기 | 영상 백본을 정책으로 미세조정하는 방식(Cosmos Policy, Kim et al. 2026b) | 미래 영상 대신 압축 코드 Z를 생성, 별도 Action DiT가 단방향으로 읽음 |
| 기준선 정책 | Mimic-Video (Pai et al. 2025) | MolmoAct로 사전학습한 영상 백본에서 출발 |
| 평가 | LARYBench (Nie et al. 2026), HumanAndRobot (Human2Robot, Xie et al. 2026), CALVIN | LARYBench 영상 모델에는 중간 프레임 복원 |
| 데이터 | MolmoAct2-BimanualYAM, Bridge, DROID, AgiBot, EgoDex, EgoVerse | 2.19k시간, 사람 21.4% |

## 3. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| ThinkWorld (`thinkworld/`) | **같은 그룹의 짝 논문입니다.** 저자 4명(Chaehyun Kim, Jisoo Kim, Seungryong Kim, Fu-En Yang)이 겹치고, 둘 다 양팔 YAM과 MolmoAct2-BimanualYAM, 사람 영상 EgoDex·EgoVerse를 씁니다. 사람·로봇 공통 행동 인터페이스를 푸는 방식이 정반대입니다. ThinkWorld의 Chain-of-Dynamics는 **손목 자세 주석에서 만든 명시적 6D/12D 움직임 토큰**이라 정렬이 구성상 보장되지만 주석이 필요합니다(ThinkWorld 분석에서 "레이블 없는 사람 영상"이라는 서술이 손 자세 주석을 쓴다는 점을 지적했습니다). DreamX는 주석 없이 latent를 학습하지만 정렬이 보장되지 않습니다(`03` §1). 미래 예측 대상도 다릅니다. ThinkWorld는 1초 뒤 DINOv3 특징을, DreamX는 영상 diffusion으로 미래 프레임 자체를 재구성합니다. toy 결과로 보면 두 접근을 합치는 것이 자연스럽습니다. 사람 클립 일부에만 손목 자세 head를 붙여 latent를 고정하면 짝 클립(P 조건)과 같은 정렬 신호가 됩니다. |
| Hydra-0 (`hydra0/`) | Hydra-0는 행동을 **이미지 평면의 점 궤적(action flow)** 으로 명시적으로 표현해 Cosmos 계열 영상 world model의 조건으로 씁니다. DreamX는 같은 문제의 암묵적 버전입니다. 학습된 latent가 조건 역할을 합니다. 두 방법 모두 "이미지에서 보이는 움직임"이 공통 인터페이스라는 점은 같고, 그래서 둘 다 시점에 묶입니다. Hydra-0는 배포할 때 URDF·카메라 보정·Isaac Lab 투영이 있어야 하고, DreamX의 행동 전이는 로봇 초기 상태만 있으면 되지만 대응 관계가 보장되지 않습니다. 명시적 track은 시점이 달라도 3D 역투영으로 맞출 수 있다는 장점이 있습니다. |
| OpenWAM (`openwam/`) | OpenWAM은 영상 prior + 행동 생성(WAM)의 설계를 통제 실험으로 비교했고, 추론 때 미래 영상의 기여는 작고 이득은 학습 때 생긴다는 결과를 냈습니다. DreamX 정책은 이 관찰을 구조로 옮긴 사례입니다. 미래 영상 대신 4 × 32 압축 코드를 생성해 Mimic-Video보다 19.1배 빠릅니다. 다만 DreamX에는 "같은 백본, 예측 대상만 다름" 같은 통제 비교가 없어서(`04` C12), OpenWAM 같은 실험이 있어야 "latent 예측 대 영상 예측"을 판단할 수 있습니다. 사람 영상 비중은 OpenWAM 30%(비공개 1인칭), DreamX 21.4%(공개 EgoDex·EgoVerse)입니다. |
| AW-0 (`agentic_wam/`) | AW-0도 Cosmos-Predict2.5 2B 영상 전문가와 행동 전문가를 MoT로 묶어 함께 denoise합니다. DreamX 정책의 구조(영상 백본이 코드 Z 생성, Action DiT가 그 특징을 단방향으로 읽음)는 AW-0의 영상 → 행동 정보 흐름과 비슷합니다. AW-0 분석에서 "4단계 추론에서 행동이 읽는 미래 영상은 σ ≥ 0.625의 흐린 상태"라고 지적했는데, DreamX는 영상 대신 작은 코드를 생성하는 쪽으로 이 문제를 피합니다. 사전학습 규모도 비슷합니다(AW-0 약 2,900시간, DreamX 2.19k시간). |
| PaceVLA (`pacevla/`) | PaceVLA는 텔레옵 시연의 느린 속도를 정책이 물려받는 문제를 다룹니다. DreamX의 latent action은 VAE의 시간 압축(latent frame 1개 = 영상 4프레임)에 묶여 있어, 같은 움직임이라도 사람(빠름)과 텔레옵 로봇(느림)에서 latent frame 수와 토큰 내용이 달라집니다. 행동 전이를 open-loop로 실행하면 사람의 속도가 그대로 로봇 명령으로 옮겨질 수 있는데, 논문은 속도·fps 정규화를 다루지 않습니다. 정책 1회 호출 1.47초라는 지연도 PaceVLA가 다루는 실행 속도 문제와 맞닿습니다. |
| GNR (`gnr/`) | GNR은 사람 손 시연을 로봇 손 궤적으로 **명시적으로** 리타겟팅합니다. 손 자세를 입력으로, MPC가 푼 궤적을 목표로 조건부 flow 생성기를 학습하고, 후보를 시뮬레이터로 거릅니다. DreamX의 행동 전이는 주석·기구학·물리 검사 없이 latent만으로 하고, 대상은 평행 그리퍼의 거친 움직임(글자 쓰기, 밀기)입니다. 물리적 타당성 검사가 없는 대신 입력 요구가 거의 없습니다. DreamX latent를 "움직임 의도"로, GNR식 시뮬레이터 거르기를 타당성 검사로 쓰는 조합을 생각할 수 있습니다. |

### DreamX가 위 논문들에 더하는 것

1. **주석 없는 공통 인터페이스의 실물 시연.** ThinkWorld(손목 자세), Hydra-0(점 궤적), GNR(손 자세 + MPC)은 모두 명시적 대응을 씁니다. DreamX는 주석 없이 사람 시연을 로봇 명령으로 옮긴 실물 사례를 보였습니다. 정량 평가가 없고 같은 시점 조건이라는 제약은 있습니다.
2. **영상 백본 내부를 읽는 latent action.** LAM 계열에서 처음으로 사전학습 영상 diffusion의 내부 표현에서 직접 latent action을 뽑았고, LARYBench에서 회귀 오차를 기존 최고 대비 절반 아래로 낮췄습니다.
3. **정책 지연.** 미래 영상 대신 압축 코드를 생성하는 WAM형 정책으로 지연을 28.2초(Mimic-Video)에서 1.5초로 줄였습니다.
4. **같은 약점.** 이 저장소의 다른 논문들처럼 같은 백본에서 한 변수만 바꾼 통제 비교, 시점이 다른 조건, 여러 seed가 없습니다.

## 4. 공개 저장소 (2026-11-09 확인, GitHub API 메타데이터)

| 저장소 | 역할 | 라이선스 | ★ | 마지막 push | 확인한 커밋 |
|---|---|---|---|---|---|
| [cvlab-kaist/DreamX](https://github.com/cvlab-kaist/DreamX) | 공식 저장소 | 없음 | 5 | 2026-11-06 | `056bd46` (README "Code-Coming Soon"과 티저 그림만) |
| [nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) | 백본 코드 | Apache-2.0 (가중치는 NVIDIA Open Model License) | 1,394 | 2026-06-08 | `a2c298b` (`docs/setup.md`만 확인) |
| [LatentActionPretraining/LAPA](https://github.com/LatentActionPretraining/LAPA) | LAPA, Table 1 기준선 | MIT | 587 | 2025-01-22 | `46aca51` |
| [OpenDriveLab/UniVLA](https://github.com/OpenDriveLab/UniVLA) | UniVLA, Table 1 기준선 | Apache-2.0 | 1,145 | 2025-11-19 | `0ab9e9d` |
| [microsoft/villa-x](https://github.com/microsoft/villa-x) | villa-X, Table 1 기준선 | MIT | 211 | 2026-09-16 | `9b44899` |
| [TencentARC/Moto](https://github.com/TencentARC/Moto) | Moto, Table 1 기준선 | NOASSERTION (GitHub가 판별 못 함) | 181 | 2025-10-01 | `63244f4` |
| [Little-Podi/AdaWorld](https://github.com/Little-Podi/AdaWorld) | AdaWorld, Table 1·2 기준선 | Apache-2.0 | 264 | 2025-06-17 | `fe3701e` |
| [ByteDance-Seed/VideoWorld](https://github.com/ByteDance-Seed/VideoWorld) | VideoWorld 계열, Table 1·2 기준선 | Apache-2.0 | 808 | 2026-02-25 | VideoWorld 2 코드 포함 여부 미확인 |
| [senngadaisuki/disentangled-latent-action-world-models](https://github.com/senngadaisuki/disentangled-latent-action-world-models) | DiLA, Table 1 최고 기준선 | 없음 | 21 | 2026-09-15 | `adac869` |
| [schmidtdominik/LAPO](https://github.com/schmidtdominik/LAPO) | LAPO (계보) | 없음 | 149 | 2024-07-31 | 확인 안 함 |
| [dunnolab/laom](https://github.com/dunnolab/laom) | LAOM (Nikulin et al. 2025) | Apache-2.0 | 41 | 2025-07-08 | `e133195` |
| [mimic-video/mimic-video](https://github.com/mimic-video/mimic-video) | Mimic-Video, 실물 기준선 | Apache-2.0 | 317 | 2026-10-19 | `a38dbfe` |

- **Genie:** 공식 코드가 없습니다.
- **LARYBench:** 공식 저장소를 찾지 못했습니다(arXiv:2604.11689).
- 어떤 저장소도 clone하지 않았습니다. DreamX 코드가 없어서 04는 코드 대조가 아니라 주장 검증입니다.
- **HF:** `nvidia/Cosmos-Predict2.5-2B`(NVIDIA Open Model License, 자동 승인 gated, bf16 EMA 4.12 GB, tokenizer 0.51 GB). 메타데이터만 확인했습니다.

## 5. 참고문헌 메모 (확인 안 한 부분 포함)

- VideoWorld 2가 Ren et al. 2026a와 2026b로 두 번 들어가 있습니다(같은 arXiv:2602.10102). 본문에서 Table 1은 2026a, §2와 §4.2는 2026b를 인용합니다.
- §3.4의 "같은 영상 백본으로 미래 영상 대신 latent action을 예측(Kim et al. 2026a)"에서 2026a는 참고문헌상 "Planning in 8 tokens"(CVPR 2026, 압축 tokenizer)입니다. 영상 백본을 정책으로 미세조정하는 논문은 2026b(Cosmos Policy)라서, 인용이 바뀌었을 가능성이 있습니다.
- CoMo, VPP, Genie, LAPO의 세부 내용은 이번에 원문을 다시 읽지 않았습니다. 계보의 한 줄 설명은 DreamX §2의 서술을 따랐습니다.
- LARYBench 원 논문(arXiv:2604.11689)의 탐침 프로토콜과 Table 1 기준선 값이 원 논문 값인지 DreamX 저자들이 다시 돌린 값인지는 확인하지 않았습니다. 부록 B.4는 "모든 표현 모델을 고정하고 공통 탐침을 학습"한다고만 적었습니다.

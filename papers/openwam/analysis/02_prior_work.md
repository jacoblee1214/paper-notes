# 02. 선행연구와 위치

## 1. 계보

```
영상 생성 사전학습                    로봇 정책                              "설계를 실험 과학으로"
Wan2.1/2.2 (Wan et al. 2025)          ACT, Diffusion Policy                  Cambrian-1, Beyond Language Modeling
Cosmos-Predict2.5 (Ali et al. 2025)   VLA: RT-2, OpenVLA, π0/π0.5            Towards Physics of Multimodal Pretraining
        │                                      │                              LBM, StarVLA-α, OpenHLM
        └──────────────┬───────────────────────┘                                      │
                World–Action Model (WAM): 영상 prior + 행동 생성                      │
   Single: Cosmos Policy, DreamZero   Dual: Fast-WAM, LingBot-VA   Tri: Motus         │
                       │                                                              │
                       └───────────────────── OpenWAM ────────────────────────────────┘
                         Infra(모듈 계약) + Study(통제 실험) + α(5B+1B, 518.5M 프레임)
```

저자들의 문제의식(§1, §2)은 WAM의 핵심 가설인 "세계 예측과 행동 생성의 시너지"가 단일 시스템 안에서 구조, 학습, 데이터, 샘플링과 얽혀 있어 검증되지 않았다는 것입니다. VLA 쪽에는 OpenVLA, StarVLA, XPolicyLab 같은 열린 생태계가 있는데 WAM 쪽에는 없다는 점도 듭니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | OpenWAM에서 바꾼 점 |
|---|---|---|
| 영상 DiT | Wan2.2-TI2V-5B (Wan et al. 2025) | 첫 프레임 clean 고정, 토큰별 timestep, 30층 모두 ActionDiT와 공유 주의 |
| 시각 인코더 | Wan2.2-VAE(48채널, 시간 4배·공간 16배 압축) | 고정. Study에서 FLUX.2-VAE, DINOv3, V-JEPA 2.1(+S-VAE)과 교체 비교 |
| S-VAE 어댑터 | Zhang et al. 2025a, Jha et al. 2026 | 표현형 특징을 48차원으로 축소 |
| 텍스트 인코더 | umT5-xxl (Wan 계열 기본) | 고정, 문맥 끝에 고유수용 상태 토큰 1개 추가 |
| ActionDiT, joint self-attention | Fast-WAM, LingBot-VA, Motus 계열의 MoT형 결합 | 폭 1024, 30층, 공유 공간 24 × 128 |
| 결합 flow matching, 독립 timestep | π0 계열 flow matching, Wan의 shift 5 timestep warp | 두 스트림 독립 표본, 유효 마스크 m |
| 비동기 denoising 경로 | Latent Forcing (Baade et al. 2026, arXiv:2602.11401)의 분산 이동 | 선형 지연을 추가하고 동기와 같은 코드 경로로 통합 |
| velocity 캐시 | DreamZero (Ye et al. 2026c)의 단계 간 재사용 | 두 스트림 cosine 유사도 기준, 연속 skip 상한 |
| 통합 행동 공간 | (자체) 80D = 34D 팔 × 2 + 예비 12 | 고정 슬롯 의미, 정규화 후 scatter |
| 데이터 정제 | Qwen-RobotManip (Yuan et al. 2026a) 정제 파이프라인 | 상태 기반 정지 구간, 시각 교차 확인 규칙 추가 |
| 평가 클라이언트 | 각 벤치마크 공식 코드, RoboDojo는 XPolicyLab | WebSocket 단일 규약 |
| 코드 기반 | DiffSynth-Studio 계열(주의 구현 환경변수 `DIFFSYNTH_ATTENTION_IMPLEMENTATION`) | 백본 계약과 조합 규칙으로 재구성 |

## 3. 이 저장소의 다른 논문과의 관계

| 논문 | 관계 |
|---|---|
| AW-0 (`agentic_wam/`) | 가장 가까운 비교 대상입니다. AW-0도 Cosmos-Predict2.5-2B 영상 전문가(VAE는 Wan2.2로 교체)와 행동 전문가를 MoT로 묶은 Dual-System WAM에 코드 전문가를 더한 Tri형 구조입니다. AW-0 분석에서 "4단계 추론에서 행동이 읽는 미래 영상은 σ ≥ 0.625인 흐린 상태"라고 지적했는데, OpenWAM Fig. 9의 행동 선행 o = 0.8(영상 σ ≥ 0.83에서 행동 완성, 92.3%)이 같은 현상을 통제 실험으로 보여 줍니다. 추론 때 미래 영상의 기여는 작고, 이득은 학습 때 생깁니다(`03` §2). AW-0의 "계획 스트림 + ViT" 효과 분해와 OpenWAM Table 1의 Tri vs Dual(+0.24%p, 유의하지 않음)도 같은 방향입니다. 추가 스트림의 이득은 작거나 통제되지 않았습니다. 두 논문 모두 RoboTwin 2.0으로 평가하지만 AW-0는 리더보드보다 약한 자체 재현 기준선을 썼고, OpenWAM은 공식 리더보드 수치를 그대로 옮겼습니다(Table 15–16). |
| Hydra-0 (`hydra0/`) | Hydra-0는 행동을 이미지 평면의 점 궤적으로 표현해 영상 모델의 조건으로 씁니다. 정책이 아니라 world model(시뮬레이터) 쪽입니다. OpenWAM의 80D 통합 행동 공간은 반대로 말단 좌표계의 고정 슬롯입니다. 둘 다 "embodiment 공통 행동 표현" 문제를 다르게 풉니다. Hydra-0 분석에서 지적한 "같은 백본 통제 비교 부재"를 OpenWAM은 Infra로 해결하려 했고, Table 1–2처럼 구조만 바꾼 비교를 실제로 냈습니다. |
| ThinkWorld (`thinkworld/`) | ThinkWorld의 이득 63%가 "1초 뒤 DINOv3 특징 예측" 보조 손실에서 나왔습니다. OpenWAM Fig. 7은 DINOv3 + S-VAE latent를 예측 대상으로 써도 Wan2.2-VAE와 거의 같다(90.18 vs 90.30, 차이 유의하지 않음)는 것을 보입니다. 두 결과를 합치면 "미래 예측 대상이 픽셀일 필요는 없다"는 결론이 강해집니다. LIBERO-Plus Avg가 과제 수 가중 평균이라는 점은 ThinkWorld 분석에서 쓴 가중치로도 재현됩니다(`04` C7). |
| Chronos Policy (`chronospolicy/`) | OpenWAM의 데이터 로더는 소스별 fps를 맞추지 않습니다. 33프레임 창과 32-step 행동 chunk가 30 fps에서는 1.1초, 15 fps(AgiBot)에서는 2.1초, 10 fps(DROID)에서는 3.2초입니다(`04` D9). Chronos Policy가 다룬 "물리 시각 대신 index로 보는 정책"의 문제가 다중 소스 사전학습에서 그대로 생깁니다. time-aware RoPE나 시각 기반 리샘플링을 넣을 자리가 분명합니다. |
| PaceVLA (`pacevla/`) | PaceVLA는 행동 chunk의 시간 해상도를 바꿔 실행 속도를 올립니다. OpenWAM의 비동기 추론 모드(지연 d, 실행 n, 버림)와 velocity 캐시는 추론 지연을 숨기는 다른 축입니다. OpenWAM-α 평가는 모두 동기 모드라서, 비동기 모드가 실제 성능에 미치는 영향은 Fig. 9(경로)와 별개로 측정되지 않았습니다. |

### "체계적 WAM 사전학습"이 위 논문들에 더하는 것

1. **같은 코드에서 바꾼 하나의 변수.** 위 다섯 논문은 각자 구조 하나를 제안하고 기준선과 비교했습니다. OpenWAM은 구조 7종, 마스크 4종, 인코더 6종, 백본 4종, 경로 17종, 사전학습 4종을 같은 trainer·서버·평가 규약에서 돌리고, **ablation 체크포인트 32개를 모두 공개**했습니다. 이 저장소의 다른 논문에서는 "통제 비교 부재"를 한계로 적었는데, 여기서는 GPU만 있으면 그 비교를 다시 평가할 수 있습니다.
2. **효과 크기의 서열.** 재계산하면 서열이 분명합니다. 사전학습의 OOD 이득(+9.3–12.1%p) > 구조 용량(Single → Dual +6.9%p) > 영상→행동 정보 경로(+5.0%p) > 사전학습된 DiT 초기화(+2.1%p) > 백본 크기(5B → 14B +1.4%p) ≫ 마스크 방향, 2단계 vs 공동학습, Tri vs Dual(잡음 범위)입니다(`03` §3). 다른 WAM 논문이 내세운 기여를 이 서열에 놓고 볼 수 있습니다.
3. **"추론 때 미래를 보고 행동한다"는 이야기에 대한 반증 자료.** 경로 실험(Fig. 9)은 WAM 계열 논문(AW-0, DreamZero의 계획-후-IDM 관점)이 암묵적으로 가정한 추론 때 정보 흐름을 직접 시험한 첫 통제 결과입니다.
4. **한계도 같은 곳에 있습니다.** 모든 통제 실험이 RoboTwin 한 벤치마크, seed 1개입니다. 그래서 위 서열의 위쪽 다섯 항목만 믿을 수 있고, 마스크·사전학습 순서에 관한 Finding은 이 설계로는 결론이 나지 않습니다.

## 4. 공개 저장소와 자산 (2026-11-04 확인)

### 4.1 GitHub

| 저장소 | 역할 | 라이선스 | ★ | 마지막 push | 확인한 커밋 |
|---|---|---|---|---|---|
| [OpenWAM-Official/OpenWAM](https://github.com/OpenWAM-Official/OpenWAM) | 공식 코드(Infra, 학습, 배포, 벤치마크 어댑터) | Apache-2.0 | 1,162 | 2026-11-02 | `a00c4ba` (shallow clone, 서브모듈 미초기화) |
| [XPolicyLab/XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) | 정책 평가·배포 생태계, RoboDojo 평가에 사용 | Apache-2.0 | 531 | 2026-11-03 | `6baccd5` (API로 HEAD만 확인, clone 안 함) |
| [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) | 영상 백본 TI2V-5B, Wan2.2-VAE | Apache-2.0 | 18,358 | 2026-09-21 | – |
| [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) | Study 백본 VACE-1.3B, I2V-14B | Apache-2.0 | 17,376 | 2026-03-05 | – |
| [nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) | Study 백본 Cosmos-Predict2.5-2B, 공식 저장소의 git 서브모듈 | Apache-2.0 | 1,392 | 2026-06-08 | – |
| [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) | Wan 계열 학습 코드 기반(추정, 환경변수 이름 기준) | Apache-2.0 | 13,340 | 2026-11-03 | – |
| [starVLA/starVLA](https://github.com/starVLA/starVLA) | 비교 VLA, 모듈형 VLA 플랫폼 | NOASSERTION | 3,904 | 2026-10-31 | – |
| [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM) | 비교 WAM(Dual), 표현 학습 가설 | NOASSERTION | 1,641 | 2026-08-20 | – |
| [Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va) | 비교 WAM(Dual), 실물 비교 대상 | Apache-2.0 | 1,973 | 2026-07-09 | – |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0, π0.5 기준선 | Apache-2.0 | 14,526 | 2026-08-24 | – |
| [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) | Study 평가 벤치마크 | MIT | 3,038 | 2026-09-24 | – |

### 4.2 Hugging Face 모델 (`OpenWAM/`, 46개, 모두 Apache-2.0 태그, 메타데이터만 확인)

| 묶음 | 개수 | 내용 | 확인한 값 |
|---|---|---|---|
| OpenWAM-Alpha-Pretrain-Foundation-Model | 1 | 사전학습 체크포인트 | `checkpoint_step_154000.safetensors` 24.81 GB, sha `52df4e6`, 2026-09-09 |
| OpenWAM-Alpha-Sim-* | 8 | LIBERO, VLABench, RoboTwin-Full, RoboTwin-Clean2Random, RoboDojo, RoboCasa365, EBench, RoboCasa-GR1 | 파일 이름의 step이 Table 10과 모두 일치(10690, 6000, 118655, 10740, 60000, 60000, 100000, 100000) |
| OpenWAM-Alpha-Real-* | 5 | Franka, RoboDojo ARX-X5·Piper·Piper-X, Wuji 손 | step 9860, 30000 × 3, 10925 (Table 10과 일치) |
| Study ablation (`robotwin_*`, `pretrain_*`, `sft_*`) | 32 | 구조 7, 마스크 3(+기본), 백본 4, 인코더 6, 사전학습 5, 사전학습 후 SFT 7 | 파일 크기는 확인하지 않음(백본·인코더마다 다름). OpenWAM-Alpha 14개는 모두 24.81 GB(고정 umT5 포함) |

### 4.3 Hugging Face 데이터셋 (`OpenWAM/`, 9개, 받지 않고 크기만 확인)

| 데이터셋 | 크기 | HF 라이선스 태그 | 비고 |
|---|---|---|---|
| assets | 0.18 GB | MIT | 데모 영상 등 |
| LIBERO | 1.94 GB | Apache-2.0 | SFT용 변환본 |
| VLABench | 13.16 GB | MIT | |
| RoboCasa365 | 83.45 GB | 없음 | |
| RoboCasa_GR1 | 43.93 GB | 없음 | |
| wuji_teleop_data_subset | 1.86 GB | 없음 | 실물 손 데이터 일부 |
| AgiBotWorld-Beta | 4,004.96 GB | 없음 | 사전학습 소스(정제본). 원본 HF 데이터셋은 gated |
| InternData-A1 | 1,389.75 GB | 없음 | 사전학습 소스(정제본). 원본은 gated |
| Droid | 148.12 GB | 없음 | 사전학습 소스(정제본) |

- RoboCOIN은 `OpenWAM/` 아래에 없고 원 저장소(`wayrise/RoboCOIN`, 약 3,929 GB)를 씁니다.
- **1인칭 사람 영상(WorldEngine, 사전학습 프레임의 30%)은 공개되지 않았습니다.**
- 공개된 로봇 사전학습 데이터만 받아도 약 9.5 TB입니다. 라이선스 태그가 없는 정제본은 원 데이터셋 약관을 따로 확인해야 합니다.

## 5. 가까운 대안

- **Fast-WAM (Yuan et al. 2026b):** Dual-System WAM이고, "WAM의 이득은 추론 때 계획이 아니라 학습 때 표현에서 나온다"는 표현 학습 가설을 냈습니다. OpenWAM Fig. 9는 이 가설을 지지하는 자료로 인용됩니다. 사전학습이 없어 LIBERO-Plus(51.5)와 RoboTwin C2R randomized(1.9)에서 크게 무너집니다.
- **ABot-M0.5 (Chen et al. 2026a):** 같은 Wan2.2-VAE 픽셀 latent WAM이면서 LIBERO, RoboTwin-Full, RoboCasa365에서 OpenWAM-α보다 높습니다. 저자들은 단일 팔 데이터 규모 차이로 설명합니다(Fig. 15).
- **LingBot-VA / 2.0 (Li et al. 2026b, Zhang et al. 2026b):** Dual-System. RoboTwin-Full에서 2.0이 OpenWAM-α와 동률(93.60)이고, Franka 실물에서 OpenWAM-α와 차이가 유의하지 않습니다(99/120 vs 93/120).
- **Qwen-RobotManip (Yuan et al. 2026a):** VLA. C2R, LIBERO-Plus, RoboTwin-Full에서 OpenWAM-α보다 높고, 데이터 정제 파이프라인의 출처이기도 합니다.
- **DreamZero (Ye et al. 2026c), Cosmos Policy (Kim et al. 2026b):** Single-System WAM. Table 1의 Single 계열 약세(85.5, 84.6)는 이 계열의 구조적 한계를 시사하지만, 원 논문 구현과 같은 조건은 아닙니다.

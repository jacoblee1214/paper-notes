# 02. 선행연구와 계보

인용은 논문 참고문헌 기준입니다. 각 논문을 따로 열어 검증하지는 않았습니다. §4는 제 판단입니다.

## 1. 계보도

```
[고정 주기 모방학습]  Diffusion Policy, action chunking
        │  문제: 시퀀스 위치 = 시간이라는 가정. 지연·주기가 바뀌면 깨짐
        ├─▶ [배포 때 지연 보정]
        │     UMI 지연 맞춤 (Chi et al., 2024)
        │     비동기 chunk 실행 + prefix inpainting: RTC (Black et al., 2026), 학습 시 조건화 (Black et al., 2025)
        │     Bidirectional Decoding (Liu et al., 2025), masked chunking (Wang et al., 2026a), DiscreteRTC (2026b)
        │     실행 시점 상태 예측: VLASH (Tang et al., 2025)
        │     측정된 지연을 조건으로: Delay-Aware DP (Liao et al., 2025)  ← 가장 가까운 정책 쪽 선행연구
        ├─▶ [추론 자체를 빠르게]  Diffusion Forcing (Chen et al., 2024), Streaming DP (Høeg et al., 2024)
        └─▶ [다중 주기 감각]
              Reactive DP: 느린 영상 + 빠른 촉각 분리 (Xue et al., 2025), ImplicitRDP (Chen et al., 2026)
              ManipForce: 주기 인식 임베딩 (Lee et al., 2025), LAG-Fusion (He et al., 2026), FA-RDP (Zhuo et al., 2026)

[연속 시간 표현]  mTAN (2021), ContiFormer (2023), RoPE (Su et al., 2024)
        └─▶ 영상 생성: VideoRoPE (2025), VFRTok (2026), BulletTime: world-time RoPE (Wang et al., 2026c) ← 가장 가까운 표현 쪽 선행연구

★ Chronos Policy: 관측 캡처 시각 + 행동 실행 시각을 RoPE 좌표로, 학습 시 시간 무작위화
```

**차별점(저자 주장):** 기존 연구는 지연 하나(추론 지연)나 주기 하나(모달리티별 속도)를 따로 다뤘습니다. Chronos는 관측 주기, 측정 지연, 모달리티 간 시간차, 추론·실행 지연을 **하나의 인터페이스(타임스탬프)**로 다룹니다. 시간을 행동 속도나 지속 시간 제어에 쓰는 연구(Rudin et al., 2022; Jia & Chen, 2025; CTCO; TARC)와 달리, "언제 관측했고 언제 실행할지"를 표현합니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 |
|---|---|
| 정책 백본 | Diffusion Policy 계열 transformer, DDIM 샘플러 |
| 영상 인코더 | SigLIP2 base |
| 회전 위치 인코딩 | RoFormer/RoPE (Su et al., 2024) |
| AddTime 비교군 | Fourier features (Tancik et al., 2020) + MLP |
| Pivoting 행동 표현 | Adaptive Compliance Policy (Hou et al., 2025) |
| 실제 기준선 | RTC (Black et al., 2026)를 DDIM prefix inpainting으로 이식 |
| 표준 과제 | DexMimicGen 6과제(평가 프로토콜도 원 논문 기준) |
| 실제 로봇 데이터 | 저자 그룹의 UMI 계열 힘 센서 구성으로 보임(Choi et al., 2026 UMI-FT; Clark et al., 2026) |

## 3. 이 저장소의 다른 논문과의 관계

- **do_as_i_do:** Do as I Do의 SAM 3D guided blending은 마지막 스텝에서 EMA 필터와 같았고(`do_as_i_do/analysis/03_derivations.md`), 그 05의 A1은 "flow action head에 적응형 guided blending을 넣되 RTC를 확인한 뒤 신규성을 판단"이었습니다. Chronos는 RTC를 기준선으로 직접 비교하고 "비동기 실행과 시간 인식은 상보적"이라고 결론냅니다. 따라서 chunk 연속성 아이디어를 평가할 때는 **RTC와 Chronos식 타임스탬프를 둘 다 비교 대상으로** 두어야 합니다.
- **dexscrew:** DexScrew의 BC는 촉각과 관측 이력을 20 Hz 제어 스텝 단위로 넣고, 분석 결과 촉각보다 이력의 효과가 더 뚜렷했습니다(`dexscrew/analysis/01_paper_analysis.md`). 촉각은 보통 제어 주기보다 빠르게 들어오므로, Chronos의 Pivoting 설정처럼 "제어 격자에 맞추지 않고 캡처 시각과 함께 넣는" 대안을 시험해 볼 수 있습니다. 다만 Chronos에서도 다중 감각 쪽 개선은 시드 단위로 유의하지 않았습니다(`03` §5).
- **gpt6_astra_tactile:** 그 분석의 D2 가설은 느린 LLM 계층(약 0.05 Hz)이 전략을 정하고 빠른 정책(20 Hz 이상)이 촉각에 반응하는 구조이고, ablation에 느린 계층 지연 주입(0/5/20 s)이 있습니다. 느린 계층의 출력은 큰 지연을 두고 도착하므로, 그 출력에 "언제 만든 계획인지" 타임스탬프를 붙여 빠른 정책에 주는 것이 Chronos식 해법입니다.
- **hydra0:** Hydra-0은 이미지 평면 점 궤적(action flow)을 조건으로 영상을 생성합니다. Chronos가 가장 가깝다고 꼽은 BulletTime도 영상 생성에서 연속 시각을 RoPE로 넣습니다. world model과 정책이 같은 "물리 시각 좌표"를 공유하면, 서로 다른 프레임레이트의 데이터를 섞을 때 유리합니다.
- **flashdexretarget:** FlashDexRetarget은 K = 10개 미래 레퍼런스 프레임을 조건으로 씁니다. 사람 시연 데이터셋(TACO, OakInk2, HOT3D)은 프레임레이트가 서로 다릅니다. 레퍼런스 프레임을 인덱스가 아니라 물리 시각으로 주면 데이터셋 간 주기 차이를 흡수할 수 있습니다.

## 4. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 연구 | 관련성 |
|---|---|
| ACT의 temporal ensembling (Zhao et al., 2023) | 겹치는 chunk를 지수 가중으로 섞는 가장 단순한 chunk 연속성 기법입니다. RTC와 함께 비교 대상이 될 만합니다 |
| Neural ODE/CDE (Chen et al., 2018; Kidger et al., 2020), GRU-D (Che et al., 2018) | 불규칙 시계열을 연속 시간 모델로 다루는 고전적 접근입니다. 관측 간격 Δt를 모델 동역학에 직접 넣습니다 |
| 연속 시간 상태공간 모델(S4/S5/Mamba 계열) | 이산화 간격 Δt를 입력마다 바꿀 수 있어, 불규칙 샘플링을 구조적으로 처리합니다. RoPE와 다른 방식의 "시간 인식"입니다 |
| Time2Vec (Kazemi et al., 2019) | 학습 가능한 주기 특징으로 시간을 임베딩합니다. AddTime 비교군과 같은 계열입니다 |
| π0 계열의 비동기 추론 실무 | RTC의 출처(Physical Intelligence)입니다. 대형 VLA에서 추론 지연이 수백 ms에 이르므로, Chronos식 타임스탬프의 효용이 더 클 수 있습니다 |

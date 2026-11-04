# 01. 논문 분석: OpenWAM

- **논문:** Wang, Huang, Li, Zhang et al. *OpenWAM: An Open, Modular Exploration Towards Systematic World–Action Model Pretraining.* arXiv:2609.07398v1 (2026-09-07). NUS, 칭화대, 베이징대, 홍콩대, 저장대, 홍콩중문대, 상하이교통대. 프로젝트 리드: Yuran Wang, Siqiao Huang, 공동 지도: Lin Shao, Hang Zhao.
- **라이선스:** arXiv abs 페이지 기준 CC BY 4.0. PDF 안에 다른 권리 고지는 없습니다(43쪽 전체 텍스트 확인).
- **공개물:** 코드 `OpenWAM-Official/OpenWAM`(Apache-2.0), Hugging Face 모델 46개(사전학습 1, 벤치마크·실물 SFT 13, Study ablation 32), 데이터셋 9개. 자세한 표는 `02` §4에 있습니다.
- **한 줄 요약:** World–Action Model(WAM, 영상 생성 모델을 이어받아 미래 영상과 행동 chunk를 함께 생성하는 정책)의 설계 선택을 부품으로 쪼개 같은 조건에서 비교하고(OpenWAM-Infra, OpenWAM-Study), 그 결론을 모아 5B 영상 DiT + 1B ActionDiT 모델을 518.5M 프레임으로 사전학습(OpenWAM-α)한 연구입니다.

## 1. Abstract와 문제 설정

영상 생성 모델은 "세상이 어떻게 변하는가"를 배웠고, 로봇 궤적은 "무엇을 하면 되는가"를 줍니다. WAM은 앞의 것을 초기값으로 삼아 뒤의 것을 배웁니다(§1). 저자들은 기존 WAM 시스템이 백본, 시각 표현, 구조, 정보 흐름, 추론 방식, 데이터가 한 덩어리로 묶여 있어서, 무엇이 성능을 만들었는지 가를 수 없다고 봅니다(§1, §2). 그래서 세 질문을 세웁니다.

| 질문 | 다루는 절 | 답(Finding) |
|---|---|---|
| Q1. WAM은 어떤 세계 지식을 물려받아야 하나 | §4.1 | 충분히 큰 생성 백본과 압축되고 정보가 많은 latent (Finding 1) |
| Q2. 세계 학습과 행동 학습의 시너지는 어떻게 만드나 | §4.2 | 행동 전용 용량, 학습 때 영상→행동 정보 경로, 추론 때 동기 denoising (Finding 2) |
| Q3. 그 시너지를 여러 도메인으로 키우려면 | §4.3 | 사전학습은 주로 OOD를 늘린다. 사람 영상 + 로봇 데이터 1단계 공동학습, mutual 마스크 (Finding 3) |

기여는 세 층입니다. OpenWAM-Infra(모듈형 학습·배포·평가 코드), OpenWAM-Study(통제 실험), OpenWAM-α(사전학습 모델과 벤치마크 8개 + 실물 3종 평가)입니다.

## 2. Method

### 2.1 조합형 모델 C(E, S, M) (§3.1)

모델 하나는 세 부품의 조합입니다.

- **시각 인코더 E(항상 고정):** 재구성형 Wan2.2-VAE, FLUX.2-VAE와 표현형 DINOv3, V-JEPA 2.1. 표현형 특징(768–1024차원)은 S-VAE로 48차원으로 줄이는 선택지가 있습니다.
- **스트림 백본 S:** 영상 백본 W(Wan2.1-VACE-1.3B, Cosmos-Predict2.5-2B, Cosmos3-Edge-4B, Wan2.2-TI2V-5B, Wan2.1-I2V-14B), 행동 백본 A(별도 ActionDiT 또는 영상 시퀀스에 합류), 선택적 VLM 백본 U(Qwen3-VL). 모든 백본이 prepare → 층별 block step(주의 전·후 반쪽으로 나눌 수 있음) → finalize라는 같은 계약으로 실행됩니다. 그래서 스트림 사이의 주의를 백본 내부를 건드리지 않고 바꿀 수 있습니다.
- **가시성 마스크 M (Eq. 1):** 2×2 블록 [M_V←V, M_V←A; M_A←V, M_A←A]. 영상 내부는 first-frame causal(깨끗한 첫 프레임은 자기만 봄), 행동 내부는 양방향으로 고정하고, 교차 블록 두 개로 네 모드를 만듭니다: mutual, action-sees-video, video-sees-action, isolated.

구조는 세 계열 일곱 변형입니다(Fig. 2 아래 줄, Table 1). Single-System(Vanilla, 행동 토큰만 다른 FFN으로 보내는 MoE), Dual-System(joint self-attention, joint cross-attention과 그 gradient 차단판, 영상을 먼저 만들고 행동을 역동역학으로 푸는 IDM), Tri-System(+Qwen3-VL, 읽기 전용 꼬리)입니다. 대표 시스템으로 Single은 Cosmos Policy·DreamZero, Dual은 Fast-WAM·LingBot-VA, Tri는 Motus를 듭니다.

### 2.2 학습: 결합 flow matching (§3.2, §5.1.2)

표본은 (지시문 ℓ, 영상 창 o₁:T, 행동 chunk a₁:H, 고유수용 상태 q, 차원별 유효 마스크 m)입니다. 논문 표기로 t = 0이 순수 noise, t = 1이 깨끗한 데이터이고, 두 스트림이 **각자 독립으로 뽑은** t_v, t_a로 noise를 받습니다.

L = λ_v E[w(t_v) ‖v̂_z − (z − ε_v)‖²] + λ_a E[w(t_a) ‖m ⊙ (v̂_a − (a − ε_a))‖²]   (Eq. 2)

깨끗한 조건 프레임은 영상 손실에서 빠지고, m이 각 embodiment가 실제로 채운 행동 좌표만 남깁니다. 저자들은 t_v, t_a를 독립으로 뽑으므로 학습이 (t_v, t_a) 평면 전체를 덮고, 따라서 어떤 추론 경로도 분포 안이라고 주장합니다(§3.2). OpenWAM-α는 λ_v = λ_a = 1, 중간 noise에서 최대인 종 모양 w(·), t = 1 − f_ρ(u)(u 균등, ρ = 5로 고noise 쪽에 치우침)를 씁니다(§5.1.2). 실제 식과 학습 분포의 모양은 `03` §1, 코드 대조는 `04` D4–D6에 있습니다.

### 2.3 배포: 추론 모드, denoising 경로, 가속 (§3.3)

- **추론 모드:** 동기(버퍼가 비면 멈추고 새 chunk를 추론)와 비동기(남은 행동이 d개일 때 다음 chunk를 미리 추론하고, 앞 d개는 건너뛰고 n개를 실행, 나머지는 버림).
- **denoising 경로 (Eq. 3):** 결합 noise 평면 위의 경로 τ = {(t_vⁱ, t_aⁱ)}. 동기는 대각선입니다. 비동기는 분산 이동 f_α(s) = αs / (1 + (α − 1)s)로 선행 스트림을 먼저 깨끗하게 하거나, 선형 지연 h_o(s) = max{(s − o)/(1 − o), 0}로 후행 스트림을 o까지 순수 noise에 둡니다. (α, o) = (1, 0)이 동기와 정확히 같습니다.
- **가속:** prompt 임베딩 캐시, 영상 decode 생략, torch.compile + CUDA graph, DiT velocity 캐시(최근 두 스트림 velocity의 cosine 유사도가 임계값 이상이면 다음 forward를 건너뛰고 캐시 값으로 적분). RTX 5090에서 Dual JSA 175 ms, 기준 대비 2.45배 빠릅니다(Fig. 4).

### 2.4 평가 규약과 80차원 통합 행동 공간 (§3.4)

벤치마크는 WebSocket 클라이언트로 정책 서버에 붙고, 관측(카메라 최대 3개, 지시문, 상태)을 보내 행동 한 개를 받습니다. 정규화, 카메라 배치, 역정규화는 서버가 체크포인트 기록대로 처리합니다. 실물 로봇도 같은 인터페이스를 씁니다. 여러 embodiment를 한 모델로 학습할 때는 **80차원 통합 행동 공간**을 씁니다. 팔 블록 34차원 × 2(말단 위치 3 + 6D 회전 6 + 그리퍼 1 + 손 24)와 예비 12칸이고, 각 데이터셋이 index map π로 흩어 넣습니다(Eq. 4). 채워지지 않은 칸은 m으로 손실에서 빠집니다.

## 3. Experiments

### 3.1 OpenWAM-Study (§4): RoboTwin 2.0에서의 통제 실험

평가는 RoboTwin 2.0 두 설정입니다. **Full**(clean 2,500 + randomized 25,000 시연으로 학습, 두 조건 평가)과 **Clean2Random**(clean만 학습, randomized가 OOD)입니다.

| 실험 | 결과 | 출처 |
|---|---|---|
| 영상 백본 크기 | 1.3B 90.14 → 2B 91.64 → 5B 92.39 → 14B 93.79. 5B를 기본으로 채택 | Fig. 6 |
| 시각 인코더 (DiT 무작위 초기화) | DINOv3 76.42, V-JEPA 80.91, FLUX.2 84.26, V-JEPA+S-VAE 88.46, DINOv3+S-VAE 90.18, Wan2.2-VAE 90.30 | Fig. 7 |
| 구조 | Single Vanilla 85.50, MoE 84.63, Dual JSA 92.36, JCA 88.25, Detached CA 91.85, IDM 87.95, Tri JSA 92.60 | Table 1 |
| 학습 때 마스크 | Isolated 87.41, Video Sees Action 87.63, Action Sees Video 92.39, Mutual 92.15 | Table 2 |
| 추론 때 경로 | 동기 93.0. 비동기 16개는 88.5–92.3으로 모두 낮음 | Fig. 9 |
| 사전학습 (600시간 예산) | OOD: scratch 14.50 → robot only 23.80 → 2단계 26.50 → 공동학습 26.62. ID는 87.00–88.50 | Fig. 10 |
| 사전학습 후 마스크 | Mutual − ASV: scratch Full −0.24, 사전학습 Full +0.16, C2R ID +0.70, OOD +0.72 | Fig. 11 |

저자들의 결론(Table 3)은 Wan2.2-TI2V-5B + 압축 latent, Dual joint self-attention + 동기 denoising, 사람 영상 + 로봇 1단계 공동학습 + mutual 마스크입니다.

해석 (재계산은 `03` §2–3, `04` C5–C6):
- 큰 효과는 확실합니다. Single → Dual(+6.9%p), 영상→행동 경로 유무(+5.0%p), 사전학습의 OOD 이득(+9.3–12.1%p)은 이항 표본 오차의 10배가 넘습니다.
- 그 밖의 차이는 대부분 잡음 범위입니다. RoboTwin 평가는 과제 50개 × 100 episode, seed 0 한 번이라 평균의 SE가 약 0.27%p, 두 실행 차이의 SE가 약 0.37%p입니다. Tri vs Dual(+0.24), Mutual vs ASV(Fig. 11의 +0.16 / +0.70 / +0.72), 공동학습 vs 2단계(+0.12)는 p > 0.25입니다. 그런데도 Finding 3은 "mutual이 일관되게 낫다"고 씁니다.
- 같은 설정으로 보이는 실행 사이의 차이가 이 효과들보다 큽니다. Dual JSA는 Table 1에서 92.36, Table 2 mutual 92.15, Table 2 ASV 92.39, Fig. 9 동기(mutual, 독립 noise) 93.0입니다.

### 3.2 OpenWAM-α (§5)

- **구조 (§5.1.1, Fig. 12a):** 고정 Wan2.2-VAE + Wan2.2-TI2V-5B DiT(폭 3072, 30층) + 1B ActionDiT(폭 1024). 30층 모두 bridge이고, 공유 주의 공간은 24 head × 128차원, mutual 마스크입니다. 첫 latent 프레임은 현재 관측이고 t_v = 1로 고정됩니다. umT5 4096차원 문맥에 q를 토큰 하나로 붙여 두 스트림이 각자 cross-attention으로 읽습니다.
- **학습 (§5.1.2, Table 9):** H200 128장 × 약 7일, global batch 3,072, 155,862 iteration(1 epoch), lr 1e-4 cosine(warmup 5%), AdamW (0.9, 0.95), ZeRO-2, 384 × 320, 33프레임 창(영상 stride 4), 흐름 shift 5/5.
- **데이터 (§5.2, Table 4):** 원 데이터 1.33B 프레임(14,348시간) → 정제·표본추출 후 518.5M 프레임(6,369시간). 자체 1인칭 사람 영상 30.1%(공개 안 됨), AgiBotWorld-Beta 18.6%, RoboCOIN 14.3%, DROID 7.0%, 합성 InternData-A1 30.0%. 사람 영상은 행동이 없어 영상 스트림만 학습시킵니다. 정제는 시각 품질 검사와 로봇 신호 4단계(신호 무결성, 상태 기반 정지 구간 자르기, 시각 교차 확인, 에피소드 삭제)입니다(§5.2.2).
- **배포 (§5.1.3):** 동기 경로, N = 10, 동기 추론 모드, RTX 5090에서 chunk당 약 170 ms입니다. 저자들은 동기 모드를 쓰므로 "결과가 모델 자체의 성능을 가장 직접 반영한다"고 씁니다. 하지만 velocity 캐시(근사)를 켠 벤치마크가 있습니다(`04` D3).

### 3.3 시뮬레이션 벤치마크 8개 (§5.3, Table 5, Table 13–20, Fig. 13–16)

| 벤치마크 | OpenWAM-α | 표의 1위 | 순위 |
|---|---|---|---|
| LIBERO (Table 13) | 99.3 | ABot-M0.5 99.4 | 2/18 |
| VLABench SR (Table 14) | 58.9 | Xiaomi-Robotics-1 59.1 | 2/6 (IS 기준 5/6) |
| RoboTwin2.0-Full (Table 16) | 93.60 | ABot-M0.5 94.10 | 4/14 (LingBot-VA 2.0과 동률) |
| RoboTwin2.0-Clean2Random (Table 15) | 69.0 | Qwen-RobotManip 77.1 | 2/13, WAM 중 1위(다음 4D-WAM 61.7) |
| RoboDojo SR (Table 17) | 11.92 | DM0.5 19.34 | 4/13, WAM 중 1위(다음 X-WAM 3.83) |
| EBench SR / Score (Table 18) | 49.4 / 64.7 | (1위) | 1/6 |
| RoboCasa365 (Table 19) | 38.2 | Xiaomi-Robotics-1 57.4 | 3/10 |
| RoboCasa-GR1 (Table 20) | 60.5 | PhysBrain 1.0 64.5 | 2/15 |
| LIBERO-Plus (Table 5) | 69.2 | Qwen-RobotManip 89.0 | 11/13 |

- 저자들은 LIBERO, VLABench, RoboTwin-Full, RoboCasa365, RoboCasa-GR1에서 "1위와 근소한 차이의 최상위권"이라고 씁니다(§5.3.1). RoboCasa365는 1위와 19.2%p 차이라서 이 서술에 맞지 않습니다(`04` C7).
- LIBERO-Plus의 낮은 점수는 카메라(33.8)와 noise(39.8) 교란에 몰려 있습니다. 저자들은 원인을 두 가지로 봅니다. 하나는 단일 팔 사전학습 데이터 부족(Fig. 15)이고, 다른 하나는 픽셀 latent 미래 예측의 시각 교란 취약성입니다(Takeaway 1, 2). LIBERO-Plus는 LIBERO 체크포인트를 추가 미세조정 없이 평가했습니다(B.2).
- VLA 대 WAM(§5.3.2, Fig. 16): 분포 안에서는 WAM이 더 잘 맞추고 분포 밖에서는 VLA가 더 잘 일반화하며, 둘 다 데이터로 보완된다는 것이 Takeaway 3입니다. 통제된 비교는 StarVLA 대 Fast-WAM 한 쌍뿐이고, 나머지는 데이터·구조·학습이 모두 다른 모델끼리의 비교입니다.

### 3.4 실물 로봇 (§5.4, Table 6–8, 부록 C)

| 플랫폼 | 비교 | 결과 | 통계 (`04` C9) |
|---|---|---|---|
| Franka 단일 팔 6과제 × 20회 (Table 6) | π0.5, LingBot-VA | 99/120 vs 93/120 vs 66/120 | LingBot-VA와 차이 p = 0.42 (유의하지 않음), π0.5와는 p = 7×10⁻⁶ |
| RoboDojo-Real 3 embodiment × 6과제 × 10회 (Table 7) | VLA 5종 | Score 37.6 / SR 24.4 vs π0.5 22.9 / 12.8 | SR 44/180 vs 23/180, p = 0.006. Score 기준 18과제 중 13과제 최고 |
| Wuji 손 + Tianji 팔 4과제 (Table 8) | π0.5 | ID SR 27/40 vs 12/40, OOD 40/65 vs 19/65 | 합치면 p = 0.002, 0.0004. 과제별로는 8개 중 2개만 p < 0.05 |

Wuji 손의 9D 말단 자세 + 21자유도 손은 사전학습에 없던 embodiment입니다. Collect Shuttlecocks OOD는 점수 분모가 π0.5 54, OpenWAM-α 57로 달라서 시행 조건이 같지 않습니다.

## 4. Limitations

저자들이 적은 한계(부록 A)는 사후학습(post-training)을 다루지 않았다는 점, 융합 방식이 교차 주의와 hard-routed MoE에 머물렀다는 점, UMI식 데이터가 빠졌다는 점, Wan2.2-VAE가 시점·잡음·장면 변화에 약하다는 점입니다.

비판적으로 읽으면 다음이 더 중요합니다.
1. **"체계적"의 통계적 기반이 얇습니다.** 통제 실험은 모두 RoboTwin 2.0 한 벤치마크, 학습 seed 1개, 평가 seed 1개입니다. Finding 2의 "동기가 최선"(최선 비동기 대비 +0.7%p, p = 0.06)과 Finding 3의 "mutual이 일관되게 낫다"(p 0.29–0.67)는 이 설계로는 가를 수 없습니다(`03` §3).
2. **추론 때 미래 영상의 기여가 작습니다.** 행동 선행 선형 지연 o = 0.8은 행동을 거의 순수 noise인 영상(σ_v ≥ 0.83)과 함께 끝내는데도 92.3%로, 동기 93.0%와 0.7%p 차이입니다. 추론 때 생성된 미래가 거의 쓰이지 않는다는 뜻이고, 저자들이 지지한다고 쓴 "표현 학습 가설"과 맞습니다. 반면 "추론 때 미래 예측이 행동을 이끈다"는 WAM의 일반적 홍보와는 다릅니다(`03` §2).
3. **사전학습 재현의 핵심 데이터가 비공개입니다.** 전체 프레임의 30%를 차지하는 1인칭 사람 영상(WorldEngine)과 그 로더가 공개되지 않았습니다. Study의 EgoDex 로더도 없습니다(`04` D1).
4. **평가 설정이 벤치마크마다 다릅니다.** RoboTwin은 velocity 캐시를 켰고 EBench는 껐습니다. 공식 README는 캐시를 끄면 RoboTwin 수치가 바뀐다고 적었습니다(`04` D3).
5. **Takeaway 3(VLA vs WAM)은 상관 관찰입니다.** 비교 모델 대부분이 데이터·구조·학습이 모두 달라서 패러다임 효과로 분리되지 않습니다.
6. **하드웨어 장벽이 큽니다.** 사전학습은 H200 약 21,500 GPU-h, 체크포인트 하나가 24.8 GB입니다(`04` §4).

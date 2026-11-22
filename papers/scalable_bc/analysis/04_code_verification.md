# 04. 공식 코드 대조와 구현 가능성 (ABC, arXiv:2606.27375)

대상은 공식 저장소 `amazon-far/abc`의 커밋 `d0832d12651d1b260a652861a14648dc5f3660c7`(2026-09-25, "Sync real-robot DAgger checkpoint release")입니다. 얕은 복제(`repo/`, gitignore)를 **정적으로만** 읽었습니다. 코드를 실행, 임포트, 설치하지 않았고 체크포인트, 데이터, 가중치를 받지 않았습니다. 논문 v1(2026-06-25)과 코드는 3개월 간격입니다(코드가 나중). 코드는 논문 이후에 바뀌었을 수 있으므로 아래 차이는 "논문 본문에 없는 설정" 또는 "논문과 다른 기본값"이고, 공개 체크포인트를 만든 설정과 같은지는 확인할 수 없습니다.

## 1. 저장소 구조

| 경로 | 줄 수(대략) | 내용 |
|---|---|---|
| `train.py`, `abc_minimal/train_loop.py`, `config.py` | 19 / 562 / 659 | 학습 진입점, 분산 학습(DDP, VLA는 FSDP2/HSDP 선택), 설정 데이터클래스(tyro CLI) |
| `abc_minimal/dit.py` | 1,071 | ABC-DiT: CLIP·DINOv3 비전 인코더, 풀링 교차 어텐션 DiT, flow matching 손실, Euler 샘플러(RTC 포함) |
| `abc_minimal/vla.py`, `gemma/` | 932 + 모델 | ABC-VLA: Gemma 3 4B + SigLIP, 어텐션 풀링, adaLN 확산 헤드, 다중 draw |
| `abc_minimal/dataloader.py`, `preprocess.py` | 556 / 185 | MP4 + 이진 에피소드 로더, 증강, 정규화 |
| `abc_minimal/policy.py`, `eval_policy.py`, `batched_eval.py` | 459 / 809 / 327 | 추론, 시뮬 평가(RTC), MJWarp 배치 평가 |
| `abc_sim/` | 파일 약 70개 | MuJoCo 장면, 과제 33개 명세, 무작위화, 평가기, 렌더링(Blender) |
| `deploy/` | 파일 약 60개 | 실로봇 배포, ZMQ 노드, GELLO, DAgger(`dagger.py`), 기록·내보내기, GELLO 조립 PDF |
| `prepare.py`, `scripts/` | 1,472 / 4개 | 체크포인트·데이터·에셋 다운로드, HF MCAP 변환 |

- 라이선스: 코드 Apache-2.0(`LICENSE`). DINOv3(Meta 라이선스)와 Gemma(Gemma Terms of Use) 가중치는 별도 약관이며, 공개 DiT 체크포인트는 DINOv3 파생 백본을, VLA 체크포인트는 Gemma 파생 가중치를 포함하므로 해당 약관이 적용된다고 README가 밝힙니다.
- 패키지 요구: Python ≥ 3.10(README는 3.12), `torch==2.11.0+cu128`(Linux), `mujoco~=3.8.0`, `mujoco-warp==3.8.0.3`, `warp-lang==1.13.0`, torchcodec(CPU), tyro, wandb 등. 배포용 선택 의존성(`i2rt` 포크 고정 커밋, mink 1.1.0, pyrealsense2)은 로봇 쪽입니다.

## 2. 논문과 코드가 일치한 항목

- **DiT 구조:** 32층, 은닉 1536, 헤드 24, MLP 비율 4, 청크 30, 상태·행동 14차원, 카메라 세 대(`DiTConfig`). 카메라별 학습 질의 12개(`vision_pool_num_queries`), DINOv3 ViT-B(은닉 768, 12층)를 카메라 간에 공유, 블록당 9-방향 adaLN 조절 + 교차 어텐션(`dit.py` `DiTBlock`). 헤드 파라미터를 코드 구조에서 세면 1.924B로 논문 1.93B와 0.3% 차이입니다(`03` V6).
- **VLA 구조:** 헤드 8층, 은닉 512, 헤드 8, 풀링 토큰 8개와 QK-Norm, 직접 상태 조건화(`VLADiTConfig`). 조건 벡터 차원이 512(시간) + 4,096(8 × 512) + 512(상태) = 5,120이고 MLP로 512로 줄입니다(`vla.py` `DiffusionActionHead`). 부록 B.2와 같습니다. 상태 MLP는 14 → 256 → 2,560(Gemma 은닉)입니다.
- **손실:** 보간 x_t = (1 - t)·행동 + t·잡음, 목표 속도 = 잡음 - 행동, 균등 t ∈ [0, 1), MSE(`dit.py` `forward`). 샘플러는 t = 1에서 0으로 Euler 10스텝입니다. 논문은 시간 분포와 부호 규약을 적지 않았습니다.
- **prefix 조건화:** 길이를 `randint(0, max_action_prefix)`로 뽑고(상한 제외, 기본 8이라 0–7) prefix 위치의 시간을 0으로 하며 손실에서 prefix 위치를 뺍니다. 부록 H.4("uniformly sampling from 0 to 7")와 같습니다. 추론에서 prefix 위치는 시간 0, 나머지는 현재 시간의 adaLN 조절을 받습니다(`sample_actions_rtc`).
- **최적화:** AdamW, 학습률 1e-4, 1,000 스텝 선형 워밍업 뒤 일정(`LambdaLR`), 가중치 감쇠 0.01, 기울기 노름 10, 고유감각 드롭아웃 0.1(`mask_state_ratio`). 모두 B.3과 일치합니다. 이미지는 224 × 224 레터박스 패딩(`resize_with_pad`), 행동·상태는 z-점수 정규화입니다.
- **VLA 학습:** 마지막 Gemma 층 특징(`feature_layer = -1`), 백본 전체 학습(`train_backbone`, `train_siglip`), bf16 자동 혼합 정밀도(파라미터 FP32), 다중 draw는 특징·행동을 `repeat_interleave`로 복제해 헤드를 k번 돌립니다(§3.2의 분산 감소 구현 그대로). 조종자 ID와 하위 과제 조건화는 라벨 드롭아웃 0.2(`PromptConfig`)이고 조종자 번호는 해당 과제 시간이 많은 순서로 0부터 매깁니다(논문 H.1의 "Op-0 = 최다 조종자"와 일치).
- **시뮬 평가:** 청크의 앞 15개 행동을 실행하고 RTC 방식으로 prefix 4를 줍니다(`execute_chunk_dim`, `rtc_prefix_length`). 논문 §3의 "비동기 실행과 행동 prefix"와 같은 방향입니다.
- **README 수치 정합:** 공식 DiT 체크포인트 약 8.1 GB는 2.016B × 4바이트와 맞습니다. 상자 접기 미세조정 기본 설정(배치 90 × 8 GPU = 720, 75k 스텝)은 §5.3의 "75k 스텝, 배치 720"과 같습니다.

## 3. 차이 D1–D18

중요도는 재현 결과에 미치는 영향 추정입니다(높음/중간/낮음).

| 번호 | 항목 | 논문 | 코드 | 영향 |
|---|---|---|---|---|
| D1 | 비전 인코더 학습률 배율 | §3.1, B.3: 비전 인코더 학습률 배율 **0.1** | `OptimConfig.vision_lr_scale = 1.0`, VLA의 `backbone_lr_scale = 1.0`(`config.py`, `_build_optimizer`). 기본 학습은 비전 인코더를 헤드와 같은 1e-4로 갱신 | 높음. 공개 체크포인트가 0.1로 학습되었는지 코드로 알 수 없음. 코드 주석은 bf16 파라미터에서 배율 0.1이 Gemma를 "93% 얼렸다"고 적음 |
| D2 | VLA 기울기 값 상한 | B.3: 기울기 노름 10 **그리고 기울기 값 최대 100** | `clip_grad_norm_(…, 10)`만 있고 값 클리핑이 코드 전체에 없음 | 낮음–중간 |
| D3 | 다중 draw 수 | 표 2, §3.2: VLA는 샘플당 **8** draw | 기본 `num_diffusion_draws = 1`, README의 VLA 예시 명령은 `--flow.num-diffusion-draws 4` | 중간. 논문 설정을 쓰려면 8을 명시해야 함 |
| D4 | prefix 잡음 | 논문에 없음 | README는 "prefix conditioning max 8 **with noise 0.05**"라고 쓰지만 `FlowConfig.prefix_noise_scale = 0.0`이고 학습 명령에도 없음 | 중간. README와 코드가 서로 다름 |
| D5 | 이미지 증강 | 서술 없음 | 상단 카메라에 회전 ±5°와 5% 무작위 크롭, 모든 카메라에 밝기 0.7–1.3, 대비 0.6–1.4, 채도 0.5–1.5(`augment_and_normalize`) | 높음. 기하 증강이 상단 카메라에만 있고 손목 카메라는 색 증강만 |
| D6 | 상태 마스킹 | "고유감각 드롭아웃 0.1" | 확률 0.1로 **정규화된 상태를 0 벡터로 치환**(z-점수 0은 평균 상태이지 별도 토큰이 아님). 마스킹된 샘플은 prefix 조건화도 끔(`apply_prefix & ~state_is_masked`) | 중간 |
| D7 | 최적화 세부 | betas, ε, 감쇠 적용 범위 없음 | AdamW(0.9, **0.95**), ε = 1e-8, 모든 파라미터 그룹에 같은 감쇠 0.01(norm·bias 포함), DiT는 `foreach`, VLA는 `fused` | 낮음 |
| D8 | 데이터 샘플링과 끝부분 | 서술 없음 | 에피소드를 건너뛰며 프레임을 균등 추출(길이에 비례), 청크가 끝을 넘지 않도록 끝 29프레임은 시작점에서 제외(`usable = length − (chunk − 1)`), 에폭 개념 없이 복원 추출 | 낮음–중간. 4.8에폭 환산(`03` V7)은 이 방식에서 "평균 4.8회 노출" |
| D9 | 표 1의 비교 변형 | CLIP-adaLN, 교차 어텐션 VLA, FAST + 교차 어텐션 VLA 포함 6종 | 코드의 DiT는 `dinov3`/`clip` 비전 인코더 + 교차 어텐션만, VLA는 풀링 adaLN만. README: "FAST tokens and the alternative conditioners are not included" | 높음. **표 1의 6종 중 3종은 공개 코드로 재현할 수 없음** (대비 기준선이 빠짐) |
| D10 | 공개 사전학습 체크포인트의 학습 데이터 | §5.1: "ABC-DiT·ABC-VLA를 ABC-130K에서 200k 스텝" | README(abc_sim): `abc_dit_xl_200k_model.pt`는 "3.5k시간 실제 혼합 + **98시간 시뮬(5과제)**"으로 학습. VLA 부모는 `abc130k` 계열과 "xdof만 학습한 200k 계열" 두 가지(`vla_abc130k_200000_v2.pt` 등) | 중간. §5.1의 순수 ABC-130K 설정과 공개 가중치가 다를 수 있음 |
| D11 | 혼합 비율과 사전학습 설정 | 초록·§1의 "co-training recipe"는 본문에 비율 없음 | 프리셋은 `bottles`(실제 0.8172 : 시뮬 0.1828), `tshirt`, `sim_task` 3개뿐. 195 과제 전체 사전학습 혼합을 만드는 프리셋·스크립트는 없음(`--mixture`로 직접 구성) | 높음. 3.5K 시간 사전학습 재현에 필요한 혼합 가중치가 공개되지 않음 |
| D12 | DAgger 혼합 80:10:10 | §5.3: 이전 80%, 현재 개입 10%, 현재 나머지 10% | 학습기에 이 비율을 만드는 기능이 없음. `export.py`는 DAgger 기록에서 **개입 구간만** 내보내는 것이 기본이라 "나머지 구간" 구성은 사용자가 직접. 공개 DAgger 체크포인트 5종은 5k–75k 스텝, 둘은 팔 강성 kp 2배로 학습 | 중간 |
| D13 | 추론 설정 | §3: 10 확산 스텝(표 1 지연), 30 Hz, 비동기 prefix | 공개 DAgger 체크포인트는 **5** 확산 스텝, RTC prefix 4, lead 4, 청크 20 실행. 배포 예시는 10 스텝, prefix 4, lead 7, 실행 16. 시뮬은 10 스텝, prefix 4, lead 4, 실행 15 | 중간. 지연과 성능의 논문 수치와 설정이 다름 |
| D14 | 시뮬 평가 프로토콜과 엔진 민감도 | §4: 과제당 50회, 시뮬 strict/progress | 기본 `num_worlds = 5`(문서는 "인용할 수치는 50 이상" 권고), README 표는 20 worlds × 2 시드. **같은 장면·정책에서 MJWarp 배치 0.71–0.77 대 CPU MuJoCo 0.54**, MuJoCo 3.6 → 3.8에서 dishrack 15/20 → 12/20(Fisher p = 0.50, 시드 변동과 구분 안 됨, `03` V16). "엔진 안에서만 비교하라"고 README가 권고 | 높음. 시뮬-실제 상관(C9)이 엔진·버전에 따라 달라질 수 있음 |
| D15 | 시뮬 과제 수 | §4는 10개를 나열, 그림 2는 20개 | `task_specs.py`에 33개 명세(변형 포함). 4개(마커 서랍 3개, 나무 블록 탑)는 채점기가 없음 | 낮음. 서술 불일치(`03` V14) |
| D16 | 시간 제한과 progress 정의 | 표 3: 과제별 120–390 s, 컵 뒤집기 없음 | 모든 과제 공통 236 청크 × 15 = 3,540 스텝 = 118 s(30 Hz), lego는 400청크. strict = "모든 물체를 한 시점에 동시에 올바르게 놓음", progress(`mean_max_progress`) = 시간에 따른 **최대** 진행 분수(되돌려져도 감점 없음) | 중간. 실제 평가(최종 상태 기준 점수, 시간 제한 최대 180 s)와 정의가 달라 상관의 의미가 바뀜 |
| D17 | VLA 토큰 구성 | 서술 없음 | SigLIP 224 → 이미지당 **64**토큰(풀링), `fixed_seq_len = 256`으로 고정, 상태 토큰과 `<action_start>` 포함, 토큰 임베딩은 학습하지 않음(`train_token_embedding = False`) | 낮음. 표 2 FLOPs와는 일관(`03` V6) |
| D18 | 체크포인트 로딩과 약관 | - | `torch.load(..., weights_only=False)`로 다운로드한 체크포인트(pickle)를 읽음(`checkpointing.py`, `dit.py`). 내려받은 가중치가 신뢰할 출처가 아니면 임의 코드 실행 위험. DINOv3, Gemma 가중치는 약관 동의 후 별도 다운로드 | 안전 주의 |

- D1은 가장 눈에 띄는 차이입니다. 비전 인코더 배율은 논문이 명시한 하이퍼파라미터이고 코드 기본값이 그것을 따르지 않습니다. 코드 README는 "학습 기본값이 production reference finetune과 같다"고 쓰므로(README Training 절), **미세조정 설정과 사전학습 설정이 다르고 사전학습 설정은 코드에 없다**고 읽는 것이 맞습니다.
- D9와 D11은 "공개 코드로 논문 결과를 재현하는 데" 가장 크게 영향합니다. 표 1이 모델 선택의 유일한 근거인데, 비교 변형의 절반과 사전학습 혼합이 코드에 없습니다.

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**논문의 학습과 실로봇 결과는 이 맥에서 재현할 수 없습니다. 공개 체크포인트 추론의 시뮬 평가는 조건부로 가능하고, 학습 구조와 분산 감소 아이디어의 축소판은 구현할 수 있습니다(토이 완료).**

| 항목 | 판정 | 근거 |
|---|---|---|
| 사전학습(DiT 200k 스텝, 배치 9,216, H200 96대 / VLA 64대) | 불가 | 약 1,800–2,000 H200 GPU 시간(DiT, 추정), VLA 약 4,900 GPU 시간(추정, `03` V7, V13). 데이터 3,553 h(HF 저장소 크기 1 TB 이상), 24 GiB 메모리로 2B 모델 + AdamW(약 32 GB) 불가 |
| 단일 과제 미세조정(DiT 2B, 배치 90 × 2 GPU, 25k 스텝) | 불가 | 파라미터 8 GB + 기울기 8 GB + Adam 16 GB(FP32) = 약 32 GB가 24 GiB를 넘음. 공식 하드웨어는 80 GB급 GPU. 소형 DiT(S, 153M)로 줄이면 메모리는 맞지만 논문 결과가 아님 |
| 실로봇(YAM 두 대, GELLO, RealSense) | 불가 | 하드웨어. 하드웨어 BOM(GELLO 조립 PDF)은 저장소에 있음 |
| 공개 DiT 체크포인트(8.1 GB)로 시뮬 평가 | 조건부 | `--camera-backend mujoco`가 "CPU/macOS 대체"라고 문서화됨. 추론 FP32 CPU에서 청크당 약 1.3 TFLOPs(10 스텝 × 2 × 1.93B × 30 토큰 + 인코더)로, 실측 전 추정은 청크당 수 초(236 청크 × 월드 수). 장치 `auto`는 cuda 아니면 cpu(MPS 분기 없음). 새 환경(uv, mujoco-warp, warp), 8.1 GB 체크포인트와 시뮬 에셋 다운로드, 공식 코드 실행의 사용자 승인이 필요하고, `weights_only=False` 로딩(D18)에 주의 |
| 공개 데이터 보기(`viz_episode.py`, 샘플 130 MB) | 조건부 | torch 없이 동작한다고 문서화됨. 다운로드와 설치 승인 필요 |
| 구조 재구현(작은 DiT + 풀링 교차 어텐션 + flow matching 손실) | 가능 | 코드 구조와 손실이 단순해서 torch로 수백 줄. 이 맥에서 소규모 합성 데이터로 학습 가능. 노력 약 1주(추정) |
| 다중 draw 분산 감소의 축소판 | 가능, **완료** | `analysis/toy_multidraw.py`, 약 10초(`03` §4) |
| 표 재계산과 상관·검정 | 가능, **완료** | `analysis/recompute.py` |

- **노력 추정(측정 전):** 공개 체크포인트 시뮬 평가를 이 맥에서 한 월드 돌려 보는 것은 환경 구성과 다운로드를 포함해 1–2일. 소형 DiT 구조 재구현과 합성 과제 학습은 1주. 사전학습과 실로봇 재현은 이 장비로는 범위 밖입니다.
- **정직한 한계:** 정적 읽기만 했으므로 코드가 실제로 도는지, 공개 체크포인트가 README의 성능(병 과제 시뮬 성공률 0.7 부근)을 내는지는 확인하지 못했습니다. README 수치는 저자 보고입니다.

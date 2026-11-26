# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- **μ0 코드:** [Yoonkyo/mu0](https://github.com/Yoonkyo/mu0), 커밋 `963d5f13a94df31921d22887ff8611763b48178d`(2026-10-27, Apache-2.0). 얕은 복제를 `repo/`에 두고 **정적으로만 읽었습니다.** 실행, 가져오기, 설치는 하지 않았습니다. 복원은 `git clone https://github.com/Yoonkyo/mu0 repo && git -C repo checkout 963d5f13a94df31921d22887ff8611763b48178d`입니다.
- **TraceExtract 코드:** [Yoonkyo/TraceExtract](https://github.com/Yoonkyo/TraceExtract), 커밋 `c02eb8efefd988dda7f0046e38efb00762a84e08`(2026-10-27, Apache-2.0). 정적으로만 읽었습니다(§3).
- 읽은 파일: `src/lerobot/policies/mu0/{modeling_smolvla,smolvlm_with_expert,mu0_policy,configuration_smolvla}.py`, `policies/pi_gemma.py`, `datasets/{trace_dataset,bspline_basis,trace_delta_stats,trace_depth}.py`, `scripts/{lerobot_train_trace_mu0,lerobot_train_policy_mu0,lerobot_eval_policy_mu0,lerobot_predict_trace_mu0_image_only}.py`와 학습·평가 셸 스크립트, `docs/release/*.md`. TraceExtract는 `run_traceextract.sh`, `infer.py`의 이동 필터, `utils/dino*`, `caption/curate_captions.py`, `docs/long_video_pipeline.md`.
- 읽지 않은 것: 서브모듈(robosuite, robocasa, Depth-Anything-V2, Grounded-SAM-2), TraceExtract의 SpatialTrackerV2/VGGT/MoGe 복사본. 체크포인트 설정 파일(`final_ckpt/config.json`)은 받지 않았으므로 학습 때 설정과 코드 기본값의 일치는 확인하지 못했습니다.
- 라벨: D1–D19(논문과 코드의 차이 또는 논문에 없는 설정), 일치 확인 항목은 §2.1입니다.

## 1. 구성, 설치 요구, 라이선스

| 항목 | 내용 |
|---|---|
| 라이선스 | 두 저장소 모두 Apache-2.0(GitHub API). μ0 저장소의 `LICENSE`는 LeRobot 원본(Hugging Face 저작권 표기)을 그대로 둡니다 |
| 요구 사양 | Python ≥ 3.12, `torch >= 2.7, < 2.11`, `transformers == 5.3.0`, peft, accelerate. 학습·평가 문서는 CUDA 12.8 빌드를 전제하고 기본 장치가 `cuda`(`--device=cuda`)입니다 |
| 추론에서 CUDA가 필요한 것 | 합성 깊이·키포인트 모드의 GroundingDINO와 SAM 2 확장(CUDA 빌드). 데이터셋 키포인트·깊이 없음 모드(`--use_kp_from_dataset=true --use_depth_from=none`)는 학습 환경만으로 문서가 안내합니다 |
| TraceExtract | PyTorch 2.8.0(CUDA 12.8), `pointops2` CUDA 확장 컴파일(nvcc 필요), TAPIP3D 체크포인트 수동 다운로드, 캡션 단계는 OpenAI 또는 Gemini API 키 |
| 공개 범위 | 학습·평가·정책 학습 코드, 데이터 엔진 코드, 사전학습 체크포인트(5.62 GB), 평가 세트(4.51 GB), 샘플 학습 세트(DROID 20 에피소드, 0.75 GB), RoboCasa 정책 체크포인트(10.97 GB). **전체 사전학습 데이터(15.7 TB 원본, 처리 결과)와 기준선 학습 설정은 저장소와 Hugging Face 목록에서 찾지 못했습니다(2026-11-26 기준)** |

## 2. 논문 ↔ 코드 대조 (D 항목)

| 번호 | 항목 | 논문 | 공개 코드 | 비고 |
|---|---|---|---|---|
| D1 | 다운스트림 정책의 질의 점 | 사전학습은 의미 키포인트 질의(§2, §3.2). 정책 쪽은 그림 3의 "Fixed query Point" 라벨만 있고 본문에 설명 없음 | `mu0_policy.py`: `GRID_SIZE = 20`, 균일 20×20 = 400개 키포인트를 질의로 쓰고, 키포인트당 마지막 토큰만 취해 2×2 평균 풀링으로 100개 토큰으로 줄여 게이트 교차 주의에 넣음(`_reduce_expert`) | 서론이 고정 격자의 한계를 지적하는 대비 설정. 정책 평가(표 2, 그림 6)의 μ0는 격자 질의로 돌아갑니다 |
| D2 | 정책에서 이력·깊이 | B.4에 서술 없음. 깊이는 "선택" | `MASK_ALL_HISTORY = True`, 깊이 예측기 없음(`Mu0PolicyConfig` 독스트링: "no depth predictor") | 정책용 특징은 RGB + 언어 + 격자 질의에서 나옴. 표 1의 μ0와 같은 RGB 전용 설정 |
| D3 | 게이트 초기화 | B.4: g를 0으로 초기화하고 시그모이드를 통과시켜 "약한 주입 영역에서 시작" | `GATED_MIXER_GATE_INIT = 0.0`, `z + torch.sigmoid(self.gate) * ca_out` → σ(0) = 0.5 | 초기값이 0.5이고 E.2의 수렴값이 0.481–0.500이라 게이트가 초기값 근처에 머뭄(`03` V14) |
| D4 | 정책 학습 설정 | 표 6: 50,000 스텝, 배치 32, AdamW, 학습률 1e-4, 워밍업 1,000 | `train_policy_mu0_atomic8.sh`: 60,000 스텝, GPU당 8 × 4 GPU = 32, 학습률 1e-4, 워밍업 1,000은 일치. 코사인 감쇠(0까지), 가중치 감쇠 0.01, 기울기 절단 1.0, β = (0.9, 0.95)는 논문에 없음 | 스텝 수가 다름(60k 대 50k). 공개 체크포인트 이름도 `atomic8_60k_s42` |
| D5 | 정책 학습·평가 구성 | 과제당 시연 100개(총 800), 과제당 50 롤아웃(D.5) | 8과제 데이터를 한 정책으로 함께 학습(과제 이름과 데이터 경로 리스트, `--num_episodes=100`), 평가 스크립트는 `reset(seed = 42 + ep)`로 50개 환경 시드 사용, `n_action_steps = 8` | 과제별 정책인지 다과제 정책인지 논문에 명시가 없음(코드는 `MultiLeRobotDataset`으로 8과제를 합쳐 한 정책을 학습). 8과제 60k 체크포인트와 65과제 150k 체크포인트가 모두 공개되어 있고 논문은 어느 쪽인지 말하지 않음(문서 기본 평가는 8과제). 같은 시드 열이면 방법 사이 초기 장면이 짝지어질 수 있으나 기준선 평가 코드가 없어 확인 불가 |
| D6 | 기준선 코드 | 표 1(Track2Act, Hamster, 3DFlowAction, Dream2Flow, TraceGen), 표 2(DP, π0, π0.5, TraceGen + AE), 그림 6 | 저장소에는 μ0 정책과 GPT, Gemini 궤적 예측 스크립트만 있음. `use_trace=false`가 "w/o Trace"(VLM + AE) 경로. π0, π0.5, DP, TraceGen, VLM 외 궤적 기준선은 없음 | 비교의 절반은 코드로 재현할 수 없음. "freeze backbone, action expert만 학습" 설정이 π0 계열에 어떻게 적용됐는지도 저장소 밖 |
| D7 | 사전학습 최적화 | B.1: AdamW 1e-4, wd 1e-10, 클립 10, 2×10⁵ 스텝, 유효 배치 24(2 GPU, GPU당 6) | 학습 스크립트: 기본값 일치, β = (0.9, 0.95), **스케줄러 없음(상수 학습률, 워밍업 없음)**. 문서 레시피: 단일 GPU 배치 2(48 GB), 다중 GPU 4장 × 배치 4 = 16 | 유효 배치가 논문(24)과 문서(16)에서 다름. 학습률 스케줄 서술은 논문에 없음 |
| D8 | B-spline 기저 | "degree-3 B-spline, D = 10 제어점", "smoothness" (§3.2, B.2) | `bspline_basis.py`: 사전 계산 매듭 `[0×4, 1/3×3, 2/3×3, 1×4]` = 내부 매듭 중복도 3인 3구간 Bézier(C0 연결). 독스트링도 "multiplicity-3-internal-knot Bezier-like cubics" | 구간 안에서는 3차이지만 두 내부 매듭에서 속도 연속이 보장되지 않음. 토이로 크기를 쟀음(`03` §3) |
| D9 | 손실 세부 | 식 9: Var_d(‖P̂_{n,d} − P̂_{n',d}‖²₂)(제곱 거리), 쌍 R 전체 평균. 식 8: Σ_t BCE / N. λ 값 없음 | 강성: `diff.norm(dim=-1)`의 분산(제곱하지 않은 거리), 군집 안 쌍 평균 후 군집 사이 평균. 유효성: BCE를 N × H 전체 원소로 평균(Σ_t 후 N 나눔이 아님). `rigidity_regularization_weight = 5`, `done_loss_weight = 0.5`(레시피) | λ_rig = 5, λ_done = 0.5는 본문에 없음. 식 8과 코드는 H배 규모 차이(논문 식이 32배 큼) |
| D10 | 흐름 시간 분포와 추론 스텝 | τ ∈ [0, 1], 추론 4스텝 오일러(식 6, B.3) | `sample_time`: Beta(1.5, 1.0) × 0.999 + 0.001(균등이 아님). 학습 스크립트 기본 `num_inference_steps = 10`, 문서 레시피는 `--num_inference_steps=4`. 정책은 상수 `MOTION_TOTAL_SOLVER_STEPS = 4`, 실행 스텝 1 | 시간 분포는 논문에 없음. 평가 스텝은 체크포인트 설정 확인 필요 |
| D11 | 학습 표본의 키포인트 선택 | N을 [1, 256]에서 균등 추출(B.1). 유효 미래 스텝이 D개 미만이면 "흐름 손실에서 뺀다"(B.2) | `trace_dataset.py`: 이동 키포인트(`is_moving`)만 후보, 현재 (u, v)가 이미지 안이고 유효, 유효 미래 스텝 ≥ D인 키포인트만 남겨 토큰으로 만듦(손실만이 아니라 입력 집합에서 제외). 후보가 없으면 슬롯을 사전 필터로 제외 | 정적 키포인트와 곧 끝나는 궤적은 학습에 들어가지 않음. 추론 질의도 이동 필터로 고르므로(D.2) 평가와 맞음 |
| D12 | 평가 프로토콜 | D.1, D.2: 129 슬롯, 슬롯당 16 키포인트, top-1/top-5, 지평 8/16/32 | 대표 명령: `--first_frame_only=true --first_frame_offset=[4,8,12] --n_min=16 --n_max=16 --use_kp_from_dataset=true --use_depth_from=none --no_provide_history=true --num_samples_for_metrics=5`. top-1은 표본 0, top-5는 5개 최소값. VLM 기준선은 u, v만 묻고 z는 정답 z로 채움(`pred_z = gt_z`), 문서 예시는 추론 노력 low | 표 1의 μ0는 깊이·이력 없음 설정(표 9의 "Full"과 일치). `first_frame_offset`이 영상당 최대 3슬롯을 만들 수 있어 129 슬롯이 영상 수와 어떻게 대응하는지는 평가 세트를 받지 않아 확인 불가 |
| D13 | 깊이 경로 | B.1: 별도 패치 스템, SigLIP 깊은 층 공유, 컬러맵 전 깊이 잡음 σ = 0.01 m | `smolvlm_with_expert.py`: 복제 스템 + **SigLIP 선형층 LoRA(r = 8, α = 16: q, k, v, out, fc1, fc2)**를 깊이 패스에만 켬. 깊이 정규화는 데이터셋 전역 로그 미터 범위(1–99 퍼센타일) | LoRA와 로그 정규화는 논문에 없음. RGB 경로는 동결 SigLIP과 같음 |
| D14 | 이력 드롭 | C.1: 전체 삭제 0.5, 키포인트별 0.3 | 두 조건을 합집합으로 적용: 키포인트당 이력이 사라질 확률 0.5 + 0.5 × 0.3 = 0.65 | 서술은 일치하지만 합성 확률이 본문에 없음(`03` V18) |
| D15 | 공개 데이터 범위 | 사전학습 15.7 TB (표 3) | 공개: 샘플 학습 세트 20 DROID 에피소드, 평가 세트 4.51 GB. 전체 처리 결과는 없음. 원 데이터셋 라이선스에 EgoDex CC-BY-NC-ND-4.0, Agibot CC BY-NC-SA 4.0 포함(표 3) | 사전학습 재현에는 원 데이터셋 접근과 1.2K GPU 시간 처리가 필요 |
| D16 | TraceExtract 프레임률 | A.7: 사람 5 fps, 로봇 10 fps | `run_traceextract.sh` 프리셋: EgoDex 10 fps, EgoVerse 10 fps, Agibot 5 fps, DROID 15 Hz에서 3프레임 간격(5 Hz) | 사람과 로봇이 반대. 프리셋은 예시 값일 수 있고 사전학습에 쓴 값은 확인 불가(`03` §1.5) |
| D17 | TraceExtract 키포인트 선택 | A.1: 청크당 예산 N, 개체별 할당량은 보이는 패치 면적에 비례, 작은 개체 최소 할당, 개체 마스크 안 최원점 샘플링 | `point_utils.py`: 개체(군집)마다 고정 목표 `num_points_per_entity`(기본 64, EgoDex 144)를 프레임별 패치 수에 비례해 프레임에 나눠 배정, 반올림 1 미만은 건너뜀(최소 할당 없음). 점은 군집 좌표의 K-means 중심(`cluster_coordinates`, `random_state = 42`) | 총 예산이 아니라 개체별 목표, 최원점 샘플링이 아니라 K-means 중심 |
| D18 | TraceExtract 캡션 | §2.3, A.5: VLM이 시작, 중간, 끝 프레임을 받음. 모델 이름과 길이 상수 없음 | `curate_captions.py`: 첫·끝 프레임 + 마스크가 기본이고 중간 프레임은 청크 길이 ≥ 30 프레임이고 지배 군집 이동 ≥ 5 px일 때만. 기본 모델 `gpt-4o-mini`(또는 `gemini-2.5-flash`), 청크 길이 25–200, 청크 최대 6, 평활 창 11 | 중간 프레임이 조건부. 캡션 품질은 모델 선택에 의존 |
| D19 | TraceExtract 추적 | A.3: 앵커에 SE(3) 정렬. A.4: 이후 조각은 전파만 서술 | `se3_utils.py`: 회전 평균(쿼터니언 고유분해) + 이동 평균, 2σ 이상치 제거(앵커 ≥ 4개일 때). `docs/long_video_pipeline.md`: 이후 조각에서 **이미 추적 중인 패치를 가린 뒤 새 키포인트를 더 생성** | 논문은 전파만 서술하고 새 키포인트 생성은 없음 |

### 2.1 일치로 확인한 항목

SmolVLM2-2.2B 앞 20층 + 폭 0.5배(1024) 20층 트레이스 전문가, 두 층마다 자기 주의(`self_attn_every_n_layers = 2`), 질의당 이력 토큰 1 + 미래 토큰 1(`trace_group_horizon = 8` = 이력 8, `ctrl_per_token = 10`), H = 32, D = 10, h = 8, λ_bsp = 0.2(1차 차분), 제어점 절단 1.5, 행 가중 능형 적합(무효 스텝 가중 0, 앵커 행 포함), 앵커 상대 + 축별 95퍼센타일 스케일, RoPE 위치를 접두 끝으로 고정, 영초기화 uv-MLP와 adaLN-Zero 머리, 키포인트 위치 DINO-base 특징의 concat + SiLU MLP 융합(식 5), 이력 드롭, 깊이 드롭 0.7, ColorJitter 0.3, 깊이 잡음 0.01 m, AdamW 1e-4/1e-10/클립 10, 2×10⁵ 스텝, N ∈ [1, 256], 키포인트별 H차원 유효성 머리(마스크 평균 풀링), 강성 손실의 "선형 경로 P̂ = P^τ − τ v" 복원, 4스텝 오일러, 정책의 4스텝 중 1스텝 부분 디노이징, 그리퍼 DINOv2 토큰, π0.5형 adaRMS Gemma 행동 전문가(폭 1024, 18층), 정책의 학습률 1e-4, 워밍업 1,000, 행동 차원 12, 청크 16, 실행 8, 배치 32, 과제당 100 시연·50 롤아웃, 이동 필터(λz = 0.1, τm = √((H/20)² + (W/20)²), `infer.py`), 사건 분할의 Savitzky–Golay 평활과 극값 분할. 또한 표 5의 행동 전문가 436.4M과 혼합기 20.6M은 코드 상수로 재계산해 일치합니다(`03` §1.8).

## 3. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 이 맥에서 논문 결과를 재현하는 것은 불가능합니다. 공개 체크포인트로 추론을 돌리는 것은 조건부로 가능하고, 축소 구현(작은 트레이스 모델)은 가능합니다.**

| 수준 | 판정 | 근거 |
|---|---|---|
| 사전학습 재현 (2.59B, 2×10⁵ 스텝, 13M 프레임) | 불가 | GPU 2장 이상, 전체 데이터(15.7 TB 원본)가 대용량이고 공개 목록에서 찾지 못함(D15). 2.59B에 동결된 부분을 빼도 활성화 메모리가 24 GiB를 넘을 가능성이 큼(미측정) |
| TraceExtract 전체 처리 (약 1.2K GPU 시간) | 불가 | CUDA 확장 `pointops2`, VGGT와 TAPIP3D, 캡션 API 필요. 데이터 크기 |
| 정책 학습·RoboCasa 평가 (표 2) | 불가 | 4 × L40S 가정(D.5), robosuite·robocasa와 자산 약 10 GB, 기준선 코드 없음(D6) |
| 실로봇 (그림 6) | 불가 | UR3 로봇이 필요하고 시연 데이터 220개는 공개 목록에서 찾지 못함 |
| 공개 체크포인트로 궤적 예측 추론·평가 | 조건부 | 가중치 5.62 GB(bf16이면 2.6–2.8B 파라미터 규모), 평가 세트 4.51 GB 다운로드, `transformers == 5.3.0`, peft 등 새 환경, 코드가 기본 장치를 `cuda`로 두므로 `--device`를 mps나 cpu로 바꿔야 하고 bf16 자동캐스트는 CUDA에서만 켜짐(`use_autocast ... device.type == "cuda"`). 설치 승인, 다운로드 승인, 코드 실행 승인이 필요하며 속도와 MPS 호환성은 측정 전 |
| 작은 트레이스 전문가 축소 구현 | 가능 | 합성 이동 점 영상(numpy/MuJoCo)에서 키포인트 질의 + B-spline 제어점 + flow matching을 작게 구현. torch 2.15, mujoco 4.0은 설치되어 있음 |
| B-spline 표현, 표 재계산, 통계 | 가능, 완료 | `analysis/toy_bspline.py`, `analysis/recompute.py` |

**노력 추정(측정 전):**
- 공개 체크포인트 추론 한 번 돌리기: 1–2일(환경 구성과 장치 수정 포함, 승인 후).
- 작은 트레이스 전문가 축소 구현(8–16개 질의, 2D 합성 영상, 수백만 파라미터): 3–5일.
- 사전학습 전체와 정책 학습: GPU 클러스터 기준 주 단위(논문이 정책 학습 시간을 보고하지 않아 추정하지 않음).

## 4. 재실행

```bash
git clone https://github.com/Yoonkyo/mu0 repo
git -C repo checkout 963d5f13a94df31921d22887ff8611763b48178d
git clone https://github.com/Yoonkyo/TraceExtract data/TraceExtract
git -C data/TraceExtract checkout c02eb8efefd988dda7f0046e38efb00762a84e08
python3 analysis/code_arithmetic.py   # 코드 상수로 파라미터 수 산술 (저장소를 실행하지 않음)
```

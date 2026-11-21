# 04. 논문과 공식 코드 대조 (정적 읽기만 함)

- 대상: https://github.com/DoppelWM/Doppel, commit `306fc88` (2026-11-06, "Link robot models"), `render_robot_wm/repo/`에 shallow clone.
- 방법: 파일을 읽어 논문 식·표·부록과 대조했다. 실행, import, 설치, 테스트(`pytest`), 가중치와 데이터 다운로드는 하지 않았다.
- 판정 기호: **일치**(코드와 논문이 맞음), **차이**(다름, 어느 쪽이 맞는지 불명), **누락**(논문이나 코드 중 한쪽에만 있음).

## 1. 대조 결과 요약

| 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|
| 백본 Wan 2.1 I2V 14B, 40 블록 학습 | §3, 부록 표 6 | `unfreeze_last_n_blocks: 40`, `model_path`는 Wan 14B 480P, revision `b184e23a`로 고정 | 일치 (코드가 리비전을 고정함) |
| 잠재 flow matching, 목표 ε − z, 첫 프레임 손실 제외 | 식 2–3 | `train.py` L1069 `target = noise - clean_latents`, L1098 `pred[:, :, 1:]` | 일치 |
| 조건 드롭아웃 0.1 (어댑터 기여를 0으로) | 부록 A.4 | `cond_dropout_prob: 0.1`; 드롭된 스텝은 `render_latents=None`으로 Wan 기본 forward | 일치 (동치: 기여 0) |
| 채널별 시공간 평균 제거 | 식 9 | `model_redar.py`/`redar_adapter.py` `RenderTokenEncoderWithSelfAttn.forward`, 평균은 fp32 | 일치 |
| Conv3D 16→256 (1×3×3), 256→512 (3×3×3), 512→512 (1×3×3) | 부록 A.4 | `conv1`, `conv2`(시간 축 replicate 패딩), `conv3`(stride (1,2,2)) | 일치 + 누락: conv2 패딩, conv3의 stride 2 (논문은 "토큰 격자"로만 씀) |
| 2025 토큰 | 부록 A.4 | 잠재 5 × 15 × 27 = 2025 (240×432 → 30×54 → stride 2 → 15×27, 17프레임 → 잠재 5프레임) | 일치 (계산으로 확인) |
| 위치 테이블 최대 32, 32, 64 | 부록 A.4 | `max_t 32, max_h 32, max_w 64` | 일치 |
| 행동 인코더 159 → 512 → 512, 시간 위치 128 | 식 11, 부록 A.4 | `ActionTokenEncoder`, `action_pos_emb` 128×512, `action_pad_dim: 159` | 일치 |
| 교차 주의 2층, 8 헤드, FFN ×4 | 식 12, 부록 A.4 | `num_cross_blocks: 2`, `num_heads: 8`, `ffn_mult: 4` | 일치 |
| 출력 LN→Linear 512→2048→GELU→Linear→5120 | 식 13 | `to_temb` | 일치 |
| 토큰별 시간 변조 (temb에 더한 뒤 time_proj) | 식 7, 14 | `RenderActionWanTransformer.forward`에서 `combine_with_temb` → `time_proj` → 블록과 `norm_out` | 일치 |
| 어댑터 파라미터 ≈ 30.5M | 부록 A.4 | 식으로 계산 30.57M (03 §5) | 일치 (반올림 차이) |
| 행동 전용 어댑터 4.2M | 부록 A.5 | `EgoWMActionAdapter` 계산 4.21M (action_dim 8) | 일치 |
| 학습 하이퍼파라미터 (표 6–9) | 표 6–9 | 설정 파일: 배치 8×8=64 또는 8×32=256, LR 5e-6/1e-5, 어댑터 ×4, warmup 500/250, EMA 0.9975/0.995, weight decay 1e-3, clip 1.0 | 일치 |
| Doppel-3 49.5k 스텝, Doppel-8 189k 스텝 | §5 | `multi_render_action_4node.json` 651 에폭 × 19,456 / 256 = 49,476; `emb8_*` 2490 × 76 = 189,240 | 일치 |
| 표본 비율 (Doppel-8 샘플러) | 표 1 | `sampler_dataset_weights` vs 클립 수 비율: 최대 차이 1e-6 | 일치 |
| Doppel-3 균등 샘플링 | §5, 부록 A.3 | `droid 0.3334 / robomind 0.3333 / bridge 0.3333` | 일치 (단, pass 수는 데이터셋마다 다름, 03 §3) |
| 25.6 s 롤아웃 = 3.2 s 창 8개 | §6 | `doppel/rollout.py` 1 + 16n 프레임, 창 연결 | 일치 |
| 평가 해상도 240×432, 프레임 1–16 평균 | 부록 A.8 | `evaluate.py`, `metrics.py` `first=1` | 일치 |
| 움직임 기준 0.25 × √면적 | 부록 A.8 | `metrics.py` `MOTION_THRESH = 0.25` | 일치 |
| SCS = SAM 2 마스크 IoU 평균, 전체 프레임 표는 "비로봇 물체 평균" | 부록 A.8 | `metrics.py` `SCS`, README는 `scs_objects`가 논문 값이라고 씀 | 일치 |
| 샘플러 50 Euler 단계, 가이던스 1.0 | 부록 표 5는 OSCAR만 (35단계, 6) | `inference.py` 기본값 50, `guidance_scale=1.0`, `evaluate.py` 독스트링 "as in the paper" | **차이/누락** (D1) |
| 영역별 PSNR(로봇, 움직이는 물체) | 표 10, 13, 그림 6 | 평가 코드에 없음. SCS만 영역별 계산 | **누락** (D2) |
| 신뢰구간 5000회 에피소드 부트스트랩 | 부록 A.10.3 | 코드 없음 | **누락** (D2) |
| 평가 가중치: EMA | 논문에 언급 없음 | `ema_eval: true` (모든 설정) | **누락** (D3) |
| 데이터 준비 (래턴트 저장소, 인덱스, 5% 검증 분할) | 부록 A.3 | README는 `scripts/data/build_latent_store.py`, `build_index.py`를 가리킴. **저장소에 없음** | **누락** (D4) |
| 플로우 타임스텝 분포 (τ ∈ [0,1]) | 식 2 | `train.py` L1058: 스케줄러의 이산 타임스텝에서 균등 샘플 | **차이 (사소)** (D5) |
| DROID 로봇 모델 | 표 1: Franka Panda | 보정·렌더러 모델 목록은 "Franka FR3" | **차이 (미확인)** (D6) |
| 보정 절차 | §4.1, RobotArena∞ [51] | `calibration/README.md`: RobotArena∞ 절차, RAFT, DINOv2, 그리드 시작 포즈 | 일치 |
| ABC 보정 | 부록 A.9 | `calibration/yam`: SAM 2 마스크 → 광도 정제 | 일치 (데이터는 없음) |
| 가중치 라이선스 | 논문 없음 | 가중치 CC BY-NC 4.0 | 참고 (D7) |

## 2. 차이와 누락 (상세)

- **D1. 추론 샘플러가 논문에 없다.** 코드 `inference.py` 기본값은 50 Euler 단계, 가이던스 1.0(분류자 없는 가이던스 없음)이다. `evaluate.py` 독스트링은 이를 "논문 프로토콜"이라 부르지만, 논문 본문과 부록에는 Doppel용 샘플러가 없다(표 5는 OSCAR만). 가이던스 없음과 50단계는 재현에 필요한 값이다.
- **D2. 영역별 PSNR과 부트스트랩이 릴리스에 없다.** 표 10의 Δarm, Δobj와 표 13, 그림 6은 `PSNR_arm`, `PSNR_obj`에 기반한다. 저장소에서 영역별 PSNR은 계산하지 않는다(SCS만 영역별). 표 10의 "Better" 비율과 SE, 부록 A.10.3의 5000회 에피소드 부트스트랩도 코드에 없다. `summarize.py`의 `_se`는 클립 단위 표준오차(std/√n)다.
- **D3. 평가에 EMA 가중치를 쓴다.** 모든 설정 파일에 `ema_eval: true`가 있다. 논문은 "final checkpoint"라고만 한다. 어느 가중치로 표 2를 냈는지 본문에서 알 수 없다.
- **D4. 데이터 준비 스크립트가 없다.** README의 학습 절차는 `scripts/data/build_latent_store.py`와 `scripts/data/build_index.py`를 쓰라고 한다. 저장소에는 `scripts/eval`과 `scripts/render`만 있다. 따라서 클립 선택 규칙(연속 창 5개 등, 부록 A.3), 5% 검증 분할, 테스트 분할 정의는 코드로 확인할 수 없다. 평가 세트 CSV도 없다.
- **D5. 타임스텝 샘플링.** 논문 식 2는 τ ∈ [0,1]을 쓴다. 코드는 Wan 스케줄러의 이산 타임스텝에서 균등하게 뽑는다(`torch.randint(0, len(scheduler.timesteps))`). 스케줄러의 shift 값은 논문에도 코드 인자로도 명시되지 않는다(체크포인트의 `scheduler` 설정을 쓴다). 영향은 작을 가능성이 높다.
- **D6. Franka 모델 종류.** 표 1은 DROID를 "Franka Panda"라 하고, `calibration/README.md`와 HF 모델 설명은 "Franka FR3"용 Dr.Robot 모델을 쓴다. 기구학이 가까워도 DROID 장치와 다르면 렌더 정렬이 어긋난다. 논문은 어느 모델로 DROID를 렌더링했는지 밝히지 않는다. 확인할 수 없음.
- **D7. 라이선스.** 코드 MIT, 가중치 CC BY-NC 4.0, `calibration/drrobot/` 일부 파일 3DGS 비상업 라이선스, Wan 2.1 Apache-2.0. 논문은 라이선스 진술이 없다. 연구용 재현에는 문제없지만 상업 이용은 불가하다.
- **D8. 저자 설명과 달리 조건 신호에 몸 종류가 들어 있다.** 논문 §6은 "모델은 어떤 몸을 보는지 알려받지 않는다"고 쓴다. 렌더 자체가 몸 모양을 담으므로 맞는 말이지만 "라벨 없음"의 실질적 의미는 약하다(01 §6.2). 코드 측면의 근거는 `redar_adapter.py`가 임베디먼트 토큰을 두지 않는다는 점이다.

## 3. 구현 정보가 논문보다 많은 부분 (코드가 채움)

- 행동 0 패딩 규칙과 데이터셋별 정규화 통계(`assets/action_stats/*.json`, `doppel/assets/embodiments.json`)
- 테스트 시 시드 규칙: `42 + sha1(row_key)[:8]` mod 2³¹ (`inference.py` `clip_seed`)
- 가중치 기반 SCS 설정: 최대 6개 객체, 프레임 0 임베디먼트 마스크(렌더 픽셀 > 25, 최소 면적 0.3%) (`metrics.py`)
- 학습 재개, EMA 샤드 저장, FSDP 설정 (`train.py`, `accelerate_fsdp.yaml`)
- 롤아웃 창 시드 규칙 `window_seed` (`rollout.py`)

## 4. 실행하지 않은 것

- `pytest tests`: 실행하지 않음. 테스트는 `tests/test_model.py`가 작은 DiT 구성을 만들므로 설치가 필요하다.
- 캘리브레이션 (gsplat 빌드, CUDA 필요): 실행하지 않음.
- `scripts/render/render_robot.py`: 읽기만 함. Dr. Robot 코드베이스를 요구한다.

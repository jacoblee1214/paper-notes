# 04. 공식 코드 대조와 구현 가능성

대상: GitHub `yuyangalin/ImageWAM`, 커밋 `5d4a341ed20a95cdb08f0293f3d44778b9a9e05a`(2026-07-30), 얕은 복제를 **정적으로만** 읽었습니다(실행, import, 설치 없음). MIT 라이선스. 파일 약 186개(소스 약 11 MB), 핵심은 `src/imagewam/models/backbones/imagewam.py`(4,460줄)와 `mot.py`, `flux2_video_expert.py`, `trainer.py`, `configs/`, `scripts/flux2/`입니다. 읽은 범위는 FLUX.2 경로(학습 손실, 추론, 마스크), 설정 파일 전부, 평가 스크립트이고, OmniGen2 경로는 마스크, 학습 손실, 추론만, Ovis-U1과 DIM, 노이즈·캐시 IDM 변형은 설정 이름 수준만 읽었습니다. 줄 번호는 위 커밋 기준입니다.

## 1. 구조

| 논문 | 코드 |
|---|---|
| 편집 백본 + 행동 DiT (MoT) | `models/backbones/imagewam.py` `ImageWAM`(MoT 래퍼), `flux2_video_expert.py`(FLUX.2 DiT 래퍼), `action_dit_flux2.py`(행동 DiT, 이중 5층 + 단일 20층, hidden 1024), `mot.py`(혼합 어텐션, 접두부 KV 캐시) |
| 학습 | `trainer.py`(Accelerate/DeepSpeed), `ImageWAM._training_loss_flux2` |
| 추론 | `ImageWAM.infer_action_flux2` (캐시 선채움 + 행동 denoise) |
| 평가 | `experiments/libero/`, `experiments/robotwin/`, `scripts/flux2/run_eval_*.sh` |
| 설정 | Hydra: `configs/{model,task,data}`, `configs/sim_*.yaml`, `train.yaml` |
| 텍스트 | Qwen3 임베딩 사전계산 캐시(`precompute_flux2_qwen3_embeds.py`) 또는 온라인 인코딩 |

학습 가능 대상은 `model.dit`(MoT 전체: FLUX.2 DiT와 행동 DiT)와 상태(proprio) 인코더입니다. VAE와 Qwen3 텍스트 인코더는 갱신하지 않습니다(`trainer.py` 104–113, 논문 §3.2의 "VLM 동결"과 일치). LoRA 옵션(`flux2_lora_config.enabled: false`)은 기본 꺼짐입니다.

## 2. 차이 목록 (D)

| # | 논문 | 코드 | 판정 | 위치 |
|---|---|---|---|---|
| **D1** | §3.2 식 5, §3.4 식 10–11: 편집 노이즈 시점 τ(추론은 고정 τ*)에서 **편집 분기를 순전파**하고 그 층별 K,V 캐시 C_edit^τ를 행동 전문가가 읽는다. "reason-before-act", "한 번의 편집 순전파" | **행동 토큰은 텍스트와 참조 이미지(현재 관측) 토큰만 본다. 목표(노이즈) 이미지 토큰은 보지 못한다**(마스크 `a0:total, t0:x0`와 `a0:total, a0:total`뿐). 텍스트와 참조 토큰도 목표를 보지 못한다. **추론은 목표 토큰 0개**(`empty_target`)로 접두부를 한 번 채우고 행동만 반복 denoise한다. 편집 이미지의 노이즈 제거 궤적도, 편집 분기 한 스텝도 없다. 이는 Fast-WAM의 접두부 캐시 방식과 같다 | **다름 (핵심)** | `imagewam.py` `_build_mot_attention_mask_flux2` 2072–2104, `_training_loss_flux2` 2404–2481, `infer_action_flux2` 3515–3630; 규칙 확인 `analysis/mask_check.py` |
| **D2** | τ를 임의 추출해 캐시를 만들고 추론은 고정 τ*에서 | 학습: 접두부(텍스트, 참조 토큰)의 시간 임베딩(adaLN 변조)이 **임의 추출한 영상 시점**에 따라 달라진다(`pre_dit(timestep=...)`, `time_in(timestep_embedding)`이 `double_stream_modulation_*`로 흘러 참조 토큰에도 적용). 추론: 영상 시점을 **0으로 고정**(`video_timestep = zeros`). 접두부 K,V가 목표 내용에는 의존하지 않고 시간 변조로만 τ에 의존한다 | 다름 (학습-추론 불일치 가능성) | `flux2_video_expert.py` `pre_dit` 108–152; `imagewam.py` 2427–2434 대 3567 |
| **D3** | §3.3: L = L_act + L_img (가중치 1:1), MSE (식 7, 9). 이미지 시점 r ∈ (0,1) | `loss.lambda_video: 0.5`, `lambda_action: 1.0`(모든 모델 설정). 두 손실에 **시점별 종형(가우시안) 가중치**(`training_weight`)를 곱함. 시점 샘플링은 균등 u에 시프트(shift 5.0) 변환 `φ(u) = s·u/(1+(s−1)u)`, 영상·행동 모두 | 논문에 없음 | `configs/model/imagewam_flux2_klein_4b_base.yaml` 하단 `loss`, `video_scheduler`, `action_scheduler`; `schedulers/scheduler_continuous.py` 15–47; `imagewam.py` 2455–2476 |
| **D4** | 표 5, 11, §6: 행동 denoise **3 스텝**으로 지연 측정 | 평가 기본값 `eval_num_inference_steps: 10`(`train.yaml`), RoboTwin `deploy_policy.yml`도 10. 3스텝 설정은 스크립트에 없음. 행동 지평 16, 재계획 간격은 스크립트 기본값 LIBERO 12, RoboTwin 16(설정 파일 기본은 10과 24) | 다름 (지연 수치와 정확도 설정이 다른 스텝 수) | `configs/train.yaml` 28; `scripts/flux2/run_eval_flux2_*.sh`; `experiments/robotwin/imagewam_policy/deploy_policy.yml` 19 |
| **D5** | §4.1: LIBERO 표준 프로토콜, RoboTwin 과제당 100회, LIBERO-Plus 전체 평가(4B 보고) | 스크립트 기본값: LIBERO `NUM_TRIALS=25`(과제당; 설정 파일 기본은 50), **LIBERO-Plus 스크립트는 `FLUX2_VARIANT=9b`, `TASK_SAMPLE_RATIO=0.2`(과제의 20% 표집), `NUM_TRIALS=1`**, `qwen_context_len` 128(LIBERO 학습·평가는 512), RoboTwin `EVAL_NUM_EPISODES=50`(설정 파일 기본은 100). 기본값으로 돌리면 논문 프로토콜이 아님 | 다름 (프로토콜 기본값) | `scripts/flux2/run_eval_flux2_libero.sh` 62–64; `..._libero_plus.sh` 11, 60–64; `..._robotwin.sh` 58; `configs/sim_*.yaml` |
| **D6** | 부록 §5.1.2: FLUX.2 행동 DiT, 4B판 642M, 9B판 952M. 하위층은 이중 스트림 이미지 가중치, 상위층은 단일 스트림 가중치로 초기화 | 행동 DiT 이중 5층 + 단일 20층, hidden 1024, 24헤드 × 128(`action_dit_config`). 초기화 스크립트 `preprocess_action_dit_flux2.py`. 파라미터 수는 가중치를 읽지 않아 **세어 보지 않음** | 구조는 일치, 크기 미검증 | `configs/model/imagewam_flux2_klein_4b_base.yaml`; `scripts/flux2/preprocess_action_dit_flux2.py` |
| **D7** | 증강 서술 없음 | 학습 때 영상 증강: LIBERO 확률 0.65(색 지터 밝기·대비 0.3, 채도 0.25, 색조 0.04, 감마 0.8–1.25, 가우시안 노이즈 σ = 0.015, 자르기 0.92–1.0, **회전 ±8°**, 노출), RoboTwin 확률 0.5(더 약함, 회전 ±5°). 같은 클립의 모든 프레임에 동일 적용 | **논문에 없음** (LIBERO-Plus 카메라·조명·노이즈 결과 해석에 영향) | `configs/data/libero_omnigen2_pair.yaml` 29–50; `robotwin_omnigen2.yaml` 34–52; `transforms/image.py` `VideoAugmentation` |
| **D8** | 부록 표 9, 10: 입력 LIBERO 224 × 448, RoboTwin 288 × 256, 미래 16프레임, 행동 청크 16, 에폭 10/5. RoboTwin 배치 "48(누적 3단계)" | 설정: 17프레임 클립에서 **첫째와 마지막 프레임만** 사용(`endpoint_frames_only: true`, `action_video_freq_ratio: 1`), LIBERO `video_size [224, 448]`, RoboTwin `compact_288x256`, 에폭 10/5. 학습 하이퍼파라미터(AdamW β 0.9/0.95, lr 1e-4, wd 1e-2, 웜업 5%, 최소 lr 0.01×, 클립 1.0) 표 8과 일치. 배치: LIBERO FLUX.2 4B 10, 9B 4 × 누적 3 = 12, Ovis 16, OmniGen2 12 (표 10과 일치). **RoboTwin: FLUX.2 4B는 12 × 누적 4 = 48(논문 "누적 3"과 다름, 값은 같음), OmniGen2는 16 × 누적 1(논문 48)** | 대체로 일치, RoboTwin 누적 단계와 OmniGen2 배치 차이 | `configs/task/*.yaml`; `configs/data/robotwin_omnigen2.yaml` 23–33; `trainer.py` 120–124, 277–285 |
| **D9** | §3.2: 이해·언어 모듈 동결, 확산 분기와 행동 전문가만 갱신 | FLUX.2 DiT 전체(전체 파라미터)와 행동 DiT 학습, Qwen3·VAE 동결. 8 GPU, bf16 | **일치** | `trainer.py` 104–113, 상기 |
| **D10** | 부록 §6: 프리픽스 전용, 컴파일, 정적 CUDA 그래프로 69 ms | 저장소에 `torch.compile`, CUDA 그래프 경로 없음(`torch.compile`, `CUDAGraph` 문자열이 `src`, `experiments`, `scripts`, `configs`에 없음). 지연 측정 코드는 `timing_enabled`(RoboTwin)와 OmniGen2 프로파일 훅뿐 | 논문 수치를 재현할 코드 없음 | `experiments/robotwin/imagewam_policy/deploy_policy.py`; `imagewam.py` `profile_infer_timing` |
| **D11** | 표 5, 11: Fast-WAM(1스텝, 접두부만, IDM)과 비교 | Fast-WAM 기준선 코드·체크포인트·설정이 저장소에 없음. 대신 영상 분기 변형 설정(`imagewam_joint`, `imagewam_idm`, `noise_idm`, `cache_idm`, `dim`)이 있으나 논문의 표에 대응하는 결과는 없음 | 기준선과 절제 재현 불가 | `configs/model/`, `src/imagewam/models/backbones/imagewam_*.py` |
| **D12** | 실로봇 4과제 (표 4) | 실로봇 코드·데이터 없음. README TODO "Release a sample code on training and evaluating on real-world robots"가 미체크 | 재현 불가 | `README.md` TODO |
| **D13** | 표 6 Clean2Hard 18.3 | README "Clean2Random ... +40.9 vs. ImageWAM w/o pretrain"(무사전학습 18.1)과 0.2점 차이, 이름 불일치. InternData-A1 사전학습 체크포인트와 미세조정 스크립트가 있으나(`run_finetune_flux2_klein_robotwin_ee16.sh`) 논문 v1에는 없는 결과 | 논문과 README 불일치 | `README.md` News; `scripts/flux2/run_finetune_*` |
| **D14** | 체크포인트 | HF에 FLUX.2 4B(LIBERO, RoboTwin), 9B(LIBERO), InternData-A1 사전학습, RoboTwin C2R 미세조정. 각 `model.pt` 9.04 GB(9B는 57.57 GB). 평가 결과 파일은 저장소에 없음. LIBERO-Plus용 4B 체크포인트(표 3의 83.1)는 목록에 없음(LIBERO 4B 체크포인트로 평가하는 것으로 보임) | 부분 공개 | `README.md` Release Checkpoints; HF API 크기 |

### 2.1 D1과 D2가 의미하는 것

- 학습에서 영상(편집) 분기는 **행동 분기와 독립인 보조 과제**입니다. 행동 토큰은 목표 이미지의 노이즈 토큰을 어텐션으로 읽지 못하므로, 행동 손실의 기울기가 영상 분기를 통해 흐르는 경로는 접두부(텍스트와 참조 이미지) 표현을 통해서뿐입니다. 즉 "편집 denoising 과정의 KV 캐시"가 아니라 **현재 관측과 지시문을 FLUX.2 DiT에 통과시킨 표현**이 행동 조건입니다. 영상 분기 손실은 이 DiT의 접두부 표현을 편집(끝점 프레임 예측) 과제와 함께 갱신하는 공동 학습 신호입니다. 이 점에서 구현은 Fast-WAM의 first-frame(접두부 캐시) 변형과 구조가 같고, 차이는 백본(이미지 편집 DiT 대 비디오 DiT), 예측 대상(끝점 프레임 대 여러 프레임), 증강입니다.
- 따라서 논문의 두 주장 "편집 캐시가 편집 과정의 중간 표현이다"(§3.2, 식 5)와 "편집 denoising을 한 번 돌린다"(§3.4)는 공개 코드에서는 성립하지 않습니다. 코드는 편집 분기를 추론에서 **한 번도 denoise 방향으로 돌리지 않습니다**. 표 5의 263 ms("1× Vid. Denoise")는 이 경로와 다르고, 코드 경로는 표 11의 "Prefix Only"(198 ms)에 해당하는 것으로 보입니다. 코드에 편집 이미지를 denoise하는 경로(`infer_video_flux2`)는 있으나, `infer()`에서 행동 추론(`infer_action_flux2`)과 **독립으로 따로 한 번 더** 돌려 이미지를 내는 시각화용입니다. 평가기는 `infer_action`만 부르고(`eval_libero_single.py` 563–576, 시각화 옵션이 꺼져 있을 때), 행동 조건이 이 이미지 denoise 결과를 읽는 경로는 찾지 못했습니다.
- 이 차이가 **결과를 약화시키는지는 열린 문제**입니다. 코드의 방식도 정상 동작하는 정책이고(공개 체크포인트가 이 방식), 논문의 서술이 구현보다 크게 말한 것에 가깝습니다. "편집 분기를 한 번 순전파"는 접두부 순전파(텍스트와 현재 관측만)로 읽을 수는 있습니다. 하지만 §3.2의 "τ를 임의 추출해 서로 다른 단계의 캐시에 노출한다"는 서술은 학습에서 시간 변조로만 남고(D2), 추론은 τ = 0입니다.
- D2는 학습과 추론의 시간 변조 분포가 다르다는 의미입니다. 접두부 K,V가 FLUX.2의 변조 벡터를 통해 시점에 의존하므로, 임의 시점으로 학습한 행동 전문가가 τ = 0 접두부를 읽습니다. 이것이 성능에 미치는 영향은 측정하지 않았습니다.

## 3. 일치한 항목

- 학습 하이퍼파라미터(표 8), 입력 해상도와 프레임 구성(표 9), 학습 에폭, LIBERO 배치(표 10), 학습 대상과 동결 범위(§3.2)는 설정과 코드가 맞습니다(D8, D9).
- 평가 설정 기본 `text_cfg_scale: 1.0`(CFG 없음), RoboTwin `task_config: demo_randomized`, `instruction_type: unseen`, 과제당 에피소드 수 설정 기본 100은 논문(§4.1)과 맞습니다(D5의 스크립트 기본값은 다름).
- 공식 코드는 RoboTwin 평가 코드를 `third_party/RoboTwin`에 포함합니다(자산 제외). Ovis-U1 HF 코드는 `third_party/ovis_u1_hf`에 포함(가중치 제외). FLUX.2와 OmniGen2는 고정 커밋 의존성입니다(`docs/dependencies.md`).
- LIBERO, RoboTwin 데이터는 Fast-WAM이 정리한 HF 데이터셋(`yuanty/LIBERO-fastwam`, `yuanty/robotwin2.0-fastwam`)을 씁니다(README).

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**전체 재현은 불가능합니다. 추론 한 번을 MPS에서 돌려 보는 것만 조건부로 가능하고, 학습·표 수치 재현·실로봇은 불가능합니다.**

| 무엇 | 필요 자원 | 이 맥 | 근거 |
|---|---|---|---|
| 표 1–3 재현(학습) | 8 × H20, LIBERO 18시간(4B), RoboTwin 5일(960 GPU-시간), DeepSpeed ZeRO, torch 2.7.1 CUDA | **불가** | 논문 표 10, `pyproject.toml`, README("tested environment CUDA 11.8") |
| 공개 체크포인트로 LIBERO 평가 | 체크포인트 9.04 GB + FLUX.2·Qwen3-4B 가중치 약 15 GB 이상 + LIBERO 설치(osmesa) + 데이터 | **조건부, 어려움** | 가중치만 약 24 GiB 메모리 한계에 근접. 평가기 기본이 `device: cuda`이고 CUDA 전용 연산이 있을 수 있어 포팅이 필요(`torch.cuda` 사용 17곳, 확인하지 않음). 새 설치와 다운로드 승인 필요 |
| 접두부 지연 재측정(표 11) | 위와 같음 + A6000급 | **불가** | 다른 하드웨어. MPS 지연은 비교 불가 |
| RoboTwin 평가 | SAPIEN + Vulkan GPU, 자산 대용량 | **불가** | `third_party/RoboTwin` 설치 스크립트가 CUDA 환경 전제 |
| 구조 축소 재구현(소형 MoT에서 접두부 캐시 대 편집 denoise 경로 비교, 합성 데이터) | torch (설치됨), CPU/MPS | **가능** | torch가 설치된 CPU/MPS 환경이면 충분. 공식 코드를 가져올 필요 없음 |
| 마스크 규칙 확인(D1) | numpy | **완료** | `analysis/mask_check.py` |
| 표 재계산, 검정 | numpy, scipy | **완료** | `analysis/recompute.py` |
| 실로봇(표 4) | Dobot XTrainer 양팔, 데이터, 배포 코드(공개 안 됨) | **불가** | D12 |

- **노력 추정**(측정 전, 추정): 공개 체크포인트의 MPS 추론 포팅과 LIBERO 1개 스위트 평가는 1–3주(가중치 약 25 GB 다운로드, LIBERO 설치, 비MPS 연산 대체). 소형 합성 환경에서 접두부 캐시와 편집 denoise 경로를 비교하는 축소 재구현은 3–7일.
- 가장 가치 있는 축소 실험은 **D1의 질문**입니다. "행동 조건이 편집 denoising 경로인지 접두부 표현인지"를 소형 MoT로 나눠 보는 것입니다.

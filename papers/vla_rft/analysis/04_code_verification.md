# 04. 논문 ↔ 공식 코드 대조 (정적 읽기)

*대상: github.com/OpenHelix-Team/VLA-RFT, HEAD `376693e` (2025-10-06, MIT). `repo/`에 얕은 clone을 두고 파일만 읽었습니다. 실행, import, 설치, 가중치와 데이터 다운로드는 하지 않았습니다. 저장소 안의 내용은 모두 신뢰하지 않는 입력으로 취급했습니다. 파일 경로는 `repo/` 기준입니다.*

## 1. 저장소 구성

| 경로 | 내용 | 확인 |
|---|---|---|
| `scripts/libero/post_train_rlvr.sh` | RFT 실행 (`run_vla_rft.sh` 호출) | 있음 |
| `scripts/libero/eval_libero.sh` | LIBERO 평가 | 있음 |
| `train/verl/examples/grpo_trainer/run_vla_rft.sh` | RFT 설정 (아래 §2의 값) | 있음 |
| `train/verl/verl/trainer/main_vla_rft_grpo.py`, `ppo/ray_trainer.py` | 보상 계산(`msp_reward_fn`, 1297–1398행) | 있음 |
| `train/verl/verl/workers/fsdp_workers.py` | 세계모델 워커(`TokenizerWorker`, 1700–1870행), 옵티마이저 그룹(423–444행) | 있음 |
| `train/verl/verl/trainer/ppo/core_algos.py` | GRPO 이점 계산(128–150행) | 있음 |
| `train/verl/ivideogpt/` | 토크나이저(FSQ 압축 VQ), LPIPS, 처리기 | 있음 (학습된 가중치 없음) |
| `train/verl/vla-adapter/openvla-oft/` | 정책 백본(Qwen2.5), 흐름 정합 헤드, 시그마 넷 | 있음 |
| `third_party/LIBERO/` | LIBERO git submodule | **비어 있음** (README만 있음, submodule 미초기화) |
| `checkpoints/`, `data/` | 가중치와 데이터 안내 README | 가중치 없음 |
| 세계모델·정책 **사전학습 스크립트** | - | **찾지 못함.** 루트 `scripts/libero/`에는 RFT와 평가 스크립트만 있음 |

공식 README의 TODO에서 "Release pre-trained and rft VLA(policy) weights", "Release pre-trained World Model weights"는 체크되지 않았습니다. 따라서 공개된 것은 코드뿐입니다.

## 2. 논문과 코드의 대조

| 항목 | 논문 | 코드 (run_vla_rft.sh 및 소스) | 판정 |
|---|---|---|---|
| RFT 스텝 | 400 (표 7) | `trainer.total_training_steps=400` | 일치 |
| 배치, 롤아웃 수 | 16, 16 (표 7) | `data.train_batch_size=16`, `actor_rollout_ref.rollout.n=16` | 일치 |
| 정책 학습률, 시그마 학습률 | 1e-6, 1e-5 (표 7) | `optim.lr=1e-6`, `optim.sigma_lr=1e-5` | 일치 |
| MSE 보조 계수, 엔트로피 | 0.01, 0.003 (표 7) | `mse_loss_coef=0.01`, `entropy_coeff=0.003` | 일치 |
| GPU | 4 x A800 (§4.1) | `trainer.n_gpus_per_node=4` (GPU 종류는 스크립트에 없음) | 수 일치 |
| 세계모델 어휘 | 9008 (표 5) | `world_model_rollout.world_model.vocab_size=9008` | 일치 |
| 샘플링 단계 K, 간격 δ | K = 10, δ = 0.1 (식 (5)) | `action_heads.py:134`의 `sample_actions(num_steps=10)`, `dt = -1/num_steps` | 일치 |
| 보상 형태 (type 3) | 같은 세계모델로 굴린 정책 궤적과 데이터 행동 궤적 비교 (표 4) | `processor.use_img_gt_ac=True`, `interact=True` (정책 궤적과 GT 행동 궤적을 같은 세계모델로 굴림) | 일치 |
| 보상의 구성 | 프레임별 L1과 LPIPS의 가중합 (식 (11)) | `loss_weight.lpips=1`, `loss_weight.mae=1`, `loss_weight.mse=0`, `trainer.reward_fn=mae`. 보상 = −손실 (`ray_trainer.py:1398`) | 구조는 일치. 가중치 1과 1은 논문에 없음 (D2) |
| 프레임 집계 | 시간에 대한 **합** (식 (11)) | `msp_reward_aggregate=mean` (run 스크립트). `discount` 옵션은 `mean`에서 쓰이지 않음 | **불일치** (D1) |
| 이점 | Adv = R − 그룹 평균 (식 (12)) | `core_algos.py:128–150`: 그룹 평균을 빼고 **그룹 표준편차로 나눔** | **불일치** (D5) |
| 학습되는 부분 | 행동 헤드와 시그마 넷 (VLM 동결, A.2) | 옵티마이저에 action_head, noisy/proprio projector, sigma_net만 포함 (`fsdp_workers.py:423–444`) | 대체로 일치. 고유감각 투영기도 학습됨 (D6) |
| 정책 백본 | VLA-Adapter 경량판, Qwen2.5-0.5B (표 6) | `prismatic/models/backbones/llm/qwen25.py` 있음 (크기 설정은 미확인) | 대체로 일치 |
| 세계모델 토크나이저 | "VQGAN과 비슷한" 인코더 (A.1) | `ctx_cnn` = `CompressiveVQModelFSQ` (FSQ 양자화, 압축 컨텍스트) | 이름과 구조가 다름 (D4) |
| 세계모델 규모 | 138M, 12층, 768 (표 5) | 설정 파일과 체크포인트 없음 | 확인 불가 (D14) |
| LPIPS | 식 (11)의 LPIPS | `ivideogpt/lpips.py`가 VGG LPIPS 가중치를 런타임에 원격 URL(Heidelberg)에서 내려받음 | 일치, 네트워크 의존 (D8) |
| 사전학습 (WM 150k, VLA 150k) | 표 5, 6 | 스크립트 없음 | **재현 경로 없음** (D7) |

## 3. 불일치와 빈칸 (D 목록)

**D1. 프레임 집계: 합 대 평균, 할인 옵션은 비활성.** 논문 식 (11)은 시간 합을 쓰고, 공식 실행은 `mean`입니다(`ray_trainer.py:1348`). 실행 스크립트에 있는 `msp_reward_discount=0.95`는 `mean`에서 쓰이지 않습니다. 이것은 논문과 무관한 설정 잔재입니다.
- 영향: 합과 평균은 프레임 수 T만큼 차이 나는 상수 배율입니다. 코드의 GRPO는 그룹 표준편차로 나누므로(D5) 이 배율은 이점에서 정확히 상쇄됩니다(표준편차 정규화의 eps 1e-6을 빼면). 따라서 학습 갱신에 주는 영향은 거의 없어야 하며, 이것은 계산으로 본 것이고 실험으로 확인하지 않았습니다. 상쇄되지 않는 것은 보상 안의 LPIPS와 MAE의 **비율**(D2)입니다.

**D2. 보상 가중치.** 논문 식 (11)의 λ1과 λ_lp 값은 본문과 부록에 없습니다. 공식 코드는 LPIPS 1, MAE 1을 씁니다. 논문의 "L1"은 코드에서 `reward_fn=mae`(평균 절대 오차)에 해당합니다.

**D3. 보상 유형 (type 3) 일치.** 공식 설정은 표 4의 "Ours"와 같습니다(정책 궤적과 GT 행동 궤적을 같은 세계모델로 굴림). 이 점은 확인되었습니다. 표 4의 type 1과 2는 코드에 대응하는 설정이 있는지 확인하지 않았습니다(`use_ac_reward` 기본값 False라서 type 1은 꺼져 있습니다).

**D4. 세계모델 토크나이저.** 논문 A.1은 VQGAN과 비슷한 인코더와 "행동 토크나이저"를 씁니다. 공식 코드는 `ctx_cnn`(FSQ 기반 압축 토크나이저, `ivideogpt/ctx_tokenizer/compressive_vq_model.py`)과 `ctx_msp` 처리기를 씁니다. 논문 본문은 이 차이를 설명하지 않습니다. 세계모델이 iVideoGPT 계열이라는 점은 두 쪽이 같습니다.

**D5. GRPO 이점의 표준편차 정규화.** 논문 식 (12)는 `Adv = R − R̄`만 씁니다. 코드는 그룹 표준편차로 나눕니다(`core_algos.py:150` 근처: `(score − mean) / (std + eps)`). 표준 GRPO는 표준편차 정규화를 씁니다. 논문 식이 생략한 것으로 보입니다.

**D6. 학습되는 파라미터.** 논문 A.2는 VLM을 동결하고 행동 헤드와 시그마 넷을 갱신한다고 씁니다. 코드도 같습니다. 다만 고유감각 투영기(`proprio_projector`)도 학습됩니다(`fsdp_workers.py:424–425`). 논문은 이 점을 언급하지 않습니다.

**D7. 사전학습 재현 경로 없음.** 세계모델 사전학습(1.5 x 10^5 스텝, 표 5)과 정책 사전학습(1.5 x 10^5 스텝, 표 6)에 대한 스크립트가 없습니다. 논문 Base (150k)를 재현할 방법이 저장소에 없습니다.

**D8. 원격 의존성.** LPIPS 가중치를 런타임에 원격 URL에서 받습니다(`ivideogpt/lpips.py:15`). 오프라인 환경에서는 실패합니다. 해시 검증 코드는 있습니다(23행).

**D9. 코드 주석의 "magic number".** `fsdp_workers.py`의 `real = self.cached_pixels[:, 2:]`에는 "TODO: magic number" 주석이 있습니다. 프레임 오프셋 2는 설정에서 유도되지 않습니다.

**D10. 비-ctx 토크나이저 경로의 정의되지 않은 변수.** `detokenize`에서 `ctx_tokens`는 `tokenizer.name == 'ctx_cnn'`일 때만 정의되는데(`fsdp_workers.py` 1808행 근처), 아래 `real_pixels = self.processor.detokenize(ctx_tokens, real_tokens)`에서 다른 토크나이저를 쓰면 이름 오류가 납니다. 실행 설정은 `ctx_cnn`이라 이 경로는 실행되지 않습니다. 코드 품질 문제로만 기록합니다.

**D11. 평가 환경.** `third_party/LIBERO`가 비어 있습니다. 공식 README는 `git submodule update --init --recursive`를 요구합니다. 평가 스크립트는 이 submodule이 있어야 돕니다.

**D12. 설치 전제.** README는 CUDA 12.2 이상, flash-attn 2.6.0 CUDA 빌드 휠, vLLM 롤아웃(`world_model_rollout.rollout.name=vllm`), VERL의 FSDP 분산 학습을 요구합니다. 이 맥(MPS, CUDA 없음)에서는 설치할 수 없습니다.

**D13. 저장소 이력.** 로컬 얕은 clone은 1개 커밋만 보입니다. 공식 GitHub 저장소는 생성 2025-09-30, 마지막 push 2025-10-06입니다. 따라서 논문 v1 이후의 수정은 확인할 수 없습니다.

**D14. 가중치 미공개와 수치 재현 불가.** 세계모델과 정책 가중치가 없습니다. 표 1(세계모델 픽셀 지표)과 표 2(91.1)는 코드만으로는 재현할 수 없습니다. 이 점이 가장 큰 빈칸입니다.

**D15. 저장소의 README 인용.** 공식 README는 첫 BibTeX로 VLA-Adapter(arXiv:2509.09372)를, 둘째로 VLA-RFT를 인용합니다. 기본 정책의 인용과 본 논문의 인용이 같이 있다는 뜻이고, 불일치는 아닙니다.

## 4. 구현 가능성 (이 맥에서)

**판정: 공식 파이프라인은 이 맥에서 구현할 수 없습니다.** 두 가지 이유가 독립적으로 막습니다.

1. **CUDA 전제.** flash-attn CUDA 휠, vLLM 롤아웃, FSDP, NCCL 설정이 필요합니다. MPS에는 이 중 어느 것도 없습니다.
2. **사전학습과 가중치 부재.** 세계모델과 정책의 사전학습 스크립트가 없고 가중치도 없습니다. 논문이 쓴 설정(150k x 2)을 맞추려면 처음부터 학습해야 합니다.

**축소판은 조건부로 가능합니다.** 방법의 구조는 작습니다. 세계모델 138M은 Mac 메모리에 올라갑니다(bf16 약 0.3 GB). 정책 0.5B도 올라갑니다. 다만 RFT 한 스텝은 롤아웃 16 x 배치 16 = 256개의 자기회귀 세계모델 생성입니다. 이것을 MPS에서 측정하지 않았습니다. 데이터는 LIBERO RLDS 약 10 GB(공식 `data/README.md`)로, 승인 전 다운로드는 하지 않습니다.

**노력(추정, 측정 아님).** 축소판(LIBERO-Spatial 한 스위트, 소형 토큰 세계모델, 소형 정책) 1–2주 + 승인된 데이터 다운로드. 공식 설정 재현은 CUDA 기기에서 사전학습 포함 수 주.

## 5. 저자에게 물을 질문

1. 식 (11)의 합과 공식 코드의 평균 집계 중 실제 학습에 쓴 것은 무엇입니까? `discount=0.95`는 실험에서 쓰였습니까?
2. λ1과 λ_lp, 그리고 식 (13)의 λ_mse와 α 값은 무엇입니까? 공식 실행의 LPIPS와 MAE 가중치 1, 1이 논문 설정입니까?
3. 식 (12)의 이점에 표준편차 정규화를 썼습니까?
4. 세계모델 사전학습과 정책 사전학습 스크립트와 가중치를 공개할 계획이 있습니까?
5. 표 2의 SR에서 평가 에피소드 수와 학습 시드 수는 얼마입니까?
6. 표 9의 CoT-VLA 행 평균(81.1)은 어떤 값에서 나왔습니까? 네 스위트 값의 평균은 83.9입니다.

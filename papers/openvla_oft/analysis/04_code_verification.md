# 04. 공식 코드 대조와 구현 가능성

대상 저장소는 `github.com/moojink/openvla-oft`(MIT)의 커밋 `e4287e9` (2025-09-08, 마지막 push 2025-09-09)입니다. 얕은 복제본(`repo/`)을 **정적으로만** 읽었습니다. 실행하거나 import하거나 설치하지 않았습니다. 복제본과 다운로드한 내용은 신뢰하지 않는 데이터로 취급했습니다. 대조 기준은 논문 v2의 본문과 부록 A–G입니다. 분석일 2026-11-22.

## 1. 대조 결과 (D)

| 번호 | 논문 주장 (위치) | 코드 (파일:줄) | 판정 |
|---|---|---|---|
| D1 | 병렬 디코딩은 행동 자리에 빈 임베딩을 넣고 위치 인코딩만 다르게 한다 (IV-B, 부록 B.1) | `prismatic/extern/hf/modeling_prismatic.py:621`, `:891`: 행동 토큰 임베딩을 0으로 바꾸고 그 뒤에 위치 정보가 더해진다는 주석 | 일치 |
| D2 | 인과 마스크를 양방향 어텐션으로 바꾼다 (부록 A(3), B.1) | 이 저장소에는 마스크를 바꾸는 코드가 없다. `modeling_prismatic.py:742`의 주석만 "non-causal bi-directional self-attention"을 말한다. `pyproject.toml:50`은 "bidirectional attn"을 위해 `moojink/transformers-openvla-oft` fork를 설치하라고 쓴다 | **이 저장소만으로는 검증 불가.** 구현은 외부 fork에 있다 (미검증) |
| D3 | 연속 행동 헤드는 4층 MLP(ReLU)다 (부록 A(4), B.2) | `prismatic/models/action_heads.py:59–95`: LayerNorm → Linear → ReLU → 잔차 블록 2개(각각 LayerNorm, Linear, ReLU) → LayerNorm → Linear. 선형층은 4개다. 입력은 청크 한 단계의 7개 은닉 상태를 이어 붙인 28,672차원(`:95`, `input_dim*ACTION_DIM`) | 부분 일치. LayerNorm과 잔차 연결, 입력 이어 붙이기는 논문에 없다 |
| D4 | 학습 손실은 예측과 정답 행동의 평균 L1이다 (IV-B) | `vla-scripts/finetune.py:390`: `torch.nn.L1Loss()`를 청크 × 차원 전체에 적용 | 일치 |
| D5 | 학습률 5e-4, 100K 스텝 뒤 5e-5 (부록 D) | `finetune.py:89` (5e-4), `:91` (`num_steps_before_decay=100_000`), `:90` (워밍업 0), `:941–945` MultiStepLR(gamma 0.1) | 일치 |
| D6 | LIBERO 설정: 150K, 150K, 50K, 150K 스텝, 배치 64, LoRA 32, 청크 8, 손목 카메라와 자세 사용 (표 IV) | `finetune.py:88`(배치 8/GPU × 8 GPU = 64), `:107`(LoRA 32). 실행 명령은 `LIBERO.md:96–122`. 150K 체크포인트 기준은 `LIBERO.md:122` | 일치 |
| D7 | ALOHA 설정: 100K, 70K, 50K, 100K 스텝, LR 감소 50K, 배치 32, 청크 25 (표 V) | `ALOHA.md:54`(청크 25), `:84`(pot 100K, LR 감소 50K). 상수 `prismatic/vla/constants.py:34–35` (청크 25, 행동 14차원) | 일치. 본문의 "50–150K"와는 다르다 (K10) |
| D8 | LoRA 대상 모듈, alpha, dropout, 초기화 (논문에 없음) | `finetune.py:847–853`: `lora_alpha=min(rank, 16)` (rank 32이면 16), `target_modules="all-linear"`, `init_lora_weights="gaussian"`, dropout 0 (`:108`) | **논문에 없음.** 재현하려면 필요한 설정이다 |
| D9 | 이미지 증강: 90% 면적 무작위 자르기, 밝기·대비·채도·색조 (표 IV) | `prismatic/vla/datasets/datasets.py:153–161` (`random_resized_crop scale=0.9`, `ratio=1`, brightness 0.2, contrast 0.8–1.2, saturation, hue 0.05). 평가는 가운데 자르기 (`experiments/robot/libero/run_libero_eval.py:99`) | 일치 |
| D10 | 과제당 50회 × 10 과제 = 500회 (표 I 캡션) | `run_libero_eval.py:114` (`num_trials_per_task=50`), `:100` (열린 루프 8), `:113` (대기 10 스텝), `:128` (시드 7) | 일치. 시드 평균 여부는 `LIBERO.md:88`이 "averaged over three random seeds"라고 씀 (K1) |
| D11 | 50K 스텝마다 체크포인트를 시험하고 최고 성능을 보고한다 (부록 D) | `finetune.py:97` (`save_freq=10_000`). 체크포인트 선택 스크립트나 평가 로그는 저장소에 없다 | **재현 불가** (선택 절차 미공개, C22) |
| D12 | FiLM은 `(1+γ)F + β`이고 γ, β는 평균 언어 임베딩에서 나오며, 패치 임베딩의 채널마다 적용하고, 블록마다 투영기를 따로 둔다. 자기주의 뒤, FFN 앞에 넣는다 (IV-C, 부록 C) | `prismatic/models/film_vit_wrapper.py:11–75`: attention 뒤 (`:69`), `x * (1 + gamma) + beta` (`:72`), MLP 앞 (`:75`). 투영기는 `nn.Linear(llm_dim, vision_dim)` 두 개 (`:53–54`) | 일치 |
| D13 | 자세 투영기는 GELU가 있는 2층 MLP다 (부록 B.3) | `prismatic/models/projectors.py:6–17`: `fc1 → GELU → fc2` | 일치 |
| D14 | 행동은 [-1, 1]로 정규화한다 (IV-A) | `prismatic/vla/constants.py:30`: LIBERO는 `BOUNDS_Q99`, `:37`: ALOHA는 `BOUNDS`. 논문 본문은 정규화 종류를 쓰지 않는다 | 부분 일치. 분위수 기준을 논문에 명시해야 한다 |
| D15 | 파라미터 수: LIBERO 279M(LoRA 111M + 헤드 151M + 자세 17M), ALOHA 853M(+ FiLM 456M, 헤드 269M) (표 IV, V) | 구조에서 계산: 헤드 151.1M과 268.6M, FiLM 456.2M, 자세 투영기 약 16.8M (`analysis/recompute.py`의 V9). LoRA는 all-linear 구조 추정 약 108M | 헤드, FiLM, 합계는 일치. LoRA 111M은 약 3% 차이 (추정) |
| D16 | 추론은 청크 전체를 실행한 뒤 다시 질의한다 (VI.A) | `run_libero_eval.py:306` (큐 길이 = `num_open_loop_steps`), `:341–344` (청크를 큐에 넣고 하나씩 꺼냄) | 일치 |
| D17 | 확산: DDIM 50 단계, squared cosine 스케줄, 4층 잡음 예측기 (부록 B.2) | `prismatic/models/action_heads.py`: `DDIMScheduler(..., beta_schedule="squaredcos_cap_v2")`, `NoisePredictionModel`은 `MLPResNet` 기반 | 일치 |
| D18 | 학습 데이터는 잡음 행동(0에 가까운 행동)과 실패 시연을 뺀다 (표 I 캡션) | `experiments/robot/libero/regenerate_libero_dataset.py:6–7`, `:46` (`is_noop`), `:178–201` (성공 궤적만 저장) | 일치 (데이터 재생성 스크립트 기준. 공개 데이터셋 자체는 받지 않음) |
| D19 | 평가 최대 스텝 (논문에 없음) | `run_libero_eval.py:TASK_MAX_STEPS` = 220, 280, 300, 520 (스위트 순서) | **논문에 없음.** 재현에 필요한 설정 |
| D20 | 추론 장치와 정밀도 (논문은 A100에서 측정) | `experiments/robot/openvla_utils.py:39`: `DEVICE = cuda:0` 또는 CPU (MPS 없음). `prismatic/models/vlas/openvla.py:79`: `torch.autocast("cuda", ...)`. `openvla_utils.py:285`: bf16 로드 | 맥(MPS)에서는 그대로 돌지 않는다. 장치 선택과 autocast 경로를 바꿔야 한다 (§4) |
| D21 | 의존성 (논문 주석 없음) | `pyproject.toml`: `torch==2.2.0`, `transformers` fork, `diffusers==0.30.3`, `peft==0.11.1`, `tensorflow==2.15.0`, `dlimp` fork, flash-attn은 설치 후 추가 | 논문의 A100 측정은 torch 2.2.0 고정이 조건 (`LIBERO.md:86–88`) |

## 2. 라이선스와 출처

| 항목 | 라이선스 | 확인 |
|---|---|---|
| `moojink/openvla-oft` (코드) | MIT | GitHub API와 `LICENSE` (2025) |
| `moojink/transformers-openvla-oft` (bidirectional fork) | Apache-2.0 | GitHub API (push 2025-02-25, 스타 12) |
| `Lifelong-Robot-Learning/LIBERO` (시뮬레이터) | MIT | GitHub API (push 2025-03-15). 이 저장소는 설치하지 않음 |
| `moojink/dlimp_openvla` | 표시 없음 | GitHub API (push 2024-07-12). 라이선스 확인 필요 |
| `moojink/openvla-7b-oft-finetuned-libero-spatial` (가중치) | MIT (HF 카드) | HF API. 저장 용량 약 15.9 GB. 받지 않음 |
| 기반 OpenVLA 가중치 | HF 카드 MIT. 기반 모델이 Llama 2 파생이라 라이선스 조건이 붙을 수 있음 (미확인) | 받지 않음 |

**논문:** arXiv abs 페이지의 "view license" 링크는 `creativecommons.org/licenses/by/4.0/`(CC BY 4.0)입니다. PDF 본문에는 ©, copyright, All rights reserved 문구가 없습니다. 그러나 그림 대부분이 실로봇 사진, 로봇 장면, 시뮬레이션 장면입니다. 공개 산출물에는 논문 그림을 넣지 않았습니다(공개본에는 이 저장소가 표 값에서 다시 그린 그래프만 씁니다).

## 3. 재현에 필요한 것 (정적 확인)

- 추론: 체크포인트 한 개(약 15.9 GB, bf16), 장면 입력(224 px 두 장과 자세), CUDA GPU를 전제로 한 `run_libero_eval.py`, LIBERO 시뮬레이터(MuJoCo).
- 미세조정: README 기준 GPU 27–80 GB, 배치 8/GPU에서 약 62 GB(`LIBERO.md:96`), 데이터 RLDS 약 수십 GB(받지 않음).
- 실로봇 ALOHA: 하드웨어(ViperX 두 팔, 카메라 세 대). 이 저장소는 실로봇 설정 파일과 실행 스크립트(`experiments/robot/aloha/`)만 있습니다.

## 4. 구현 가능성 (Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)

**요약: 표 재계산과 코드 읽기는 끝냈습니다. 추론은 조건부입니다. 미세조정, 사전학습, 실로봇은 이 맥에서 불가능합니다.**

| 작업 | 판정 | 근거 |
|---|---|---|
| 표 재계산과 검정 | 완료 | `analysis/recompute.py`, `03` |
| 공식 코드 정적 대조 | 완료 | §1 |
| 추론 한 청크 (LIBERO 체크포인트) | **조건부**: 포팅 2–4일 추정 (측정 전, 추정) | 장치는 `openvla_utils.py:39`에서 CPU로 떨어지므로 MPS 분기가 필요하다. `openvla.py:79`의 `autocast("cuda")`는 추론에서 꺼져 있으나 확인이 필요하다. bf16 경로가 MPS에서 되는지 torch 2.2.0 고정 조건에서 불확실하다. 메모리는 가중치만 약 15 GB이므로 24 GiB에서 빠듯하다. fork 설치는 승인 뒤 |
| LIBERO 4개 스위트 × 3시드 평가 | 불가 (추정) | 상한 기준 시드당 약 8만 회 추론 호출 (스위트별 500 × 최대 스텝 ÷ 8). 호출당 지연을 재기 전에는 시간을 추정할 수 없다. 시뮬레이터와 추론이 모두 필요하다 |
| 미세조정 (LoRA, LIBERO) | 불가 | README 기준 최소 27 GB GPU 메모리. 이 맥은 CUDA가 없다 |
| 전체 미세조정, 사전학습 | 불가 | 8 GPU 이상, 수일 |
| ALOHA 실로봇 | 불가 | 하드웨어 필요 |
| 확산 T=1 0.0%의 원인 확인 | 불가 | 원자료와 로그가 없다. 코드를 실행해 확인하려면 GPU가 필요하다 |

노력 추정(추정): 추론만 맥에서 되게 하는 포팅은 2–4일입니다. 측정 전이라 불확실성이 큽니다. fork 설치와 의존성 맞추기가 시간의 대부분을 차지할 것입니다.

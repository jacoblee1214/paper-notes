# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- **저장소:** `OpenDriveLab/NativeMEM` (공식, 프로젝트 페이지의 Code 링크). 커밋 `c599f81f80f5fc3b9c2c7c947f6fb2dbcd6f0315` (2026-10-19, "upload method.png"). 얕은 복제를 `repo/`에 두었습니다(파일 107개, 약 5 MB). 같은 계정 소유로 공식 README가 링크한 시뮬레이션 평가 저장소 `Ziyeeee/NativeMEM-RMBench` 커밋 `36cf2ca95bbea80b6f4edc1b74c9bc169e58e358` (2026-10-16)도 얕게 받아 `data/rmbench/`에 두고 평가 쪽 코드를 읽었습니다.
- **정적 읽기만 했습니다.** 코드를 실행, import, 설치하지 않았습니다. 체크포인트와 데이터는 받지 않았고(Hugging Face API 메타데이터의 크기만 확인), 복제한 내용은 신뢰하지 않는 데이터로 다뤘습니다.
- 읽은 파일: `README.md`, `pyproject.toml`, `src/openpi/models/{mem_tokenizer,mem_tokenizer_config,nativemem,nativemem_config,nativemem_rtc,siglip_encoder,tokenizer}.py`, `src/openpi/training/{config_mem_tokenizer,config_nativemem,data_loader_mem,data_loader_mem_tokenizer,weight_loaders,optimizer}.py`, `scripts/train_{mem_tokenizer,nativemem}.py`, `examples/robotwin/convert_raw_to_lerobot_mem_tokenizer.py`, 평가 저장소의 `policy/nativemem/{deploy_policy.py,deploy_policy.yml,pi_model.py,memory_tokenizer.py}`, `envs/click_button*.py`, `task_config/*.yml`.
- 코드는 "openpi"(Physical Intelligence)의 포크 위에 기억 토크나이저(`MemTokenizerModel`), 기억을 받는 정책(`NativeMEM`, `NativeMEMRTC`), 두 단계 학습 스크립트, 데이터 파이프라인을 더한 구조입니다. 새로 쓴 핵심 파일은 위 목록의 앞 열 개 안팎입니다.

## 1. 설치 요구사항과 라이선스

- **환경:** Python 3.11 이상, `jax[cuda12]==0.5.3`, `flax==0.10.2`, `torch==2.7.1`(openpi의 PyTorch 경로용), `lerobot<=0.3.3` 등 (`pyproject.toml`, `uv.lock` 5,074줄). README: "CUDA-capable NVIDIA GPU for practical training". 도커 파일은 NVIDIA 컨테이너 도구를 전제합니다.
- **PyTorch 경로:** `src/openpi/models_pytorch/`의 `pi0_pytorch.py`는 openpi 원본의 PyTorch 포팅이며 기억 토큰 입력 경로는 없습니다(grep으로 확인).
- **라이선스:** `LICENSE`는 Apache-2.0이고 `LICENSE_GEMMA.txt`(Gemma 이용 약관)가 함께 있습니다. README 라이선스 절은 두 파일을 가리킵니다. 평가 저장소는 MIT(RMBench를 따름)입니다. 체크포인트(PaliGemma 기반 가중치 포함)의 이용 조건은 Gemma 약관을 따를 가능성이 있습니다. 재배포 전에 확인이 필요합니다.
- **체크포인트 규모:** Stage 1 토크나이저 6.8 GB, 시뮬 과제별 정책 11.3 GB × 5 = 56.7 GB, 합계 약 63.6 GB(파일 199개, API 메타데이터).

## 2. 논문 ↔ 코드 대조 (D 항목)

판정: **일치**는 논문 값과 코드 값이 같음, **차이**는 다름(또는 논문에 없는 기능·값), **미확인**은 저장소에서 찾지 못함.

| 번호 | 항목 | 논문 | 코드 | 판정 | 영향 |
|---|---|---|---|---|---|
| D1 | Stage 1 배치 크기 | 배치 16 (부록 A) | `TrainConfig`의 클래스 기본값은 16이지만(`config_mem_tokenizer.py` L36) 이름 있는 설정 `mem_tokenizer_pretrain`은 `batch_size=64`, `num_workers=32` (L100–101). 학습 50,000 스텝, 학습률 5e-5(워밍업 1,000, 코사인 감쇠 끝값 5e-5라 사실상 일정)는 논문과 같음 | **차이 (해석 가능)** | 4 GPU × 장치당 16 = 64라면 같은 설정입니다. 논문이 어느 쪽인지 적지 않았습니다. 총 샘플 수는 800,000 대 3,200,000으로 4배 차이 |
| D2 | Stage 1 한 샘플의 프레임 수 | "첫 프레임과 최근 8프레임 (간격 8~16 스텝)", 현재 프레임도 포함 (Sec. 3.2, 4.1) | `history_seq_len=8`, 인덱스는 `[0] + 7개(현재에서 stride 간격으로 거슬러) + [현재]` = 9프레임 (`data_loader_mem_tokenizer.py` L402–406, 시퀀스 길이 `history_seq_len + 1`). 기억 토큰은 BOS 1 + 9 × 3시점 = 28 | **차이 (집계 방식)** | 첫 프레임 + 7 + 현재라서, "최근 8프레임"이 현재 프레임을 포함하는 수라면 일치. 본문은 현재 프레임을 별도로 더한다고도 적어 열 개로 읽힐 수 있음 |
| D3 | Stage 1 창의 폭과 간격 추출 | 간격 8~16 스텝 | 간격은 프레임마다 `default_rng(0)`으로 한 번 뽑아 고정(L375), 학습 내내 같은 간격. 창 폭은 최대 7 × 16 = 112 스텝 + 첫 프레임. 에피소드 끝 가까운 프레임은 제외(`local_index + 0.75 × 행동 지평 ≤ 길이`, L378–381) | **차이 (논문에 없는 세부)** | Stage 1 토크나이저가 본 이력은 첫 프레임 + 최대 약 112스텝입니다. Stage 2와 추론은 에피소드 전체(최대 수백 스텝)를 입력으로 쓰는데(D8), 이 길이는 Stage 1에서 보지 않습니다 |
| D4 | 기억 토큰의 위치 번호 | 서술 없음 | Stage 1: 기억 토큰 위치 = 프롬프트 끝 + 1 + `history_positions` × V + 시점 (`mem_tokenizer.py` L243–297). `history_positions` = `프레임 인덱스 // stride`로 첫 프레임은 0, 이후 실제 경과(간격 단위)를 반영. Stage 2와 추론: `cumsum(input_mask) − 1`로 빈틈없는 연속 번호 (`nativemem.py` L241, L274) | **차이 (논문에 없음)** | Stage 1은 경과 시간에 비례한 위치 번호, Stage 2는 연속 번호를 씁니다. Stage 2에서 VLA를 풀어 미세조정하므로 적응할 수 있지만, 두 단계의 위치 부호 규약이 다르다는 점은 본문에 없습니다 |
| D5 | Stage 1의 본문에 없는 장치 | "복원 손실 없음, 행동 손실만" (Sec. 3.2) | 행동 손실만 쓰는 점은 일치(흐름 정합 MSE, `mem_tokenizer.py` L362). 추가 기능: 현재 프레임 패치 무작위 제거(`current_patch_drop_prob`, 기본 0.0, 코사인 워밍업 코드 있음), 기억 토큰 간 코사인 유사도 로그(`mem_token_sim`), 지수이동평균 0.9999, 학습 때 이미지 증강(`preprocess_observation(train=True)`: 밝기 0.3, 대비 0.4, 채도 0.5의 색 변환 포함) | **일치 (손실), 차이 (장치)** | 패치 제거 기능은 꺼져 있어 기본 설정의 결과에는 영향이 없습니다. 증강은 Stage 1 입력에만 적용되고 캐시한 토큰(Stage 2)은 증강 없이 만들어집니다 |
| D6 | 기억 질의 토큰과 학습 대상 | 질의 토큰 q_mem을 패치 앞에 붙이고 VLA 비전 인코더로 초기화한 기억 인코더에 통과 (식 5), 선형 투영 (식 6). VLA 얼림 | `SiglipEncoder`: 영으로 초기화한 `cls` 파라미터를 패치 토큰 앞에 붙여 SigLIP 전체(27층)와 마지막 LayerNorm을 거친 뒤 그 위치 출력만 사용(`siglip_encoder.py`). 학습 대상은 `encoder/*`, `memory_proj_in/*`, `mem_bos` 세 묶음 뿐이고 나머지(PaliGemma LLM, 행동 전문가, 행동 입출력·시간 MLP)는 얼림 (`mem_tokenizer_config.py` L97–103). 기억 인코더 가중치는 π0.5의 SigLIP 가중치로 초기화(`MemTokenizerWeightLoader`) | **일치** | 질의 토큰의 초기값이 영인 점은 본문에 없지만 다른 항목은 서술과 같음 |
| D7 | 기억 프레임 간격 (Stage 2, 추론) | "10 제어 스텝마다 기억 갱신" (Sec. 4.1) | Stage 2 데이터: 샘플마다 간격을 8~16에서 균등 난수로 뽑아 현재 프레임에서 에피소드 시작까지 그 간격으로 거슬러 올라가고 첫 프레임을 더함(`data_loader_mem.py` L37–44). 평가 저장소 기본값 `memory_interval: 12`, `pi0_step: 48`(50스텝 중 48 실행) (`deploy_policy.yml`) | **차이** | 평가 기본 간격이 12이고 학습 간격은 8–16 분포입니다. 본문의 "10스텝"과 다릅니다. 시뮬과 실로봇 설정이 다를 수 있으나 본문에 구분이 없습니다. 같은 10스텝 설정을 시뮬에서 쓰면 학습 분포의 가운데 값(12)과 약간 다릅니다 |
| D8 | 기억 길이 상한 | "160프레임 이상, 분 단위", 5,000프레임까지 용량 (그림 1, 4(a)) | 기억은 토큰 수 상한 `memory_tokenizer_max_len`(기본 256)에 맞춰 패딩하거나 **뒤쪽 최대 길이만 남기고 앞쪽을 자릅니다**(`tokenizer.py` L79, 에피소드 시작의 첫 프레임이 가장 먼저 잘림). 과제 접두어별 상한: 시뮬 과제 256, `cover_blocks` 512, 실로봇 `arx/click_button` 256, `arx/put_cube` 256, `arx/scan_code` 512 (`config_nativemem.py` L29–40) | **차이 (논문에 없는 상한)** | 3시점이면 256토큰 = 85프레임, 512토큰 = 170프레임입니다. 시뮬 과제는 단계 한도(250–1,500)와 간격 12에서 최대 21–126프레임이어서 상한 안에 들어갑니다(`03` V16). 160프레임 이상은 실로봇 `scan_code` 설정(512) 규모입니다. 상한을 넘으면 첫 프레임 앵커가 사라집니다 |
| D9 | Stage 2 학습 설정 | 20,000 스텝, 배치 32, VLA 전체 미세조정 (Sec. 4.1, 부록 A) | 20,000 스텝, 배치 32 (기본), 동결 필터 없음(전체 미세조정, `nativemem_config.py`의 LoRA 변형이 아닌 경우). 학습률 최대 2e-5(워밍업 1,000, 코사인으로 1e-6까지), 지수이동평균 0.99, AdamW(β2 0.95, 가중치 감쇠 1e-10, 기울기 절단 1.0) | **일치 (공개된 값), 논문에 없음 (학습률 등)** | 논문에 학습률과 지수이동평균이 없습니다. 학습 손실은 π0.5와 같은 흐름 정합 MSE(`nativemem.py` L250) |
| D10 | Stage 2 초기화 | "사전학습 단일 프레임 정책에서 초기화하고 W_mem과 b_mem을 1단계에서 로드" (Sec. 3.3) | `NativeMEMWeightLoader`가 Stage 1 체크포인트 한 개에서 전부 불러옴. 그 체크포인트의 LLM·행동 전문가·SigLIP은 π0.5 기본 가중치(동결)이므로 결과적으로 π0.5 기본 가중치 + Stage 1의 `memory_proj_in`, `mem_bos` | **일치** | — |
| D11 | Stage 1 데이터 | 시뮬과 실로봇 혼합, 표준 조작 + 기억 과제, 약 4,000 궤적 (부록 A) | 기본 설정은 `data/sim.h5` 하나만 읽고 실로봇용 `arx.h5` 줄은 주석 처리(`config_mem_tokenizer.py` L93–94). 데이터 준비 스크립트는 `sim`, `trossen`, `arx` 세 유형을 지원. Stage 1 학습 데이터 자체는 저장소에 없음 | **미확인** | 4,000 궤적의 구성(과제별 개수, 평가 과제 포함 여부)을 코드로 확인할 수 없습니다 (`03` §7) |
| D12 | RTC(실시간 청킹) | 표 2 기준선 "π0.5 + RTC". Ours의 RTC 사용 서술 없음 | `NativeMEMRTC`(`nativemem_rtc.py`): 학습 때 행동 접두부를 지연 길이(0–7, `max_delay=8`)만큼 깨끗한 값으로 주고 손실을 나머지에만 계산, 샘플링에서 접두부 고정. 공개 학습 설정 `nativemem_pi05`는 `rtc=False`(기본). 저자 계정의 `openpi_rlt`는 "NativeMEM과 RTC" 위의 구현이라고 설명 | **차이 (서술 부재)** | 실로봇 정책이 RTC 학습을 썼는지 본문과 설정으로 확인 불가. 표 2의 π0.5 + RTC와 비교할 때 중요 |
| D13 | 실시간 비동기 기억 갱신 | "토크나이저가 VLA와 독립적으로 높은 주기로 갱신하고 VLA는 최신 큐를 사용" (Sec. 3.3), 보충 영상의 지연 대시보드 | 평가 저장소의 시뮬 클라이언트는 같은 프로세스에서 기억 프레임(간격 배수)마다 `memory_tokenizer(...)`를 동기 호출하고 행동 요청 때 그 프레임들의 토큰을 이어 붙임(`pi_model.py`). 비동기 큐, 지연 측정 코드, 그림 4(a)를 만든 스크립트는 없음 | **미확인 (비동기·지연 측정)** | 효율 주장(그림 4(a))과 실시간 갱신은 이 저장소로 시험할 수 없습니다 |
| D14 | 추론 때 기억 정렬 | Eq. 7: 선택한 프레임 인덱스의 토큰을 이어 붙임 | 클라이언트가 현재 프레임에서 `memory_interval` 간격으로 0까지, 그리고 0을 더한 목록을 만듦(학습의 `_get_memory_indices`와 같은 규칙). 청크 시작이 간격의 배수가 아니면 오류(`pi0_step`이 `memory_interval`의 배수여야 함) | **일치 (학습-추론 정렬)** | 학습 때 간격이 난수 8–16인데 추론은 고정 12 한 값입니다(D7) |
| D15 | 공개 범위: 설정과 체크포인트 | 표 1–8, LIBERO(표 3), 실로봇(표 2) | 학습 설정은 `nativemem_pi05`(ALOHA 시뮬, `gripper_type="sim"`) 하나. LIBERO용 설정(표 3), 실로봇용 설정과 데이터(표 2), 처음 보는 조명(표 5) 설정은 없음. 체크포인트는 시뮬 5개 과제(`click_button`, `click_button_rand`, `observe_and_pickup`, `put_back_block`, `swap_blocks`)뿐이며 표 8의 Rearrange Block, Swap T 체크포인트는 목록에 없음(설정의 상한 표에는 해당 접두어 있음) | **미확인** | 표 1의 시뮬 5과제 중 4과제(Hard 포함)는 공개 체크포인트로 평가 경로가 있고, 표 2, 3, 5, 8의 설정은 공개되지 않음 |
| D16 | 기준선·절제 코드 | HAMLET*, MEM-short*, MemER, Native History, w/o Stage 1(평균 풀링), Unfrozen VLA | 저장소에 없음(평균 풀링 기억 토큰 코드 없음, `mem_tokenizer_config.py` 동결 필터 안에 주석 처리된 LLM 줄만 있음). Mem-0 정책은 평가 저장소의 `policy/Mem-0` | **미확인** | 표 4, 6과 기준선 수치는 공개 코드로 재현할 수 없음 |
| D17 | 라이선스 | — | Apache-2.0(코드), Gemma 약관 파일 동봉. 모델 카드 apache-2.0 | **일치(파일 기준)** | 가중치 사용은 Gemma 약관 확인이 필요할 수 있음 |

### 2.1 논문 주장의 코드 쪽 확인 (일치한 부분)

- **원래 행동 손실로 학습:** Stage 1과 Stage 2 모두 π0.5와 같은 흐름 정합 손실(시간 Beta(1.5, 1), 노이즈 보간, 속도 MSE)입니다. 기억 전용 보조 손실은 없습니다.
- **VLA 얼림(Stage 1):** 학습 가능한 묶음이 `encoder/*`, `memory_proj_in/*`, `mem_bos`로만 지정됩니다. 행동 전문가와 LLM은 학습 대상이 아닙니다.
- **프레임·시점당 토큰 1개:** 3개 시점마다 인코더를 따로 통과시키고 질의 위치의 벡터 하나(폭 1,152)만 가져옵니다(`mem_tokenizer.py`의 `_encode_hist_cur`, 변환 스크립트의 `_extract_mem_tokens`). 캐시는 float32입니다.
- **캐시 후 미세조정:** 변환 스크립트가 전 프레임의 `mem_token`을 LeRobot 데이터셋에 저장하고(JPEG 왕복과 resize-with-pad를 Stage 1과 같게 적용), Stage 2 데이터 로더가 선택한 인덱스의 토큰을 모아 `memory`로 넣습니다. `stop_gradient`로 토크나이저를 고정합니다.
- **입력 구성:** 이미지 토큰, 프롬프트, BOS, 기억 토큰의 순서이고 기억 토큰 사이 어텐션은 양방향(`memory_insert_type="self"`)입니다. 기억 토큰은 사전 채움(prefix)에 들어가고 행동 전문가가 이를 읽습니다.
- **효율의 구조적 근거:** 기억 토큰 수가 `프레임 수 × 시점 수`에 선형이고, Stage 2에서 토큰 계산이 사전에 끝나므로(캐시) 학습 비용이 늘지 않는다는 서술과 구조상 맞습니다.

## 3. 평가 환경 메모 (NativeMEM-RMBench)

- 과제와 단계 한도(`task_config/_eval_step_limit.yml`): `observe_and_pickup` 250, `put_back_block` 500, `click_button` 600, `swap_T` 600, `rearrange_blocks` 700, `swap_blocks` 1,000, `cover_blocks` 1,500.
- **Click Buttons (Hard)에 해당하는 환경은 `click_button_rand`로 보입니다.** 8색 중 한 색을 골라 세 단추를 모두 같은 색으로 만들고(색 구분 없음), 단추를 x 좌표 순으로 정렬해 번호를 붙이며 `demo_rand` 설정(D435 카메라, 무작위 배경·조명)을 씁니다. 반면 `click_button`은 파랑, 분홍, 노랑 세 색의 단추를 씁니다. 논문 표 1의 Hard 정의("색 구분 제거")와 맞는 것으로 읽었습니다. 환경 이름 대응은 저장소 파일 기준의 추정입니다.
- 서버-클라이언트 구조: 정책은 JAX(Python 3.11), 시뮬은 SAPIEN + CuRobo(Python 3.10, CUDA 전제). 두 프로세스가 TCP(JSON)로 통신합니다. 서버는 인증이 없다고 README가 경고합니다.
- 평가 클라이언트는 `instruction_type: unseen`을 기본으로 씁니다. 표의 평가가 seen/unseen 중 무엇이었는지 본문에 없습니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 통합 메모리 약 24 GiB, MPS, CUDA 없음)

**논문 수치의 재현은 이 맥에서 불가능합니다. 구조 재구현의 축소판은 가능하지만 아직 하지 않았고, 이번 분석에서는 코드 정적 대조와 표 재계산까지 했습니다.**

| 수준 | 판정 | 근거 |
|---|---|---|
| Stage 1·Stage 2 학습 (공식 설정) | 불가 | 약 3B 파라미터 모델의 전체 미세조정(Stage 1은 H100 4장 16시간, Stage 2는 H100 2장 5시간, 논문 부록 A). AdamW 상태까지 24 GiB에 들어가지 않고, 코드는 `jax[cuda12]` 전제 |
| 시뮬레이션 평가 (표 1) | 불가 | SAPIEN, CuRobo, CUDA GPU 필요. 정책 서버도 11.3 GB 체크포인트와 JAX가 필요 |
| 실로봇 평가 (표 2) | 불가 | ARX/Trossen 계열 로봇과 데이터, 정책이 공개되지 않음 (D15) |
| 공개 체크포인트로 추론 | 조건부 | 새 환경 설치(JAX CPU, flax 0.10.2, orbax, lerobot 등)와 체크포인트 다운로드(Stage 1 6.8 GB + 과제당 11.3 GB), 공식 코드 실행 승인이 필요합니다. 24 GiB에서 fp32 약 11 GB는 올라가지만 CPU 속도는 측정하지 않았습니다. 시뮬 환경이 없으므로 추론 결과를 확인할 시연 데이터(LeRobot)가 따로 필요합니다 |
| 토큰 수·길이 산술, 표 재계산 | 가능 (완료) | `analysis/recompute.py` (수 초, numpy·scipy) |
| 구조 재구현 (소형 stand-in VLA + 질의 토큰 압축기, 얼린 정책 + 행동 손실) | 가능 | PyTorch로 작은 비전 인코더와 얼린 소형 정책을 만들고 "행동 손실로 학습한 1토큰 요약 대 평균 풀링 요약"을 합성 기억 과제(예: 처음 프레임의 단서가 나중 행동을 정함)에서 비교하는 구상입니다. 논문 설정과 규모가 달라 논문 수치와는 직접 비교할 수 없습니다. 이번에는 하지 않았습니다. 논문 핵심(사전학습 VLA의 토큰 공간에 정렬)을 작은 모델이 대신하지 못하므로, 결과는 "행동 손실만으로 요약 토큰이 정보를 보존하는가"라는 더 좁은 질문의 답이 됩니다 |
| π0.5를 PyTorch로 옮기고 MPS에서 기억 경로 추가 | 조건부 | openpi의 PyTorch 포팅(기억 경로 없음)에 기억 토큰 입력을 더하는 작업. 가중치 다운로드와 새 설치, 공식 코드 실행 승인이 필요합니다. MPS에서 3B 추론은 가능할 수 있으나 속도는 미측정 |

**노력 추정(측정 전):** 소형 stand-in 구현은 3–5일. 공개 체크포인트 추론 환경 구성은 1–3일(승인과 다운로드 이후). 사실상 논문 규모의 재현은 GPU 클러스터와 실로봇이 필요합니다.

**이 맥에서 이미 끝낸 것:** 표·그림 수치 재계산과 검정(`03`), 토큰·길이 산술, 코드의 설정값 대조(`04` D1–D17).

## 5. 재실행 (정적 대조의 근거)

- `repo/`는 `git clone https://github.com/OpenDriveLab/NativeMEM repo && git -C repo checkout c599f81f80f5fc3b9c2c7c947f6fb2dbcd6f0315`로 다시 받을 수 있습니다. 평가 저장소는 `git clone https://github.com/Ziyeeee/NativeMEM-RMBench data/rmbench && git -C data/rmbench checkout 36cf2ca95bbea80b6f4edc1b74c9bc169e58e358`입니다.
- 이 문서의 줄 번호는 위 커밋 기준입니다. `analysis/recompute.py`의 V16은 평가 저장소의 단계 한도와 기억 간격 12, 접두어별 상한을 입력으로 씁니다(저장소를 읽지는 않고 값을 옮겨 적은 계산입니다).

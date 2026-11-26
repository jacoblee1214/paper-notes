# 04. 공식 코드와 논문 대조

대상: https://github.com/ahadjawaid/nac, **커밋 `2e2d9db3d6abde0415f77500f41ff292262a1343`**(2026-06-28, 기본 가지 main, 얕은 복제). 정적으로만 읽었고 실행, import, 설치는 하지 않았습니다. 이 저장소의 `repo/`를 다시 만들려면 `git clone https://github.com/ahadjawaid/nac repo && git -C repo checkout 2e2d9db3d6abde0415f77500f41ff292262a1343`입니다. 논문은 v2(2026-09-25)이고, 코드의 마지막 커밋은 v2보다 약 3개월 앞섭니다. 즉 이 코드는 v1(2026-06-19) 시기의 공개본에 가깝고, 아래 차이 일부는 v1과 v2 사이에 논문이 바뀐 데서 생겼을 수 있습니다.

번호 체계: **D**는 논문과 코드의 차이 또는 논문에서 확인되지 않는 코드 쪽 사실입니다.

## 1. 구성

| 경로 | 내용 |
|---|---|
| `nac/tokenizer/nac/` | NAC 본체: `tokenizer.py`(NACTok 학습 손실·토큰화), `encoder/`(SEANet, 양자화기: `quantization/fsq.py`의 `MScaleFSQ`, `vq.py`의 `MScaleRVQ`), `decoder/`(Vocos 백본, ISTFT 헤드, DAC 판별기, 손실) |
| `nac/policy/` | `nacpolicy.py`(오프셋 코드북, BOS, 거친-먼저 순서)와 기준선 정책(bin, fast, oat, vqvla, diff) |
| `nac/tokenizer/{bin,fast,oat,vqvla}` | 기준선 토크나이저 (OAT와 VQ-VLA는 코드를 가져와 포함) |
| `nac/workspace/` | 토크나이저·정책 학습 루프 (accelerate + hydra) |
| `nac/env_runner/`, `nac/env/` | LIBERO, RoboMimic 평가 러너 |
| `nac/config/model/{libero,robomimic}/` | 학습 설정 (`train_nactok.yaml`, `train_nacpolicy.yaml` 등 12개씩) |
| `scripts/` | 학습 진입점(`run_workspace.py`), 평가(`eval_policy_sim.py`), 데이터 변환 |
| `tests/` | 토큰 배치(거친-먼저) 단위 시험 두 개 |
| `third_party/LIBERO` | LIBERO 포크 (서브모듈) |

## 2. 차이 (D)

| 번호 | 항목 | 논문 | 코드 | 비고 |
|---|---|---|---|---|
| **D1** | **양자화기** | "2-scale RVQ"(표 4), 식 3–4와 알고리즘 1: 코드북 k-means 초기화 + EMA 갱신, 커밋 손실 λcommit = 10³ | **공개 설정은 `quantizer_type: fsq`, `fsq_levels: [8, 8, 4, 4]`**(8·8·4·4 = 1024코드). 스테이지마다 512 → 4 선형 투영, 코드북 없음, k-means/EMA 없음. `MScaleFSQ.forward`는 `penalty = 0`을 돌려주어 커밋 손실 항(× 1000)이 0 | RVQ 구현(`MScaleRVQ`)은 저장소에 있으나 `train_nactok.yaml`(LIBERO, RoboMimic 모두)의 기본이 FSQ입니다. 표 1·2의 수치를 어떤 양자화기로 얻었는지 저장소만으로는 알 수 없습니다. 같은 설정 폴더의 OAT 기준선도 FSQ [8, 8, 4, 4]를 쓰므로, 공개 설정 기준으로는 NAC와 OAT가 같은 양자화기 계열입니다 |
| **D2** | 인코더 | "fully convolutional SEANet"(§2.3, §3.2) | SEANet 인코더가 **LSTM 2층**(`lstm=2`)을 끝에 포함 | 시간 방향 순환 층이 들어 있어 "완전 합성곱"이라는 서술과 다름 |
| **D3** | 생성기 손실 | 식 (5): λrec·MSE + λcommit·커밋 + λadv·적대 (45, 10³, 1) | `45 × MSE + 1000 × commit + DAC LSGAN 적대 항 + DAC 특징 매칭 항`(둘 다 계수 1) | 특징 매칭 항이 논문 식에 없음. λrec = 45, λadv = 1(적대 항은 계수 1)은 일치. 위 D1 때문에 커밋 항은 FSQ에서 0 |
| D4 | 양자화 보조 설정 | 해당 없음 | `quantize_dropout: true`, `rand_num_quant: [4, 4]`, `adanorm_num_embeddings: 4`, `ema_decay: 0.95` | FSQ 경로는 `rand_quantize_dropout_index`를 무시(`**_`), RVQ 경로에서만 의미. 오디오 코덱 설정의 잔재로 보임 |
| **D5** | 토크나이저 학습 일정 | 학습률 2e-4, 코사인 스케줄, 50k 스텝(표 4) | 학습률 2e-4, 코사인, 50k(LIBERO) 일치. **워밍업 15,000스텝(전체의 30%)**, 판별기·생성기 AdamW 둘, **EMA 가중치(감쇠 최대 0.9999)로 체크포인트를 저장해 사용**(`from_checkpoint`가 EMA 가중치를 읽음) | 워밍업과 EMA는 논문에 없음 |
| **D6** | RoboMimic 설정 | "모든 실험은 행동 지평 32, 시연 500개, 학습 50k 스텝"(B.1) | RoboMimic 설정은 **`num_demo: 800`**(데이터 `mt4_N800`)이고 **토크나이저 `num_steps: 25000`**(체크포인트 6250마다). 정책 학습은 50,000 스텝 | RoboMimic 설정은 OAT, VQ-VLA 토크나이저도 25,000스텝이고 모든 RoboMimic 설정이 800시연입니다. 표 4·5의 "50k, 500" 서술과 다름 |
| D7 | 정책 학습의 세부 | 학습률 5e-5/1e-5, 배치 256, 50k, d = 256, L = 4, H = 4, 드롭아웃 0.1, top-k 10, 온도 1.0 (표 5) | 모두 일치. 추가로 **EMA, 그래디언트 클리핑 1.0, 학습률 상수 + 워밍업 100스텝(FAST와 Diffusion Policy는 200)**, AdamW β = (0.9, 0.95), 가중치 감쇠 0, bf16 허용 | 워밍업 200 대 100: §4.1의 "same training schedule"과 약간 다름. 효과는 작을 것으로 보이나 확인되지 않음 |
| **D8** | 평가 프로토콜 | 시뮬 "시드 8개, 과제당 50회"(표 2), ±의 정의 없음 | 러너의 `n_test`: LIBERO-10 **500**(10과제 × 50), RoboMimic **200**(4과제 × 50, mt4 = lift, can, square, tool_hang). `eval_policy_sim.py`는 `--num_exp`(기본 **1**, README 예시 5)번 반복하며 반복마다 환경 시드를 `1000 + 반복 × n_test`로 옮기고 **같은 체크포인트**를 평가. 평균, 표본 표준편차(ddof = 1), 표준오차를 JSON에 저장. 최대 에피소드 스텝 550(LIBERO), 800(RoboMimic). 학습 시드는 정책 42, 토크나이저 3407로 고정 | 논문의 "시드 8개"가 평가 시드 반복인지 학습 시드 반복인지 코드로는 확정되지 않습니다(기본 흐름은 평가 시드). 체크포인트 선택은 `save_last_ckpt`(마지막)이고 상위 k 저장은 `k: 0`. 학습 중 롤아웃은 12,500스텝마다 돌지만 보고 값이 어느 체크포인트인지는 논문에 없음. RoboMimic 과제 구성은 논문에 없음 |
| D9 | FAST, Bin 기준선 | FAST는 "pretrained FAST"(표 4), 토큰 36(표 3) | FAST: `physical-intelligence/fast`를 불러온 뒤 **학습 행동 데이터로 BPE를 다시 맞춤**(`fit`, 스케일 10, 어휘 1024). 정책 시퀀스 최대 길이 128. 샘플된 토큰이 유효하지 않으면 계수를 0으로 채우거나 자름. Bin: [−1, 1] 1024구간 | 표 3의 36토큰이 평균인지 최대인지 코드로도 알 수 없음. 1024어휘로 다시 맞춘 FAST는 공개 FAST+(2048)와 다름 |
| D10 | OAT, VQ-VLA 기준선 | 표 4: OAT는 FSQ + 레지스터 12개, 인코더 L = 3, 디코더 L = 10, 학습률 2e-4. VQ-VLA는 12그룹 VQ, VQ 가중치 5, 학습률 5e-5 | 모두 일치 (OAT FSQ [8, 8, 4, 4], 레지스터 12, 깊이 3/10, VQ-VLA 12그룹 × 1024, 가중치 5.0) | 일치 항목 |
| D11 | 정책 구조 | "VLA", 관측 인코더 + 작업 조건 (§3.1, §3.3) | **VLM 없음.** ResNet18(GroupNorm, 특징 64, 랜덤 크롭 76 × 76)을 처음부터 학습, 작업은 `task_uid`(상태 입력)로 조건. 언어 프롬프트는 관측에 담기지만 정책 입력에는 안 쓰임 | 표 5의 "Transformer d = 256, 4층"과 일치. 논문 서술(언어 또는 작업 UID)과도 모순되지 않음 |
| D12 | 표 3 지연 | RTX 4090에서 Enc/Dec/Recon 측정 | 지연 측정 스크립트 없음 (`time`, `perf_counter`를 쓰는 평가·측정 코드 없음). 파라미터 수는 정책 초기화 때 출력만 함 | 표 3은 코드로 재현할 수 없음 |
| D13 | 공개 범위 | 실로봇 8과제(표 6) | 실로봇 코드, 환경 설정, 데이터 없음. 체크포인트 링크 없음. 릴리스 없음 (2026-11-26) | 실로봇 결과는 저장소 밖에서만 확인 가능 |
| D14 | 데이터 출처 | 500 시연 | LIBERO-10은 제3자 Hugging Face 데이터셋(`libero10_N500`, 약 3.5 GB, HF 메타데이터 값)과 LIBERO 포크 서브모듈, RoboMimic mt4는 별도 zarr(약 3.5 GB)와 robosuite 포크 | 설정 파일의 예시 토크나이저 체크포인트 경로 이름이 `..._mpd0_mrd1_dac1`로 **MRD를 켠 실행** 이름이고, 같은 파일의 기본 `use_mrd`는 false. 논문 "DAC 판별기"와 어느 쪽이 기본인지 불분명 |
| D15 | 정규화 | "선택적으로 차원별 선형 정규화"(§3.2) | `normalize_actions: false`(NAC 기본). 표 4의 "[−1, 1]"은 Bin 열 | LIBERO와 RoboMimic 행동은 이미 [−1, 1] 범위라 영향이 작아 보이나 확인되지 않음 |
| D16 | 시험 | 해당 없음 | 단위 시험은 토큰 배치(거친 토큰 먼저, 스케일 오프셋 되돌리기) 두 파일뿐. 토크나이저 재구성 품질, 양자화기 시험 없음 | |

## 3. 논문과 일치한 것

- 행동 지평 32, 실행 스텝 16, 관측 이력 2, 배치 256, d = 256, L = 4, H = 4, 드롭아웃 0.1, 정책 lr 5e-5, 비전 lr 1e-5, top-k 10, 온도 1.0(표 5).
- 토큰 배치: 224 샘플 → hop 28(7 × 4) → 8프레임, 스케일 0 풀링 2(4토큰), 스케일 1 풀링 1(8토큰), 합 12토큰, 토큰당 1024코드 → 120비트(표 3, `03` V5). 어휘 2 × 1024 + BOS = 2049, 스케일 ID 구간 [s·1024, (s+1)·1024), 거친 토큰이 먼저, 나머지 연산으로 인덱스 복원(식 6, 알고리즘 2, 단위 시험).
- 디코더: Conv1D 임베딩, ResNet 블록 + 어텐션, 12층 ConvNeXt(dim 768), ISTFT 헤드(n_fft 210, hop은 인코더 hop과 동기화)(§3.2).
- 손실 가중치 λrec = 45(MSE 계수), 판별기는 DAC 하나(MPD, MRD는 꺼짐), 학습률 2e-4, 50k 스텝(LIBERO).
- 기준선 설정(OAT, VQ-VLA, Bin, FAST 스케일 10, DP: DDIM 10스텝, 학습 100스텝, 어휘 1024)(표 4, 5).
- 평가 시행 수: LIBERO-10 과제당 50회, RoboMimic 과제당 50회(캡션의 "50 trials per task").

## 4. 이 맥에서 구현할 수 있는가 (Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**결론: 방법의 구조(토큰 배치, 다중 스케일 양자화, 손실 감도)는 이 맥에서 구현하고 검증할 수 있습니다. 논문 수치의 재현은 이 맥에서는 사실상 불가능하고, 실로봇은 저장소 밖 자원이 필요합니다.**

| 작업 | 판정 | 근거와 추정 |
|---|---|---|
| 표 재계산, 일관성 검사, 코드 정적 대조 | **완료** | `analysis/recompute.py`, 이 문서 |
| 손실 감도 토이 (스펙트로그램 손실의 부호·위상 둔감성) | **완료** | `analysis/toy_loss_sensitivity.py` (numpy/scipy, 약 1초). 한계는 `03` T1 |
| 토큰 배치와 다중 스케일 FSQ의 소형 재구현 (합성 행동 데이터) | **가능** | torch 2.15가 이미 설치되어 있어 새 설치가 필요 없음(CPU 또는 MPS). 추정 노력 1–2일, 수백 줄. 이번에는 하지 않음 |
| 공식 토크나이저 학습 (63M, 50k 스텝, 배치 256, LIBERO zarr 약 3.5 GB) | **조건부** | 새 패키지 설치(uv로 accelerate, hydra, wandb, zarr, torchaudio 등), 공식 코드 실행 승인, 약 3.5 GB 다운로드 승인이 필요합니다. M4 Pro의 MPS 학습 시간은 측정하지 않았고 `weight_norm`, LSTM, STFT의 MPS 호환도 확인하지 않았습니다(추정: 하루에서 수 일) |
| 공식 정책 학습 (ResNet18 두 카메라 + 4층 트랜스포머, 50k 스텝, 배치 256) | **사실상 불가** | 이미지 배치 256 × 카메라 2대 × 50k스텝 규모라 GPU를 전제한 설정으로 보임. 이 맥에서는 수 일 이상으로 추정(측정 전) |
| LIBERO 시뮬 평가 (500 롤아웃 × 최대 550스텝) | **불가에 가까움** | README는 `MUJOCO_GL=egl`을 쓰고 robosuite/LIBERO 포크 설치가 필요합니다. 이 맥에서의 렌더링 설정은 확인하지 못했습니다 |
| 표 3 지연 (RTX 4090) | 불가 | 측정 스크립트가 없고 GPU 값. M4 Pro에서의 상대 지연은 새 코드로 잴 수 있음 |
| 실로봇 8과제 (표 6) | **불가** | 로봇 기종, 카메라, 시연 수, 코드, 데이터가 논문과 저장소에 없음 |

- 새로 짜야 한다면 SEANet 인코더 + 다중 스케일 FSQ + Vocos 디코더(ISTFT) + DAC 판별기입니다. 공식 코드의 NAC 본체(`nac/tokenizer/nac/`)는 약 6.5천 줄이라 이를 읽고 필요한 부분만 추리는 편이 빠릅니다. 선택은 D1(FSQ)과 D2(LSTM)처럼 논문과 다른 부분을 어느 쪽으로 따를지에 달려 있습니다.
- 가장 싼 확인은 "FSQ 대 RVQ, LSTM 유무가 12토큰 재구성에 주는 차이"를 합성 데이터에서 보는 것입니다.

## 5. 라이선스

저장소는 Apache-2.0(LICENSE 파일, GitHub API도 apache-2.0)입니다. 일부 소스 파일(SEANet 등)의 머리말은 Meta Platforms 저작권과 해당 파일 LICENSE를 가리킵니다. OAT 저장소(라이선스 미지정)에서 코드를 가져왔다고 README가 밝힙니다. 논문은 arXiv CC BY 4.0입니다(`01` §0).

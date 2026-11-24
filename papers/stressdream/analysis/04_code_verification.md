# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: `CMU-IntentLab/StressDream` (공식. 프로젝트 페이지 stressdream.github.io 의 Code 링크, README 가 arXiv:2606.00267 을 가리킴). 커밋 `b16fa067d3d949ef09ad48a984c62ea8ecbbfe6d` (2026-06-02, "Add arxiv link"), 커밋 6개, 파일 103개, MIT 라이선스(저자 9인 명의). 얕은 복제를 `repo/` 에 두었습니다(체크포인트 포함 약 257 MB, `.gitignore`).
- **정적 읽기만 했습니다.** 코드를 실행, import, 설치하지 않았습니다. 가중치와 데이터는 받지 않았고, 저장소 안에 커밋된 작은 체크포인트(Dubins 77 MB)도 열어 보지 않았습니다.
- 읽은 파일: `README.md`, `wm_steer/regularizer.py`, `dubins/{steer,run_steering,env,generate_dataset,steering_config,config}.*`, `vista/{run_steering,rewards,steering_config}.*`, `ctrl_world/{run_steering,rewards,wm_helpers,config,steering_config}.*` 와 각 모듈 README. Vista 와 Ctrl-World 의 모델 본체(`vwm/`, `models/`)는 원 저장소에서 가져온 것이라 훑기만 했습니다.
- 코드 상수의 기준값(i.i.d. 가우시안)은 코드를 실행하는 대신 같은 식을 numpy 로 다시 구현해 계산했습니다(`analysis/recompute.py` V11).

## 1. 설치 요구사항, 라이선스, 자원

| 항목 | 내용 |
|---|---|
| 구성 | `dubins/`(전 과정: 데이터 생성, VAE·WM 학습, 조종, 시연 노트북), `vista/`(조종, 시연), `ctrl_world/`(조종, 공칭 생성), `wm_steer/regularizer.py`(공통 규제) |
| 환경 | Dubins: pip(`torch`, h5py, einops, ruamel.yaml, imageio, wandb, jupyter; cu118 extra-index). Vista: conda, Python 3.10, torch 2.9.1+cu128, xformers 0.0.33. Ctrl-World: conda, Python 3.11, torch 2.9.1+cu128, diffusers 0.36, transformers 4.57 |
| 자원 (README) | Vista: **VRAM 약 48 GB** (Qwen2.5-VL 보상까지는 약 80 GB), 샘플링만 40 GB, 20회 최적화 30–40분(H100). Ctrl-World: **VRAM 약 20 GB**, 체크포인트 3종 합 약 16.6 GB + Qwen3-VL-4B 약 8 GB. NVIDIA 드라이버 520 이상 |
| 체크포인트 | 저장소 안: Dubins VAE 49.5 MB, WM 27.4 MB(커밋됨). 밖: Vista 기본 가중치(OpenDriveLab/Vista, 10 GB), Ctrl-World(`junwon-seo/StressDream` 허브 저장소, README 는 `coffee_bag.pt` 하나만 안내) |
| 허브 저장소(API 목록, 내려받지 않음) | `coffee_bag.pt`, `candy_bag.pt`, `coffee_spill.pt`, `jenga.pt`, `knife.pt`, `spoon.pt` 여섯 개(각 9.28 GB, 합 약 55.7 GB). 과제별 `train_traj_0000–0004`, `eval_traj_0000–0004`(hdf5), `stat.json`, `task.json`. 라이선스 필드 없음 |
| 라이선스 | 코드 MIT. 허브 저장소는 라이선스 표기 없음. 기반 모델: Vista(Apache-2.0), Ctrl-World(MIT), SVD(원 라이선스), Qwen3-VL-4B(Apache-2.0), X-CLIP(MIT), PAI-AV 데이터(NVIDIA 라이선스 "other") |

## 2. 논문 ↔ 코드 대조 (D 항목)

판정: **일치**, **차이**(값·식이 다름), **논문에 없음**, **미공개**(논문이 쓴 요소가 저장소에 없음). 영향은 결과 해석에 미치는 방향으로 적었고, 효과 크기는 측정하지 않았습니다.

| 번호 | 항목 | 논문 | 코드 (파일:줄) | 판정 | 영향 |
|---|---|---|---|---|---|
| D1 | 모멘텀 | B.5 "we omit momentum" | Vista `momentum=0.0`(`vista/run_steering.py:268`), Ctrl-World 0.0(`ctrl_world/run_steering.py:316`)은 일치. **Dubins 는 `momentum=0.9`**(`dubins/steer.py:245`) | 차이 (Dubins) | Dubins 의 10회 이동 상한이 모멘텀 없을 때의 약 4.1배(클립 5 기준 207 대 50, `03` V9). 그림 2의 설정이 이 값인지는 확인 불가 |
| D2 | Dubins 등방성(Gram) 항 | 표 3: λ2 = 0.5, k = 16, 순열 100 | `steering_config.yaml:17–20`: `std_coeff 0.5`, **`std_permutation_coeff: 0.0`**. Gram 항은 `std_permutation_coeff * relu(perm − 0.96)`(`wm_steer/regularizer.py:92`)라 계수 0이면 기울기가 없음. 0.5는 표준편차 항(`std_coeff`) | 차이 | 기본 설정에서는 논문이 말하는 블록 등방성 항이 꺼져 있음. 표 3의 λ2 와 k = 16, 순열 100은 계산만 되고 최적화에는 쓰이지 않음 |
| D3 | 그럴듯함 항의 구성 | 세 항: 노름, 등방성, 스펙트럼(4.2절) | `compute_regularizer_video`(`regularizer.py:234–311`)와 `compute_noise_regularizer`(`:65–101`)는 총합 = κ_s·노름² + κ_kl·[표본 평균², 차원별 평균², **이상치(|ε| > 4.7, Dubins 3.5)**, 스펙트럼 힌지, **시간축 스펙트럼 평탄도**] + κ_std·[**표준편차**, 차원별 표준편차, Gram 힌지] | 차이 (논문에 없는 항 다수) | 논문의 C_pla 와 구성이 다름. 평균, 이상치, 표준편차, 시간 스펙트럼 항이 규제에 기여하는 정도는 측정하지 않음 |
| D4 | 스펙트럼 항 | B 개 주파수 구간 평균 파워의 분산(4.2절, B.2) | 주파수 위치별(구간 없음) 파워를 슬라이스·채널에 걸쳐 평균하고 `log(평균)²` 를 평균(`regularizer.py:16–22`), 문턱을 넘는 만큼만 벌점(`relu(spec − thr)`). 문턱은 Vista 0.0098, Ctrl-World 0.055(YAML), Dubins 는 **초기 잡음 자신의 값**(`dubins/steer.py:252–254`) | 차이 | 가우시안 기준값(`03`/V11 재계산)은 Vista 0.01007 ± 0.00006(문턱이 2.7% 아래), Ctrl-World 0.0522 ± 0.0013(문턱이 5% 위). 문턱이 기준값 근처라 한쪽에서만 벌점이 생김 |
| D5 | 등방성 항 | Σ̂ = (1/m)Σεᵢεᵢᵀ 와 I_k 의 Frobenius 거리/k, Vista k = 192, Ctrl-World k = 240 | 행 그룹의 Gram(`ZcZcᵀ`)에서 비대각 + 대각 편차를 `/(n²·k)`(`regularizer.py:36–62, 121–172`). **Vista 는 (5,6,8) 패치로 k = 240**, Ctrl-World 는 정규화 + 제곱. 2D 패치 (12,16)(k = 192)는 계산되지만 YAML 의 `std_perm_include_patched: false` 로 규제에는 안 들어감. `regularizer.py` 독스트링은 Ctrl-World 가 `include_patched=True` 라 적지만 YAML 은 false | 차이 | 표 5의 Vista k = 192 는 코드가 규제에 쓰는 값(240)과 다름. 가우시안 기준값은 Vista 0.994(힌지 0.992), Ctrl-World 1.7e-5(정규화), Dubins 1.008(힌지 0.96, 계수 0) |
| D6 | 계수 이름 대응 | 표 5: (λ1, λ2, λ3) = (0.5, 10, 100). 표 6: (0.2, 0.1, 100) | Vista: λ1 = `kl_coeff_spherical`, λ2 = `std_permutation_coeff`(10), λ3 = `spectral_coeff`(100). Ctrl-World: λ1 = `kl_coeff_spherical`(0.2), **λ2 = `std_coeff`(0.1)** 이고 Gram 계수 `std_permutation_coeff` 는 100. 표에 없는 `kl_coeff = 10`(`steering_config.yaml`)이 스펙트럼·평균·이상치 항에 곱해짐 | 차이 (표기) | 유효 계수는 노름 0.5/0.2, Gram 1.0/10(제곱), 스펙트럼 약 1000/1000(힌지 단위). 표의 두 열이 서로 다른 노브를 λ2 로 적고, 유효 λ3 는 표 값의 10배 |
| D7 | 기울기 처리 | 알고리즘 1: 성분 클램프 ±0.3 → 노름 클립 1.0 | Vista: VLM 기울기를 노름 20 으로 클립한 뒤(`:314–315`), 합친 기울기를 노름 20 클립 → 클램프 ±0.3(`:356–357`). Ctrl-World: 손실별 노름 40, 최종 노름 20, 클램프 ±0.5(`run_steering.py:371–372, 394–395`, YAML `grad_clamp 0.5`). Dubins: 노름 5, 클램프 ±0.5(`steer.py:309–310`) | 차이 | 순서가 다르고 노름 상한이 20배(Vista), 20배(Ctrl-World), 5배(Dubins). 이동 상한은 `03` V9: 알고리즘 1이면 반복 수, 코드면 그 20배 |
| D8 | 노름 되돌림 | ‖ε‖ − √D 의 절댓값 > 3.0 이면 ε ← √D·ε/‖ε‖ (알고리즘 1 11–12행) | Ctrl-World(`:401–402`)와 Dubins(`steer.py:257–263`)는 문턱 3.0 에서 **(ε − 평균)/표준편차 로 표준화**(평균 제거 포함). **Vista 는 되돌림이 없고** 노름을 로그로만 기록(`vista/run_steering.py:361–363`) | 차이 | 표준화된 벡터의 노름은 √(D−1) 근처라 shell 중심과 약간 다르고 평균이 제거됨. Vista 의 노름은 규제 항 외에는 제어되지 않음 |
| D9 | 최적 반환 | 최고 C_i 의 생성 o* 를 반환(알고리즘 1 13–16행) | Ctrl-World: 반복 전 잡음의 보상으로 최고를 추적(`:406–412`)해 일치. **Vista: 최고 추적 없음**, 반복마다 영상 저장과 마지막 잡음만 저장(`:368–391`). **Dubins: 점수는 갱신 전 잡음의 margin, 저장하는 latent 는 갱신 후 잡음에서 다시 생성**(`steer.py:314–321`) | 차이 (Vista, Dubins) | Dubins 에서 선택 점수와 반환 샘플이 같은 잡음이 아님. Vista 는 "worst-case 생성"이 아니라 마지막 반복의 결과를 볼 수 있음 |
| D10 | Ctrl-World 반복 수 | 표 6, §6: 10회 | `steering_config.yaml:26` `iters: 5`, `interact_num: 20`(윈도 수) | 차이 (기본값) | CLI `--iters` 로 바꿀 수 있음. 논문 평가의 윈도 수는 적혀 있지 않음 |
| D11 | Vista 보상 | C_sem = 로그오즈(Qwen)(식 6), CLIP류는 코사인 유사도 차(B.3). 드라이빙은 Qwen2.5-VL(Wolf) + X-CLIP | 기본은 **X-CLIP 의 프롬프트 간 softmax 확률** 평균(`vista/rewards.py:compute_xclip_reward`). Qwen2.5-VL 은 `--use_qwen`(기본 꺼짐)이고 계수 `qwen_coeff = 10`을 곱함. 표 5에 두 VLM 의 상대 가중이 없음 | 차이 | 코드 기본 실행이 논문의 드라이빙 설정과 다름(VRAM 도 48 GB 대 80 GB). 두 보상의 결합 비율은 논문에 없음 |
| D12 | Ctrl-World 행동 표현 | D.1, 표 6: 관절 위치 + 그리퍼, 행동 차원 40 = 8 x H | `config.py` `action_dim = 7`, `stat.json` 길이 7, `wm_helpers.py:50–58` 이 **말단 위치 3 + 오일러 3 + 그리퍼 1**(7차원 EE 상태)을 만듦 | 차이 | 공개 체크포인트는 EE 상태 조건으로 읽힘. 논문 서술(관절 위치)과 다름. 체크포인트별 학습 설정은 확인 불가 |
| D13 | Dubins 학습 데이터와 점수 규약 | 식 9: p = 0.2. 안전 점수 C(s) = p² − 0.25²(실패는 C ≤ 0) | `generate_dataset.py:27` 기본 `naughty_prob = 0.4`(README 명령도 0.4). 코드의 margin = r² − d² (**실패는 margin > 0**, 비관 = margin 최대화)(`env.py:compute_gt_margins`) | 차이 | 공개 체크포인트가 어떤 확률로 학습됐는지 저장소에 없음. 학습이 0.4였다면 WM 이 상상하는 뒤집힘이 평가 라벨(0.2)보다 잦음. 점수는 부호만 다름 |
| D14 | Dubins 목적식과 시간 구조 | 표 3, 안전 점수 최소화, 10회 최적화 | 마진 범위 벌점(`margin_range_coeff 70`, 범위 [−1.8, 0.3])과 `margin_coeff 5`(`steer.py:276–281`). 최적화는 **자기회귀 시간 단계마다** 반복 10회 후 최고 샘플을 고정하고 다음 단계로 진행(`:239–332`) | 논문에 없음 | 시간 단계별 탐욕 최적화이므로 "10회"는 단계마다의 횟수. Best-of-N 기준선이 같은 구조(단계별 선택)인지는 저장소에 없음 |
| D15 | 평가·기준선 코드 | Best-of-N, Classifier Guidance, TPR/TNR·회수율 평가, WorldModelBench·Gemini·Robometer·WorldLens, 표 4의 밀도 대리 모델, 평가 데이터(Dubins 5,000개, PAI-AV 100 + Nexar 200, 실패 궤적 100), π0.5 가중 미세조정 | 저장소에 **없음**. 공칭 생성 스크립트(`dubins/generate_nominal.py`, `ctrl_world/generate_nominal.py`)와 단일 예시 조종 스크립트만 있음 | 미공개 | 그림 2, 4, 5, 6, 8, 표 4의 수치를 저장소로 재현하는 경로가 없음 |
| D16 | 드라이빙 미세조정 가중치 | 13,750 반복 미세조정한 Vista(PAI-AV + nuScenes + Nexar, D.1) | README 는 기본 Vista 가중치만 안내. 충돌 미세조정 모델이 없고, 시연 프롬프트는 "앞차 간격 증가" | 미공개 | 논문 그림 6이 보이듯 기본 Vista 는 충돌을 조종으로 만들지 못하므로, 충돌·보행자 결과(그림 3, 4, 6)의 재현 경로가 없음 |
| D17 | Qwen 보상 부호 | 실패 문장에 대한 yes − no 를 최대화(식 6) | `ctrl_world/rewards.py` 다중 시점 함수는 `target_success=True` 에서 `log p(yes) − log p(no)` 를 보상으로 씀(프롬프트가 "spilled?" 이므로 쏟아짐을 키움). 단일 시점 함수는 같은 플래그에서 부호가 반대(`log p(no) − log p(yes)`) | 일치 (실행 경로), 이름 혼동 | 실행 경로(다중 시점)는 논문과 맞음. 플래그 이름 `target_success` 는 의미와 반대로 읽히고 단일 시점 경로는 부호가 다름 |

### 2.1 일치하는 항목

- Dubins WM(표 2): 128 x 128 입력, 잠재 4 x 16 x 16 = 1,024, VAE 기본 채널 32, 디노이저 은닉 [32, 64, 128], 5단계 오일러, σ_min/max 0.002/80, ρ 7, 배치 128, 100 에폭, 학습률 1e-4(`dubins/config.yaml`). 학습 궤적 4,000(3,800 학습 + 검증).
- Dubins 조종(표 3): 반복 10, η 1.0, 순열 100, k 16, λ1 1.0(`kl_coeff_spherical`), λ3 5.0(`spectral_coeff`). 전체 기울기 경로(`grad_mode: full`)가 기본이고 β(`grad_scale 10`)는 근사 모드에서만 쓰임(K6과 일치).
- Vista(표 5): 576 x 1024, 25프레임, 50단계, CFG 2.5, 반복 20, η 1.0, β 300, λ1 0.5, 순열 1000, 노름 클립 상수는 D7 참조. 오일러 EDM 샘플러 `s_churn = 0` (결정적).
- Ctrl-World(표 6): 3시점 192 x 320, 5프레임, 50단계, CFG 2.0, β 100, λ1 0.2, 순열 1000, 잡음 (5, 4, 72, 40) = 57,600. Qwen3-VL-4B 에 5프레임 x 3시점 + 프롬프트("Is the coffee bean spilled? Respond with a single word: Yes or No."), 윈도마다 새 잡음에서 최적화, 다음 윈도의 조건은 최고 latent 의 마지막 프레임(`:406–412, 472–487`).
- 근사 기울기(식 7): VLM 기울기를 생성 latent 에서 계산하고 `(noise * g).sum().backward()` 로 잡음에 전달(`vista/run_steering.py:316`, `ctrl_world/run_steering.py:368–369`). 전체 기울기는 Dubins 에만.
- 시간 주장: README 의 Vista "생성 1–2분, 20회 30–40분"은 부록 B.6 과 일치(`03` V10).

## 3. 코드가 알려 주는 구조적 사실

1. **"조종"은 윈도·단계 단위의 국소 최적화**입니다. Ctrl-World 는 윈도마다 새 잡음을 초기화해 최적화하고, Dubins 는 단계마다 탐욕적으로 진행합니다. 부록 F 의 "순차 최적화" 서술과 같습니다.
2. **정책 개선 코드가 없습니다.** 가중 흐름 매칭, 40개 시연 가중, π0.5 미세조정은 저장소 밖입니다(D15). 논문의 §6.2 결과는 이 저장소로 대조할 수 없습니다.
3. **공개된 설정은 한 장면 데모**입니다. Vista 는 트럭 이미지 한 장, Ctrl-World 는 `traj_0001.hdf5` 한 개, Dubins 는 `example_traj` 한 개를 기본으로 씁니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 논문 수치의 재현은 불가능합니다. 알고리즘의 축소판(저차원 WM, 잡음 최적화, 규제 세 항, 회수율/TNR 평가)은 CPU 에서 구현할 수 있고, 공식 Dubins 데모는 승인 후 CPU 로 돌릴 수 있습니다.**

| 수준 | 가능성 | 근거 |
|---|---|---|
| Dubins 공식 데모(`dubins/run_steering.py`) | 조건부 | 체크포인트가 저장소에 있어 다운로드 없음. 모델이 작음(WM 27 MB). 장치는 CUDA 없으면 CPU 로 선택됨(`run_steering.py`). 패키지 설치(h5py, einops, ruamel.yaml, imageio, wandb 등)와 공식 코드 실행 승인이 필요. 시간은 측정 전 |
| 논문의 Dubins 정량(그림 2, 표 4) | 조건부(직접 구현) | 평가 코드·5,000개 데이터가 없으므로 평가를 직접 짜야 함. 라벨(10,000회 몬테카를로)은 numpy 로 가능, WM 학습(4,000 궤적, 100 에폭)은 MPS 로 가능할 가능성이 있으나 미측정 |
| 알고리즘 축소판(저차원 확산 WM + 잡음 최적화 + 세 규제 + Best-of-N + 회수율/TNR) | **가능** | torch 2.15(설치됨)로 CPU. 노력 약 1–2주(추정, 측정 전) |
| 근사 기울기(식 7) 타당성 검사 | 가능 | 소형 확산 모델에서 정확 기울기와의 코사인 유사도. 반나절–2일(추정) |
| Vista 조종 | **불가** | README 기준 VRAM 48–80 GB, CUDA 12.8, 체크포인트 10 GB. 24 GiB 통합 메모리와 MPS 로는 로드·역전파 모두 맞지 않음(추정, 미측정) |
| Ctrl-World 조종 | **불가(추정)** | README 기준 VRAM 약 20 GB + Qwen3-VL-4B 약 8 GB, 체크포인트 9.28 GB. 24 GiB 에서 활성값과 역전파까지 담기는 어려움(미측정) |
| 정책 개선(π0.5-DROID 미세조정) | **불가** | H100 한 장 10k 스텝, 코드 없음, 실로봇 평가 |
| 평가 데이터(PAI-AV, Nexar, 사람 라벨) | 불가 | 대용량 데이터와 사람 판정 |

노력 추정(측정 전): 축소판 구현 1–2주, 공식 Dubins 데모 반나절(승인 후), CUDA 가능한 기기에서 Ctrl-World 조종 데모는 환경 구성 1–2일, 논문 그림 5·8 재현은 데이터와 로봇이 필요해 별개의 프로젝트입니다.

## 5. 저자에게 물을 것

1. 그림 5와 그림 22의 Best-of-N 은 N = 10 입니까 20 입니까, 그리고 "같은 예산"은 반복 수, 생성 횟수, 계산량 중 무엇입니까.
2. 평가용 실패 궤적 100개는 표 7의 미세조정 궤적과 분리되어 있습니까. 과제별 개수는 얼마입니까.
3. 오차 막대의 정의(그림 2, 5, 8, 16, 22, 26)는 무엇입니까. 그림 26의 과제별 시행 수는 20 입니까.
4. 표 5·6의 λ2 와 k 는 공개 코드의 어느 설정에 해당합니까(D2, D5, D6). Dubins 결과는 `std_permutation_coeff = 0` 과 모멘텀 0.9 로 얻은 것입니까(D1, D2).
5. 공개 Dubins 체크포인트의 학습 뒤집기 확률은 0.2 입니까 0.4 입니까(D13). 표 4의 등방성·스펙트럼 값의 k 와 구간 수는 무엇입니까.
6. 드라이빙 결과에 쓴 미세조정 Vista 와 Best-of-N·Classifier Guidance 평가 코드를 공개할 계획이 있습니까(D15, D16).

# 04. 공식 코드 대조와 구현 가능성: τ0-WM

공식 저장소 https://github.com/sii-research/tau-0-wm 를 얕게 clone해 **정적으로만** 읽었습니다(실행, import, 설치, 가중치 받기 없음). 커밋 `e03902a4a8e0cf89b681e89920784ae8ca7b4b3c`(2026-10-13, "Merge pull request #5 from dhandhalyabhavik/xpu-support"), 커밋은 이력이 하나인 얕은 clone이라 하나만 보입니다. 재현하려면 `git clone --depth 1 https://github.com/sii-research/tau-0-wm repo` 뒤 위 커밋인지 확인합니다. 줄 번호는 이 커밋 기준입니다.

## 1. 저장소가 담은 것

| 영역 | 내용 | 논문과의 관계 |
|---|---|---|
| 사후학습 | `runner/posttrain.py`(VAM), `runner/posttrain_sim.py`(ACVS), `scripts/train.sh`(torchrun, `nvidia-smi`로 GPU 수 탐지), 예제 데이터 `taco_play`(LeRobot) | 사전학습 코드는 없음 |
| 추론 | `web_infer_utils/`(websocket 서버, openpi 클라이언트 기반), 사전학습 설정 `configs/deployment/tau_pretrain_rela_eef6d.yaml` | 정책 서버 |
| 테스트 시간 계산 | `web_infer_utils/posttrain_taco_play_withTTC/`, `models/.../pipeline/textimage2video_ttc.py` | RCS, 시뮬레이터 평가, 재질의 |
| 시뮬레이터 | `models/.../transformers/model_sim.py`, `pipeline/textimage2video_sim.py`, `TauSimulator.py` | ACVS |
| 모델 | `models/wan_2_2_models/`(Wan2.2 코드 기반) | 영상 백본, 행동 가지(`action_blocks`), 보상 가지(`reward_blocks`) |
| 평가·데이터 | 없음 | 논문 과제, 기준선(π0.5, Fast-WAM), 사전학습 데이터 처리, 평가 코드 모두 없음 |

저장소 크기는 약 18 MB(대부분 그림 파일 두 개 약 10 MB)이고, 소스 파일은 수십 개입니다. 공개 가중치는 두 가지(VAM, simulator)이며 크기는 Hugging Face API 메타데이터의 값입니다(`02` §4). 가중치는 받지 않았습니다.

## 2. 논문 ↔ 코드 대조 (D1–D20)

표기: 일치는 ✓, 다름은 △, 논문에 없거나 코드에 없음은 ○.

| ID | 항목 | 논문 | 코드 (위치) | 판정 |
|---|---|---|---|---|
| D1 | 공개 범위 | 27.3K h 사전학습 + 감독 마스크(III, IV-C) | 사후학습과 추론만. 학습 루프에 마스크가 없고 모든 샘플이 영상과 행동(또는 보상)을 함께 감독함(`runner/posttrain.py` 740–840) | ○ 사전학습 코드, 마스크 구현 없음 |
| D2 | 손실 | 식 2: `λz‖·‖² + λa‖·‖²`, `λz = λa = 1` | `loss = loss_video·video_loss_scale + loss_action·action_loss_scale`(기본 1.0)(`posttrain.py` 836). 단 영상 손실은 조건 프레임 칸(`cond_mask = 0`)을 포함해 **전체 요소 평균**이라 미래 프레임 칸의 가중이 (미래 프레임 수/전체)로 줄고, 행동 손실도 요소 평균 | △ 정규화 방식 (식은 노름 제곱이고 정규화 규칙이 적혀 있지 않음) |
| D3 | 잡음 수준 | 영상·행동에 각자 `u_z`, `u_a` | 독립 ✓. 시각 샘플링은 `sample_mode="uniform"`으로 고정되어 설정의 `flow_logit_mean/std`는 쓰이지 않음(`posttrain.py` 748, 792). 영상 스케줄은 shift 5.0(UniPC), 행동은 shift 1.0 오일러(`posttrain.py` 462–470) | ✓ 구조, ○ 분포와 shift는 논문에 없음 |
| D4 | 시야와 해상도 | `Hv`, `Ha`, 해상도 없음. 덩어리 길이 30 (부록 A-2) | `chunk: 9`(영상 9프레임 = 잠재 3칸: 현재 1 + 미래 2), `action_chunk: 33`, `img_size: [192, 256]`, 시점별로 가로 결합(`configs/deployment/*.yaml`). 클라이언트 예제는 `execution_step: 30` | ○ 논문에 값 없음, 코드로 보면 예측 33, 실행 30 |
| D5 | 행동 전용 추론 | "action-only deployment" (IV-D) | 영상 가지를 **한 번만** 시각 1000(미래 칸이 순수 잡음)으로 돌려 특징을 저장하고, 행동은 5 스텝 오일러(shift 1.0)로 노이즈 제거, CFG 없음(`textimage2video.py` 484–485, 563, 611–625). 미래 영상은 노이즈 제거되지 않음 | △ 논문이 "미래 잠재를 행동 생성에 쓴다"고 읽히는 서술과 다름. 학습은 영상 잡음 수준이 무작위라 이 입력이 분포 안이긴 함 |
| D6 | RCS | `K`개의 무작위 흐름 시각, 오차의 평균(식 6, 7) | `RCS_timestep_ratios=[0.5]` 비율 목록(기본 `K = 1`, **무작위 아님**)으로 샘플러 시각 index `int(len·ratio)`를 고르고, **같은 새 잡음을 후보 전체에 공유**, 오차 = 속도 예측과 `(noise − x)`의 평균 제곱, 부호 반전(`textimage2video_ttc.py` 556–585) | △ 무작위 시각 대신 고정 비율, 후보 간 잡음 공유 |
| D7 | γ 게이트와 LAR | RCS가 γ 미만일 때만 ACVS 호출, 최고 롤아웃으로 VAM 재질의(알고리즘 1, 2, 식 8) | γ 없음. `sim_refine=True`면 **매번 ACVS로 모든 후보를 평가**하고, 최고 예측 보상의 **부호가 0 미만일 때만** 재질의(`TauPolicyTTC.py` 477). 아니면 RCS 최고 후보를 보상 최대 지점까지 자름(`new_execution_step`, 474). 재질의는 시뮬레이터의 중간 잠재(`sim_refine_return_latent_timestep=3`)를 VAM의 영상 입력으로 넣고 후보 N개를 다시 뽑아 RCS로 한 번 더 고름. `sim_refine_ths=-0.015`는 인자로만 있고 쓰이지 않음 | △ 게이트 기준(RCS 임계값 대 보상 부호), 중간 잠재, 실행 길이 절단 |
| D8 | TTC 기본값 | `N = 4` (VII-B) | 함수 기본값은 `N_proposals=1`, `sim_refine=False`. 예제 클라이언트는 `N_proposals: 4`, `sim_refine: True`(`simple_client.py`) | ✓ 예제에서 |
| D9 | 가이던스 기준선 | 표 II의 CFG [18], ACG [32] | 행동 경로에 CFG 없음(`guide_scale`은 영상 생성에만), ACG 구현 없음(검색 결과 0건) | ○ 기준선 코드 없음 |
| D10 | 보상 가지 | 그림 2b "0.5B", 식 5의 `h`는 행동 조건 영상 특징 | 보상 블록 차원은 기본값 1536(설정에 지정 없음)으로 0.91B(`03` V11). 보상은 영상 특징에 `detach`해서 연결(`reward_detach_feat=True`, `model_sim.py` 490, 811, 831): 보상 손실이 백본을 갱신하지 않음. 보상 손실의 가중에 `action_loss_scale` 인자가 쓰임(`posttrain_sim.py` 850) | △ 크기 표기, 기울기 흐름, 인자 이름 |
| D11 | 보상 라벨 | 하위 과제 단위 진행, 구간 안 몬테카를로, 실패 구간 음수 (V-C) | 예제 데이터셋은 **에피소드 전체를 한 구간**으로 보고 몬테카를로 반환(감쇠 0.9, 끝 보상 1, `rescale=False`이라 범위 [0, 1])을 쓰고, 유효 단계가 없을 때만 −1(`data/example_dataset_sim.py` 523–548). 반환 함수의 문서는 [−1, 1]로 재조정한다고 적혀 있으나 호출은 `rescale=False` | ○ 논문 방식의 라벨 코드는 없고 예제만 있음 (예제 주석: "simply use the task progress as reward for example") |
| D12 | 기억 관측 | `o_{t−M:t}` (식 3) | 기억 프레임 3개(`n_mem=3`)를 **에피소드 시작부터 현재까지 균등 간격**으로 뽑음(학습: `example_dataset_sim.py` 80, 배포: `TauPolicyTTC.py` 359의 `np.linspace(0, n_history-1, n_mem+1)`) | △ "최근 M개"로 읽히는 표기와 다름. M = 3 |
| D13 | 행동 조건 (식 4) | 행동 묶음 → MLP 두 개 → 시간 임베딩과 AdaLN 변조에 더함 | 일치(`model_sim.py` 740–744). 묶음은 잠재 칸당 16개 행동, 첫 칸(관측)은 0. 학습에서 행동을 확률 0.2로 0으로 만들고 캡션은 확률 0.5로 비움(VAM 사후학습은 캡션 0.06) | ✓ 구조, ○ 묶음 16, 드롭아웃은 논문에 없음 |
| D14 | 행동 표현 | 표현이 적혀 있지 않음 | 사전학습 설정은 팔마다 xyz + 6D 회전의 **상대 EEF 행동 20차원**(그리퍼 별도), 사후학습 예제는 7차원 절대 관절, 시뮬레이터는 전신 26차원(정책 20 + 머리 2, 허리 2, 속도 2의 기본값)이며 정책 출력과 시뮬레이터 입력은 통계를 바꿔 변환(`TauPolicyTTC.py` 127, 431–455) | ○ 논문에 없음 |
| D15 | 구조와 크기 | VAM 5.5B (영상 5B + 행동 0.5B) | 계산 5.511B, 행동 가지는 영상 특징에 층마다 cross-attn, 영상 가지는 행동을 받지 않음(`model.py` 585–601, 750–800). 파일 크기와 일치(`03` V11) | ✓ |
| D16 | 최적화 | AdamW, 학습률 5×10⁻⁵, 전역 배치 12,288(사전)/384(사후) | AdamW, `lr 5e-5` ✓. 베타 (0.9, 0.95), 가중치 감쇠 1e-5, 워밍업 1,000 스텝 뒤 상수, 그래디언트 클리핑 1.0, bf16. 설정의 `batch_size: 12`는 **장치당** 값(`scale_lr: false`) | ✓ 일부, ○ 나머지는 논문에 없음 |
| D17 | 추론 최적화 | KV 캐시, 융합 QKV, 1차원 RoPE 캐시, 층별 `torch.compile`, 텍스트 캐시 (부록 B) | 융합 QKV와 RoPE 캐시는 기본 켜짐, KV 캐시는 서버 인자(`--enable-action-cross-attn-kv-cache`)를 줘야 켜짐, `compile`은 기본 꺼짐, 텍스트 캐시 구현(`server.py`, `TauPolicy.py`) | ✓ (지연 수치는 검증 불가) |
| D18 | 장치 지원 | RTX 5090 (A-2) | README는 `--device`로 cpu, cuda, xpu를 지원한다고 적음(MPS 없음). 그러나 TTC 파이프라인과 시뮬레이터 모델에 `torch.amp.autocast('cuda')`, `torch.cuda.empty_cache()`가 고정(`textimage2video_ttc.py` 418, `model_sim.py` 379 외, `textimage2video_sim.py` 386) | △ TTC·시뮬레이터 경로는 CUDA 전제 |
| D19 | 라이선스 | - | README "Apache License 2.0"이지만 링크가 diffusers 저장소의 LICENSE이고 저장소에 LICENSE 파일이 없음, GitHub API license 없음. HF 카드는 apache-2.0. 코드 파일 머리글에 Wan 저작권 표시 | ○ 확인이 필요 |
| D20 | 평가와 재현 | 네 과제, 표 I, II, 그림 4 | 평가 코드, 과제 설정, 기준선, 사전학습 데이터, 진행·보상 라벨 도구, 실로봇 연결 코드 없음(`taco_play` 예제만) | ○ |

### 2.1 D5와 D7의 의미

- **D5.** 행동 전용 추론에서는 영상 DiT가 미래 칸이 순수 잡음인 상태로 한 번만 돌아 특징을 만들고, 그 특징에 행동 가지가 cross-attention으로 붙습니다. 논문이 "미래 영상 예측이 행동 학습의 보조 목적일 뿐 아니라 전이 가능한 상호작용 동역학을 배우게 한다"고 쓰는 것(IV-A)과는 학습 설명이고, 배포 시점의 계산 경로는 Fast-WAM [45]이 다룬 "추론 시 상상 제거"와 구조가 같습니다. 코드의 "rollout-enabled" 모드(`infer_cotrain`)는 영상을 1스텝으로만 노이즈 제거하고(`video_sampling_steps=1`) 그 결과는 시각화용입니다. 행동 가지가 보는 특징은 마지막 영상 스텝 입력에서 나옵니다.
- **D7.** 논문의 알고리즘 1은 RCS를 값싼 거름망으로 두고 ACVS는 RCS가 낮을 때만 부릅니다. 공개 코드는 `sim_refine`를 켜면 항상 ACVS를 부르고(`03` V9: 영상 백본 계산이 VAM 한 번의 약 40배), 재질의는 예측 보상이 음수일 때만 합니다. 그래서 코드 기본 경로의 지연은 부록의 180–220 ms 범위가 아닙니다. 논문 알고리즘의 γ 값과 코드의 부호 게이트가 같은 결과를 내는지는 알 수 없습니다.

## 3. 라이선스

- **코드:** README는 Apache-2.0(링크는 diffusers의 LICENSE), 저장소에 LICENSE 파일이 없고 GitHub API의 license 필드도 비어 있습니다(D19). 사용 전에 저자에게 확인하는 편이 안전합니다. 코드의 일부는 Wan2.2(Apache-2.0)와 Genie Envisioner V1(API 라이선스 없음, 저장소 README가 CC BY-NC-SA 4.0과 Apache-2.0 혼합을 적음)에서 수정했다고 README가 밝힙니다.
- **가중치:** Hugging Face 카드 apache-2.0.
- **논문:** arXiv CC BY-NC-SA 4.0 (`01` §0.2).

## 4. 구현 가능성 판정

**"구현할 수 있는가"에 대한 판정. 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 논문 수치의 재현은 불가능합니다. 구조의 축소 재구현과 테스트 시간 계산 로직의 검증은 CPU에서 할 수 있고, 공식 가중치 추론은 승인과 포팅이 필요한 조건부입니다.**

| 경로 | 이 맥 | 필요한 것 | 근거 |
|---|---|---|---|
| 사전학습 재현 | 불가 | 27.3K h 데이터(로봇 17.8K h는 비공개), 전역 배치 12,288, 코드 없음 | D1, `01` §2.4 |
| 사후학습(공식 `taco_play` 예제 포함) | 불가 | 5.5B 파라미터 AdamW 학습은 bf16 가중치만 11 GB에 옵티마이저·그래디언트 상태가 더해져 24 GiB를 넘음(추정), 코드는 CUDA와 `torchrun`, DeepSpeed 전제, 데이터 다운로드 | `scripts/train.sh`, D16 |
| 논문 실로봇 평가(표 I, II, 그림 4) | 불가 | AGIBOT-G01, ARX, 이중 Franka, 과제 환경, 기준선 | D20 |
| 공식 VAM 가중치 추론 | 조건부 | 가중치 11.0 GB + umT5-XXL 11.4 GB + VAE 2.8 GB 다운로드(합계 약 25 GB, 텍스트 인코더를 먼저 돌려 비우면 약 14 GB 상주), 새 가상환경에 `requirements.txt` 설치, 공식 코드 실행 승인, MPS 포팅(README 지원 장치에 MPS가 없고 CPU는 가능) | D18, `03` V9 |
| 공식 simulator + TTC 추론 | 조건부 (더 무거움) | 위 + simulator 11.9 GB, CUDA 고정 호출 수정, 후보 4개 평가에 영상 백본 약 115 TFLOP | D7, D18 |
| 축소 재구현 (영상 토큰 + 행동 DiT + RCS/LAR, 수 M 파라미터) | 가능 | torch(설치됨), MuJoCo(설치됨). 큰 과제는 불가, 장난감 과제 | 아래 |
| 재계산, 구조·파라미터 계산, 토이 | 완료 | 새 설치 없음 | `03` V1–V12, T1 |

**노력 추정(측정 전, 추정):** 공식 가중치 추론 포팅과 측정은 1–2주(다운로드 승인 후). 축소 VAM을 MuJoCo 푸시 과제에 맞춰 만들고 RCS/LAR를 붙여 확인하는 데 2–4주. 논문 수준 전체 재현은 하드웨어, GPU, 비공개 데이터가 필요해 공개 자료만으로는 어렵습니다.

## 5. 저자에게 확인하고 싶은 것

1. 그림 4, 표 I의 시행 수, 성공 판정 규칙, 재시도 허용 여부, 그림 4의 τ0-WM이 TTC를 쓴 값인지(`01` G1, G8).
2. 진행도(그림 4b)의 정의와, School Bag에서 진행도가 성공률보다 낮은 이유(`03` V4).
3. 표 I zero-shot 칸의 시행 수(`03` V3).
4. 알고리즘 1의 γ, `K`, `Hv`, `Ha`, 메모리 길이 `M`과 공개 코드의 부호 게이트(`D6`, `D7`, `D12`)의 관계.
5. 그림 2b의 보상 가지 크기(0.5B)와 공개 simulator의 구성(1536차원, 0.91B)의 관계(D10).
6. 롤아웃·실패 궤적이 27.3K h에 포함되는지(`01` G9).
7. CFG, ACG 기준선의 설정과 코드(D9).

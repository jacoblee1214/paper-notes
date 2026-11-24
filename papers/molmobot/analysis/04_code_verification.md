# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: `allenai/MolmoBot` (공식, 프로젝트 페이지와 논문 코드 링크). 커밋 `be82891b50437fb081589fe3be353567241053da` (2026-11-23). 얕은 복제(179개 파일, 4.1 MB)를 로컬에만 두었습니다(저장소에 올리지 않음). 데이터 생성 엔진(MolmoBot-Datagen)은 이 저장소가 아니라 `allenai/molmospaces`에 있습니다. 그 저장소는 복제하지 않고 GitHub API로 설정 파일 하나(`molmo_spaces/configs/robot_configs.py`)만 읽었습니다.
- **정적 읽기만 했습니다.** 코드를 실행, import, 설치하지 않았습니다. 가중치와 데이터는 받지 않았고, Hugging Face의 메타데이터(크기, 라이선스, 파일 목록)만 읽었습니다.
- 읽은 파일: `README.md`(루트와 하위 3개), `MolmoBot/launch_scripts/{train_molmobot,serve_molmo,run_eval}.py`, `olmo/models/molmobot/{molmobot,inference_wrapper}.py`, `olmo/nn/action_expert.py`, `olmo/data/{synthmanip_dataset,synthmanip_grasp_sampling,synthmanip_presets}.py`, `olmo/train_init_utils.py`, `olmo/preprocessing/image_preprocessor.py`(증강), `olmo/eval/{configure_molmo_spaces,configure_real_robot}.py`, `olmo/train/{trainer_config,run_trainer}.py`(동결 설정), `MolmoBot-Pi0/molmobot_pi0/{config_openpi,eval/policies/pi,eval/eval}.py`와 `config/eval/*`, `MolmoBot-SPOC/architecture/{__init__,config/model_config}.py`, `training/config/*`, `eval/config/franka_eval_config.py`, `robot_eval/scripts/droid/run_policy.py`와 설정.
- 논문 수치는 PDF(v3) 텍스트에서, 코드 값은 파일 줄에서 옮겼습니다. 코드의 기본값과 README가 주는 명령행 값을 구분해 적었습니다.

## 1. 저장소, 라이선스, 자원

| 항목 | 내용 |
|---|---|
| 구성 | `MolmoBot/`(MolmoBot 학습·추론·평가, OLMo 계열 코드), `MolmoBot-Pi0/`(openpi 기반 π0 구조), `MolmoBot-SPOC/`(경량 정책, PyTorch Lightning), `robot_eval/`(DROID 실로봇 실행 스크립트) |
| 라이선스 | 루트와 하위 3개 폴더 모두 Apache-2.0 파일 (GitHub API도 Apache-2.0) |
| 환경 | MolmoBot은 `uv`와 Python 3.11(`~=3.11.0`), eval 추가 패키지가 `molmo_spaces`를 커밋 `cd23bec`로 고정해 설치. 학습 README는 H100 8–64장(MolmoBot), 32장(MolmoBot-Pi0)을 가정 |
| 모델 (Hugging Face) | `MolmoBot-DROID`(Apache-2.0, 약 20 GB), `MolmoBot-Img-DROID`(20 GB), `MolmoBot-Pi0-DROID`(gemma 라이선스, 약 7 GB), `MolmoBot-SPOC-DROID`(약 0.4 GB, 라이선스 "other"), RB-Y1 모델 등 9종 |
| 데이터 (Hugging Face) | `allenai/molmobot-data`, ODC-BY 1.0. 파일 크기 합 약 10.3 TB(저장 용량 필드 8.87 TB). 상용 사용이 가능한 에피소드 목록 `commercial_episodes.parquet`가 따로 있고, 논문 표 3에 없는 `FrankaPickAndPlaceOmniCamConfig_ObjectBackfill`(학습한 공개 모델에는 쓰지 않았다고 명시)이 포함 |

## 2. 논문 ↔ 코드 대조 (D 항목)

판정: **일치**는 논문과 코드 값이 같음, **차이**는 다름, **논문에 없음**은 코드에만 있는 설정, **미확인**은 이 저장소에서 확인할 수 없음. "README"는 공식 README의 학습 명령행 값입니다(`train_molmobot.py` 안의 기본값과 다를 수 있음).

| 번호 | 항목 | 논문 | 코드 | 판정 | 영향 |
|---|---|---|---|---|---|
| D1 | MolmoBot 최적화 설정 | 배치 1024, 200k 스텝(정적), 100k(이동), "나머지는 변경 없음"(§4.1). 학습률 등 미기재 | AdamW, 베타 (0.9, 0.95), eps 1e-6, 가중치 감쇠 0, 기울기 클리핑 1.0, bf16. 학습률 행동 헤드 **1e-4**, LLM **1e-5**, ViT·연결부 5e-6. 워밍업 200(헤드·ViT·연결부)/**2000(LLM)** 스텝, 최종 학습률 비율 0.1 (`train_molmobot.py`) | **논문에 없음** | 재현에 필요한 값이 논문에 없음 |
| D2 | 학습되는 부분 | "SigLIP 비전 인코더를 고정"(§4.1) | `train_molmobot.py`의 기본은 LLM, ViT, 연결부 모두 동결(`ft_llm=False, ft_vit=False, ft_connector=False`). README는 `--ft_llm=True --ft_embedding=ae`로 **LLM을 학습**하고, 연결부는 동결로 둠 | **차이 (논문 서술보다 범위가 구체적)** | LLM 전체 미세조정(LR 1e-5)이 논문 문장에는 없음. 연결부 동결도 없음 |
| D3 | 행동 헤드 크기 | "VLM 백본과 같은 층 수의 DiT, 층마다 대응 VLM 층에 교차 주의"(§4.1) | 층 수 = LLM 층 수(`per_layer`, 불일치 시 오류) **일치**. 은닉 768, 헤드 8, MLP 비율 4, 시각 임베딩 256 (설정 클래스 기본 1024/16/32층을 학습 스크립트가 덮어씀) | **일치 (층 수), 논문에 없음 (폭)** | 헤드 파라미터 수를 논문에서 알 수 없음 |
| D4 | 흐름 매칭 설정 | 시각 8개 병렬 지도(§4.1, C.7) 외 미기재 | 시각 `t = 0.999 · Beta(1, 1.5)`, `x_t = (1−t)ε + t·a`, 목표 속도 `a − ε`, MSE. 이는 π0 논문의 시각 분포와 같은 모양. 추론은 오일러 **10스텝**, ε ~ N(0, I). 병렬 시각 수 `num_flow_timestamps`의 클래스 기본은 **1**이고 README가 `=8`을 넘김 | **논문에 없음 (8은 README로 일치)** | 스크립트를 인자 없이 돌리면 시각 1개로 학습됨 |
| D5 | 프레임 샘플링 가중치 | 재시도 3×, 성공 파지 2×, 완료 행동 2×(§4.1) | 가중 샘플링 시 `final_grasp_weight 2.0`(성공 파지, 파지 시점 앞 10프레임), `release_after_failed_grasp_weight 3.0`(실패 파지의 **해제 시점** 앞 10프레임), `go_home_weight 2.0`(성공 에피소드에서 마지막 해제 5–20스텝 뒤), 그리고 **실패한 파지 구간은 0.5×로 낮추고 그 접근 구간을 0.5 이하로 제한** (`train_init_utils.py`, `synthmanip_grasp_sampling.py`) | **차이 (재시도 3×는 해제 구간, 0.5× 감소는 논문에 없음)** | "재시도 3×"의 정의가 코드에서는 구간 가중치(바닥값)이고 실패 파지는 오히려 낮춤 |
| D6 | 학습 카메라 선택 | "손목 + 무작위 외부 카메라 하나"(A.2.1) | README의 `--furthest_camera_prob=0.5`: 에피소드 시작 시점에서 ZED2 유사 카메라 둘 중 로봇에서 **가장 먼 쪽을 50% 확률로** 고르고, 아니면 `randomized_zed2_analogue_1` 고정 (`synthmanip_dataset.py`) | **논문에 없음** | "하나의 무작위 외부 카메라"가 실제로는 둘 중 선택 규칙입니다 |
| D7 | 정규화 | 미기재 | 행동과 상태를 데이터셋 전체의 **1%, 99% 분위수**로 정규화(여러 데이터셋의 q01은 최솟값, q99는 최댓값을 합침), 상태는 0으로 은닉 크기까지 채운 뒤 선형층 하나로 토큰화 | **논문에 없음** | 분포가 다른 데이터로 재학습하면 통계를 다시 계산해야 함 |
| D8 | 이력 입력 | 과거 1개(H=1) 또는 2개(H=2), 간격 D = 8 스텝, MolmoBot-Img에서 50k 스텝 추가(§4.1) | README: `n_obs_steps=2, obs_step_delta=8, max_images=4, seq_len=928, max_duration=50000`, `--reset_trainer_state --reset_optimizer_state`(학습률 스케줄과 워밍업이 다시 시작) | **일치 (H=1 설정)** | 학습률 재시작은 논문의 "그 외 파라미터 동일"과 같은 서술이지만 스케줄 재워밍업은 명시되지 않음 |
| D9 | Franka 데이터 혼합 | 표 5: Pick 20, PnP 고정 높이 10, PnP 무작위 높이 35, Next-to 20, Color 15% | README: `FrankaPickOmniCamConfig/part0` 0.2, `FrankaPickAndPlaceOmniCamConfig/part0` 0.1, `part1` 0.35, `NextTo/part2` 0.2, `Color/part0` 0.15 | **일치** (part 번호와 높이 조건의 대응은 README에 없음) | 사용한 분할(part)이 전체 1.7M 중 어느 비율인지 논문과 README에 없음 |
| D10 | RB-Y1 행동 표현 | "토르소와 팔은 변화량 관절 위치, 베이스는 선형·각 변화"(B.5), 토르소 6-DoF(§3.2) | `RBY1_multitask`: 베이스와 팔은 `joint_pos_rel`(변화량), 그리퍼는 절대, **토르소는 1차원 절대값**(상태는 토르소 관절 1–3), 평가는 토르소를 "height" 모드로 6관절에 매핑. `RBY1_door_opening`은 토르소와 머리가 없음 | **차이** | 논문 서술보다 단순한 행동 공간입니다. 6-DoF 토르소 전체를 제어하는 것이 아님 |
| D11 | RB-Y1 학습 예시 | 배치 1024는 정적 조작 설정, 이동 조작은 100k 스텝, 표 6의 4과제 혼합 | README의 RB-Y1 예시: 100k 스텝(일치), **배치 256**, 데이터 `RBY1OpenDataGenConfig/part0` 하나, 시작점이 Molmo2가 아니라 공개 `MolmoBot-RBY1Multitask` 체크포인트 | **차이 (예시 명령)** | RB-Y1 모델의 표 6 혼합 학습 명령은 공개되지 않음 |
| D12 | 평가 주기와 시간 한도 | Franka 정책 주기 66 ms(15 Hz), 한도 303스텝 20 s(Pick) 또는 606스텝 40 s(PnP) (C.3) | README 평가 구성 `FrankaState8ClampAbsPosConfig`는 66 ms **일치**. 그러나 같은 파일의 `FrankaAbsPosRandomCamConfig`와 `SynthVLAFrankaBenchmarkOriginalEvalConfig`는 **200 ms**이고 후자에 "이것이 실제로 평가에 쓰는 설정"이라는 주석. README는 `--task_horizon 600`(논문 606). MolmoBot-Pi0 평가 기본 한도 40 s, SPOC Franka 기본 400스텝(≈ 26.4 s). RB-Y1 평가는 100 ms, 400스텝(40 s) | **불일치 가능, 미확인** | 200 ms이면 303스텝이 60.6 s여서 표 1의 시간 한도가 달라짐. 어느 설정이 표 1을 만들었는지 확인 필요(V15) |
| D13 | 관절 변화 상한 | 미기재 | 시뮬과 실로봇 평가 모두 **스텝당 최대 관절 변화 0.2 rad로 비례 축소**(절대 위치 행동도 현재 관절 대비 변화량으로 계산). MolmoBot-Pi0도 `max_joint_delta: 0.2`. `--no-max-delta`로 끌 수 있음 (`configure_*.py`, `serve_molmo.py`) | **논문에 없음** | 15 Hz에서 ≈ 3 rad/s의 속도 상한이라 정책 동작에 영향. 논문 수치는 이 상한 아래의 결과 |
| D14 | 그리퍼 출력 처리 | 미기재 | MolmoBot: 그리퍼 값이 128을 넘으면 255, 아니면 0으로 **이진화**(`clamp_gripper=True` 기본). RB-Y1 문 열기는 임계 5.0, RB-Y1 Pick/PnP는 이진화 끔. MolmoBot-Pi0는 0–1로 자르고 255를 곱한 연속값 | **논문에 없음** | 정책 사이 그리퍼 처리가 달라 같은 기준 비교가 아님 |
| D15 | MolmoBot-Pi0 평가 설정 | 학습은 openpi 기본, 평가 서술 없음 | `buffer_length=-1`이 1000이 되고 예측 청크(16)를 **끝까지 실행한 뒤** 다시 추론(MolmoBot은 16개 예측 중 8개 실행). 성공 판정에 `min_success_duration: 0.5` s 사용 | **논문에 없음** | 두 정책의 폐루프 빈도가 다름(Pi0 15 Hz 기준 약 1 s마다 대 MolmoBot 약 0.5 s마다) |
| D16 | MolmoBot-Pi0 학습 | 200k 스텝, 배치 1024, 학습률 5e-5, 워밍업 1k, 비전 인코더 고정(§4.3) | `molmobot_pi0_droid` 설정: 워밍업 1,000, 최고 5e-5, 감쇠 최종 5e-5(**상수**), 스텝 200,001, 배치 기본 256(README가 1024), `freeze_filter="vision_tower"`, 행동 차원 8, 행동 길이 16 | **일치** (README 배치) | 학습률이 사실상 상수임은 논문에 없음 |
| D17 | π0.5-Finetune | MolmoBot-Data로 15k 스텝 미세조정(§5.2.1) | `pi05_mlspaces_finetune`: `pi05_droid`에서 시작, 상수 5e-5, **200,000 스텝**, 배치 256, 행동은 DROID 관절 속도(`joint_pos_actions=False`). 15k를 만드는 명령은 README에 없음 | **차이 (기본값)** 명령행 덮어쓰기일 수 있음 | 기준선 학습량이 코드 기본값과 논문이 다름 |
| D18 | MolmoBot-SPOC (Franka) | 108M 학습 파라미터, 분위수 256빈, 병렬 디코더 16청크(§4.2). Franka 학습 카메라는 서술 없음 | 24층, dim 512, 32헤드, 분위수 빈(기본 256), 파라미터 약 107M(`03` V14), 주석은 ~150M. **Franka 모델은 고정 어깨 카메라**(`droid_shoulder_light_randomization`)와 손목 카메라로 학습되고 README가 "카메라 무작위화는 지원하지 않는다"고 경고. 학습은 에폭 기준(100, RB-Y1 100/250), 학습률 2e-4, `32-true` 정밀도 | **일치 (구조·파라미터), 논문에 없음 (카메라, 에폭)** | SPOC가 표 11(고정 카메라)에만 나오는 것과 맞음 |
| D19 | 점 조건화 | 첫 프레임을 "grounding frame"으로, XML `<point coord=…>`(B.1), RB-Y1에서 후보 10개 중 샘플(B.5) | 학습 기본 `conditioning_frame="random_first_10"`(앞 10프레임 중 무작위), 학습 점 수 기본 1, 평가는 물체당 점 최대 10, 포맷은 Molmo의 `html-v2` | **차이 (세부)** | 평가 입력이 학습 분포와 약간 다름(점 1개 대 10개 후보) |
| D20 | 이미지 증강 | ColorJitter, GaussianBlur, RandomPosterize, RandomSharpness, RandomGrayscale, "서로 다른 확률"(B.3) | ColorJitter(밝기·대비 0.4, 채도 0.2, 색상 0.05)와 GaussianBlur는 **항상 적용**, Posterize 7·6·5·4비트 각 0.2, Sharpness 0.5, Grayscale 0.05 | **일치** | — |
| D21 | 데이터 생성 상수 | 행동 잡음 0.1·‖Δx‖, ±2 cm, ±0.1 rad, 베이스 ±2 cm, ±0.05 rad, 초기 관절 잡음, 머리 ±0.2 rad, 그리퍼 ±0.01(A.1) | `allenai/molmospaces`의 `robot_configs.py`에 같은 값이 있음(`action_scale_factor 0.1`, `max_tcp_position_noise 0.02`, `max_tcp_rotation_noise 0.1`, 초기 관절 `[0.025 … 0.175]`, RB-Y1 `[0.05, 0.05, 0.075 … 0.175]`, 머리 `[0.2, 0.2]`, 그리퍼 `[0.01]`). 평가에서는 잡음 끔 | **일치** | 생성기의 다른 부분(파지 순위, 재시도, CuRobo 설정)은 읽지 않았음 |
| D22 | 학습 샘플 수 | 200k 스텝 × 배치 1024(§4.1) | 200,000 × 1,024 = 2.05억 샘플. README의 데이터 혼합이 전체 Franka 프레임(약 2.6억, v2 표 1) 대비 어느 비율인지는 `03` V17 | — | 데이터가 "한 번 미만"으로 쓰이는 정도를 보여 줌 |
| D23 | 데이터 생성 코드 위치 | "fully open-source pipeline" (초록) | 이 저장소에는 데이터 생성 코드가 없고 MolmoSpaces 저장소(`scripts/datagen`, `projects`)에 있음. 이 저장소 README의 "Using MolmoBot Data"는 다운로드, 검증, 통계 계산만 다룸 | **위치 안내 (미읽음)** | 엔진 자체의 재현성은 이 문서에서 확인하지 못함 |

요약: 학습 구조(층별 교차 주의 DiT, 가중 샘플링, 이력 간격, 데이터 혼합 비율, 증강, 생성 상수)는 논문과 맞습니다. 차이는 (a) 논문에 없는 **학습률, LLM 미세조정, 정규화, 카메라 선택 규칙**(D1, D2, D6, D7), (b) 평가 경로의 **관절 변화 상한 0.2 rad, 그리퍼 이진화, 시간 한도 설정이 둘**(D12–D15), (c) RB-Y1 행동 공간이 논문 서술보다 단순함(D10)에 몰려 있습니다. RB-Y1 모델의 표 6 혼합 학습 명령과 π0.5-Finetune 15k 명령은 공개 코드에서 찾지 못했습니다.

## 3. 평가 프로토콜 (읽은 범위)

- **카메라.** 평가 구성 클래스는 "에피소드별 카메라, 자세, 과제 변수는 벤치마크가 정한다"고 주석에 적고, MolmoBot-Pi0 평가는 `policy/cameras=sim_exo_shoulder`(고정 어깨)와 `sim_exo_random` 두 가지를 둡니다. 표 1의 "DROID-Light"(고정 어깨)와 "RC"(무작위 외부)에 대응하는 것으로 보입니다. 학습은 무작위 외부 카메라이므로 Pick v1, Pick v2, PnP 칸은 학습 분포에서 벗어난 평가이고, 표 11의 "제한 카메라"와 같은 설정입니다.
- **성공 판정.** v3 본문은 오라클을 "과제 달성이 어느 시점이든"으로 정의합니다. v1은 "5 스텝 연속", MolmoBot-Pi0 평가 스크립트는 `min_success_duration 0.5` s(15 Hz에서 약 8스텝)를 기본으로 둡니다. 세 서술은 같지 않으므로 오라클과 종료 시 성공의 정확한 정의는 평가 코드(MolmoSpaces)에서 확인해야 합니다.
- **실로봇.** `robot_eval/scripts/droid/run_policy.py`는 정책 주기 0.066 s, 이미지를 1280×720에서 640×360으로 줄임(downsample), 그리퍼 속도 0.085, 힘 10, 시작 자세 `[0, −0.7, 0, −2.356, 0, 1.571, 0]`로 되어 있습니다. 논문의 해상도와 주기(C.1)와 맞습니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 논문 수준의 학습·평가·실로봇 재현은 불가능하고, 구조 읽기와 수치 재계산은 끝냈습니다. 작은 정책(MolmoBot-SPOC)의 추론 확인은 조건부로 가능합니다.**

| 수준 | 가능성 | 근거 |
|---|---|---|
| 공식 코드 정적 읽기, 수치 재계산 | **완료** | 이 문서와 `03` |
| MolmoBot(4B 계열) 학습 | **불가** | README가 H100 8–64장. 학습 스크립트는 FSDP2, `torchrun`, 컴파일 설정이 CUDA 전제 |
| MolmoBot-Pi0 학습 | **불가** | README가 H100 32장 |
| 학습 데이터 전부 사용 | **불가** | Hugging Face 파일 합 약 10.3 TB. 맥의 디스크와 무관하게 의미 있는 규모가 아님 |
| 데이터 일부(샤드 1개) 내려받아 형식 읽기 | **조건부** | `bulk_download.py --max_part_shards 1`. 설정마다 수 GB(Franka PnP는 파일 1,605개에 3.47 TB, 파일당 약 2.2 GB). 용량 보고 후 승인 필요 |
| MolmoBot-SPOC(108M) 추론 | **조건부** | 체크포인트 약 0.4 GB. conda Python 3.11과 `pip install -e .`, 시뮬 평가는 MolmoSpaces 설치 필요. 맥 CPU/MPS 속도는 미측정 |
| MolmoBot-DROID(4B + DiT) 추론 | **조건부 (어려움)** | 체크포인트 약 20 GB(fp32로 보임), `inference_wrapper`는 장치를 `cuda` 또는 `cpu`만 고르고 `torch.autocast(device_type='cuda')`가 코드에 있어 MPS는 수정이 필요. CPU 지연은 미측정 |
| MolmoSpaces에서 소량 Franka 에피소드 생성(MuJoCo, CPU) | **조건부** | 파이썬 패키지와 자산(수 GB) 내려받기 필요. RB-Y1은 CuRobo(CUDA) 전제 |
| 실로봇 평가(표 8, 표 9) | **불가** | DROID Franka 3대와 RB-Y1, 평가 환경 |
| 작은 토이(예: 카메라 무작위화 범위와 전이) | **가능** | torch 2.15, mujoco 4.0 설치 완료. 이번 분석에서는 하지 않았습니다(논문의 핵심 주장 하나를 곧바로 검증하는 토이가 아니기 때문) |

**노력 추정(측정 전, 추정치).**

- SPOC 추론 확인: 설치와 다운로드 승인 포함 2–4일. 실제 정책 평가는 MolmoSpaces 설치 후 추가 1주.
- MolmoBot-DROID CPU 추론 지연 측정: 변환(bf16)과 장치 수정 포함 2–3일.
- 데이터 일부 읽기와 통계 확인: 샤드 1개 기준 1–2일.
- 학습 재현: 이 맥에서는 의미 있는 규모가 없습니다. 소규모 모방 학습 재구현(Franka 1과제, 작은 VLM 또는 SPOC 계열)은 2–3주(추정)이고 논문 주장의 근거가 되지는 않습니다.

## 5. 재실행

이 문서의 검사는 파일 읽기로 끝납니다. 공식 저장소를 다시 얻으려면 다음 명령을 쓰되 실행하지는 마십시오.

```bash
git clone --depth 1 https://github.com/allenai/MolmoBot repo   # 분석 시점 커밋: be82891b50437fb081589fe3be353567241053da
```

# 04. 코드 대조: 논문 ↔ AdeeshDesai/CONTACT (d0ad7f5)

공식 저장소를 얕게 클론했다(`repo/CONTACT/`, git 제외, 커밋 `d0ad7f5`, 2026-07-11, MIT). 저장소는 커밋 1개짜리 얕은 복사본이다. 데이터(Google Drive), 체크포인트, 그리고 의존 저장소(`thirdparty/`: IsaacGym, manifeel-isaacgymenvs, diffusion_policy)는 받지 않았다. 따라서 `diffusion_policy` 내부 코드(MultiImageObsEncoder, DDIM 샘플러, 다운샘플링)는 **표준 구현을 기준으로 추정**했고 표시했다.

## 1. 확인한 파일

| 항목 | 파일 |
|---|---|
| 학습 스크립트 | `training/<task>/{vision,vistac,visff}.sh`, `scripts/run_all.sh` |
| 설정 | `contact/config/train_diffusion_workspace_disassembly.yaml`, `contact/config/task/{vision,vistac,visff}_disassembly.yaml`, `contact/config/task/TacSLTaskBarbedFlat.yaml` |
| 데이터 | `contact/dataset/contact_dataset_equidp.py` |
| 평가 | `contact/env_runner/vistac_pih_runner.py`(vision), `vistac_pih_runner_unit.py`(vistac, visff), `contact/gym_util/multistep_wrapper.py`, `contact/envs/vistac_isaacgym_multiple_env_wrapper.py` |
| 과제 환경 | `isaacgymenvs/tasks/tacsl/tacsl_task_barbed_flat.py`, `tacsl_env_barbed_flat.py` |

## 2. 논문 ↔ 코드 대조 (D-항목)

판정 기호: **일치**, **불일치**, **미공개**(코드·설정 어디에도 없음), **추정**(표준 구현 기준, 미클론 코드 필요).

| ID | 항목 | 논문 | 코드 | 판정 | 영향 |
|---|---|---|---|---|---|
| **D1** | 학습 데모 수 | 시뮬 수 미명시. 실물 50개(Sec. III-A.2) | 시뮬 `max_train_episodes`: vision **90**, vistac **90**, visff **30** (`vision_disassembly.yaml:41`, `visff_disassembly.yaml:45`). 부분 표집은 `downsample_mask`로 시드 42 고정 | **불일치(통제 깨짐)** | TacFF 정책은 같은 과제에서 데모의 1/3만 본다. 시드 42로 한 번 뽑힌 30개 고정 |
| **D2** | 학습 epoch | 실물 400 epoch(Sec. III-A.2). 시뮬은 없음 | `num_epochs = int(50000 / n_demo)` (`train_..._disassembly.yaml:108`): vision·vistac 555, visff **1666** | **불일치** | 총 샘플 제시량은 대략 같다(약 50,000 episode-epoch). 데모 수가 다르므로 과적합 정도도 다르다 |
| **D3** | 평가 체크포인트 | 시뮬: 마지막 10 체크포인트 평균(Table I 설명), 다른 곳은 "final 10 training epochs" | `rollout_every=50`, `checkpoint_every=50`, top-k 저장은 **train_loss** 기준(`checkpoint.topk`) | **추정 불일치** | 평균 창이 설정마다 다르다. visff는 약 33개 평가 중 마지막 10개(약 epoch 1200–1650), vision·vistac은 약 11개 중 마지막 10개(약 epoch 100–550)로 추정. 평가 창이 다르면 비교가 바뀐다 |
| **D4** | 추론 확산 스텝 | DDIM, K = 10 (Sec. II-C 끝) | `num_inference_steps: 100`, 스케줄러 `DDPMScheduler`(`train_..._disassembly.yaml:59, 28–35`) | **불일치** | 샘플러가 DDPM 100스텝일 가능성이 크다(`diffusion_policy` 미클론이라 추정). 이 경우 지연과 샘플이 논문과 다르다 |
| **D5** | 행동 차원 | 5차원(dx, dy, dz, dθz, g), 회전은 yaw만(Sec. II-C) | `action.shape: [7]`; `_apply_actions_as_ctrl_targets`가 위치 3 + 축-각 회전 3 + 그리퍼 1 사용 (`tacsl_task_barbed_flat.py:854–862`) | **불일치** | 코드는 3축 회전을 쓴다. yaw 제한이 구현에 없다 |
| **D6** | 입력 해상도 | 정면·손목 240×320, TacRGB 240×160(Sec. III-A) | 설정상 이미지 256×256(`vision_disassembly.yaml`), obsDims 촉각 `[320, 240, 3]`, `tactile_size` 256 또는 224 | **불일치** | 재현 시 입력 크기를 논문값으로 맞출지 결정 필요 |
| **D7** | 인코더 | ResNet-18(사전학습 여부 미기재) | `resnet18`, `weights: null`(**ImageNet 미사용**), GroupNorm, 키마다 별도 인코더(`share_rgb_model: False`), `random_crop: True`, `crop_shape: null` | **일치(구조), 불일치(사전학습 미기재)** | TacFF 10×14 입력은 ResNet-18을 거치면 공간 차원이 1×1까지 줄어든다. 사실상 전역 풀링 특징 |
| **D8** | 성공 기준 | 완료: "fully extracted and lifted above a predefined height threshold"(Sec. III-A.1) | `task_type: insertion` 전부. `_check_success` → `_check_plug_close_to_socket`: 4개 키포인트 평균 거리 **> 0.1 m**일 때 성공(높이 조건 없음). 함수 이름은 "close"지만 실제로는 멀어진 경우를 본다 | **불일치** | 성공 = "10 cm 이상 분리"이고 들어 올리기는 확인하지 않는다. 논문 정의보다 느슨할 수 있다 |
| **D9** | 평가 집계 | 50 초기조건 × 체크포인트 × 시드 | 실행 루프는 `for i in range(n_envs)`(`n_envs = n_test = 50`). 주석은 논문 수치가 주석 처리한 `range(len(self.env_fns))` 루프로 나왔다고 쓴다(`vistac_pih_runner_unit.py` 근처 주석) | **추정** | 저자는 "평균에는 영향이 없다"고 주석에 적었다. 실제 env 수가 다르면 평가 표본이 달라진다. 확인 불가 |
| **D10** | 시험 초기조건 | 체크포인트당 50개, 시드 3개로 "1500 independent rollouts"(Sec. III-B) | `test_start_seed: 100000`이 **모든 실행에서 같다**(`*_disassembly.yaml`) | **불일치(독립성)** | 같은 50개 초기조건이 모든 체크포인트·시드·모달리티에 반복된다. 1500은 독립 표본이 아니다(`03` §2.2) |
| **D11** | 데이터 부분 표집 시드 | 시드 3개 | `dataset.seed: 42`, `downsample_mask(seed=42)`, `val_ratio 0.02`(고정). 학습 시드는 `training.seed`로만 바뀐다 | **추정** | 세 시드는 신경망 초기화만 바꾼다. 데모 부분 집합 변동은 포함되지 않는다 |
| **D12** | 실물 학습·평가 | 실물 R1–R5, 과제당 시연 50, 20회 시험, TacFF 계산(광흐름·깊이 복원) | 실물 과제 설정, 실물 학습 스크립트, 광흐름·깊이 복원 코드 **없음**. `grep optical` 결과 0건. 실물 데이터셋 로더(`real_*_dataset.py`)와 텔레옵 데모(`demo_teleop*.py`)만 있다 | **미공개** | 실물 표는 코드로 검증할 수 없다 |
| **D13** | 시뮬 TacFF의 출처 | "광흐름과 깊이 복원에서 계산"(Sec. II-B). 시뮬 설명은 없음 | 시뮬 TacFF는 시뮬레이터 접촉 힘이다(`setup_tactile_force_field`, `use_shear_force`, `get_tactile_shear_force_fields`: 법선·전단 격자). TacRGB만 TacSL 렌더링을 쓴다 | **불일치(정보원)** | 시뮬 TacFF는 물리적으로 정확한 접촉 힘이다. 실물 TacFF는 영상 추정이다. 시뮬의 TacFF 우위는 실물보다 낙관적일 수 있다 |
| **D14** | 힘장 정규화 | "[-1, 1]로 정규화"(Sec. III-A) | TacFF를 `rgb` 키로 등록. `get_image_range_normalizer()`는 [0, 1] → [-1, 1] 고정 선형 변환이고, 원단위 힘에는 데이터 기반 정규화가 없다. `cv2.resize`로 10×14에 맞춘다(`contact_dataset_equidp.py:122–150`) | **불일치(추정)** | 힘의 실제 단위가 논문에 없다. 시각화 임계값 0.0008(`get_tactile_force_field_tensors_dict`의 debug)은 힘 값이 약 1e-3 규모일 수 있음을 시사한다. `03` §7 E2 참조 |
| **D15** | 평가 러너 차이 | 세 설정은 "같은 학습 틀"(Sec. III-A) | vision은 `vistac_pih_runner.py`, 촉각 두 설정은 `vistac_pih_runner_unit.py`. 차이는 촉각 크기 조정 루프와 `time.sleep(2)`, `past_action` 주석 정도 | **일치(사소한 차이)** | 로직상 같은 평가 |
| **D16** | 평가 중 회전 행동 | yaw만 사용 | `vistac_pih_runner.py:147`의 `action[:, :, 3:6] = 0`이 주석 처리됨 | **불일치(사소)** | 회전 행동이 평가에서 실행된다 |
| **D17** | 상태·이력·호라이즌 | 상태 7차원, 이력 2, H = 16, 실행 8 | `state [7]`, `n_obs_steps 2`, `horizon 16`, `n_action_steps 8` | **일치** | — |
| **D18** | 학습 하이퍼파라미터 | 논문 없음 | AdamW, lr 8e-5, betas (0.95, 0.999), weight decay 1e-6, 코사인 스케줄, warmup 500, EMA(max 0.9999), 배치 8, `down_dims [512, 1024, 2048]`, `diffusion_step_embed_dim 128`, `kernel_size 5`, `n_groups 8` | **미공개(논문) / 코드에 있음** | 재구현 시 이 값을 기준으로 삼을 수 있다 |
| **D19** | 의존 버전 | 논문 없음 | `install.sh`가 `diffusion_policy`와 `manifeel-isaacgymenvs`를 **고정 커밋 없이** 클론. `requirements.txt`는 numpy 1.23.3, diffusers 0.11.1, opencv 4.10 등. 환경은 Python 3.8 | **재현 장벽** | 현 시점 버전으로 설치하면 결과가 달라질 수 있다 |
| **D20** | 라이선스 | 논문 PDF: arXiv 비독점 | 저장소 MIT. 의존: manifeel-isaacgymenvs NOASSERTION, tacsl 없음, diffusion_policy MIT, IsaacGym는 NVIDIA 자체 라이선스 | **확인** | 코드 재사용은 MIT를 따른다. 의존 라이선스는 별도로 확인해야 한다 |

일치 항목(D17, D15)과 불일치 항목을 함께 보면, 코드는 논문 방법을 대체로 구현했다. 그러나 **실험 통제(D1, D2, D10)와 측정 정의(D8, D9)가 논문 문장과 다르다.** 즉 방법은 맞지만 "통제된 비교"라는 주장이 코드로는 성립하지 않는다.

## 3. 재현에 필요한데 논문에 없는 정보

1. 시뮬 데모 수와 실물 데모 수(논문은 실물만 50개).
2. 힘장의 단위, 정규화 통계, 실물 계산 파이프라인(광흐름 알고리즘, 깊이 복원, 보정).
3. 시드 3개의 값과 시드별 결과. 체크포인트별 결과.
4. 확산 샘플러 종류와 스텝 수(D4).
5. 사전학습 가중치 사용 여부(D7).
6. 실물 학습 설정과 평가 기록(D12).
7. 시드 간 표준편차(03 §2.3에서 필요 조건으로만 제시).

## 4. 구현할 수 있는지

**판정: 공식 재현은 이 맥에서 불가능하다. 아이디어 재구현은 중간 난이도다. 실험의 핵심 가치는 재구현이 아니라 통제 실험을 다시 설계하는 데 있다.**

### 4.1 공식 코드로 재현 (시뮬)

- 필요: Linux, NVIDIA GPU(CUDA), IsaacGym(NVIDIA 다운로드), Python 3.8, conda, TacSL 자산, manifeel-isaacgymenvs, Google Drive 데이터(과제별 zip).
- 규모: 3 모달리티 × 5 과제 × 3 시드 = 45 실행. 1회 실행은 555 또는 1666 epoch이다(D2). 평가는 과제별 50 초기조건.
- **이 맥(Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음, torch 미설치):** IsaacGym은 macOS와 CUDA 없이 실행할 수 없다. **불가능.**
- 자원 추정: 과제당 학습 시간과 GPU 시간은 논문에 없다. 45 실행을 GPU 한 장으로 돌리면 수일 이상으로 추정(근거 없는 추정, 확인 필요).

### 4.2 공식 코드로 재현 (실물)

- 코드 없음(D12). 필요 하드웨어: Franka 팔, GelSight R1.5, RealSense D415 두 대, 3Dconnexion SpaceMouse 텔레옵, 과제 소품.
- **불가능.**

### 4.3 아이디어만 다시 구현

- 쉬움~중간: 확산 정책(1D U-Net)은 공개 구현으로 가능하다. 모달리티별 인코더, 특징 이어 붙이기, 힘장 격자(10×14×3) 입력은 수백 줄이다. 시뮬러터 없이 쓸 수 있는 부분은 정책 코드와 토이다.
- 이 맥에서 할 수 있는 것: torch 설치(승인 필요) 후 MPS로 소형 정책 학습. 단, 학습 데이터(시연)가 필요하다. 공개 데이터는 Google Drive 다운로드(승인 필요)이거나 직접 수집해야 한다.
- 이 맥에서 할 수 없는 것: IsaacGym 기반 폐루프 평가, 실물 GelSight 수집.
- 통제 설계 재현(가장 가치 있음): 데모 수를 맞추고, 시드를 5–10개로 늘리고, 시험 초기조건을 시드별로 바꾸면, D1·D10·§2.3의 문제를 직접 검증할 수 있다. 이것은 IsaacGym 없이도 **오프라인 시연 데이터**가 있으면 가능하다(§4.3 참조).

### 4.4 요약 표

| 경로 | 필요 자원 | 이 맥 | 판정 |
|---|---|---|---|
| 시뮬 공식 재현 | Linux, CUDA GPU, IsaacGym, 데이터 다운로드, 45 실행 | 불가(CUDA·IsaacGym) | 불가 |
| 실물 공식 재현 | 로봇, 촉각 센서 3종, 코드 없음 | 불가 | 불가 |
| 아이디어 재구현 | torch, 시연 데이터 | 조건부(설치·데이터 승인 후 MPS) | 중간 |
| 통제 실험 재설계 | 시연 데이터(오프라인), numpy·torch | 조건부 | 가치 높음 |

이 맥에서 끝낸 것: 표 재계산(`table_check.py`), 두 토이(`concat_toy.py`, `lowdim_toy.py`), 코드 대조(이 문서), 설정 확인.

## 5. 권리 표기 (license verdict)

- arXiv abs 페이지: "view license" → `nonexclusive-distrib/1.0`(비독점 배포). **CC BY 아님.** 따라서 논문 그림(Fig. 1–5, Table 이미지)은 개인용 리포트와 PPT에만 넣었고, 게시본과 `report_public.pdf`에서는 뺐다.
- PDF 본문 검사: 저작권·IEEE·재인쇄 문구를 찾지 못했다(추출 텍스트에서 `copyright`, `IEEE`(참고문헌 제외), `permission` 검색). 다만 PDF에는 arXiv 스탬프만 있다.
- 프로젝트 페이지 하단에 `creativecommons.org/licenses/by-sa/4.0` 링크가 있다. 사이트 템플릿(nerfies 계열) 표기로 보이며, 논문 PDF의 권리와는 별개다. 확인 불가.
- 코드: MIT(저장소 `LICENSE`). 코드 라이선스는 논문 그림과 무관하다.
- 게시본에는 논문 그림, 원문 PDF, 논문 원문 발췌를 넣지 않았다.

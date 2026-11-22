# 04. 공식 코드 대조와 구현 가능성

공식 코드는 로컬 얕은 복제(`--depth 1`, gitignore)로 받았고, **정적으로만 읽었습니다.** 설치, import, 실행, 데이터·체크포인트 다운로드는 하지 않았습니다. 코드는 신뢰하지 않는 외부 자료로 취급했습니다.

## 1. 저장소 상태

- 저장소: [YanjieZe/3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy), 기본 브랜치 `master`, 복제한 커밋 `47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97`(2025-10-16 PDT, "Update README with community applications and extensions").
- 라이선스: MIT (루트 `LICENSE`, GitHub API). `third_party/`에는 mujoco-py, gym, MetaWorld, VRL3, dexart-release 등 별도 라이선스 파일이 있습니다.
- 크기: 복제본 136 MB, DP3 본체 `3D-Diffusion-Policy/`는 668 KB입니다. 나머지는 `third_party/`입니다.
- **시점 불일치:** 논문 v7은 2024-09-27입니다. 저장소 최신 커밋은 2025-10-16입니다. 결과를 낸 코드가 어느 커밋인지 확인하지 못했습니다. 아래 대조는 이 커밋을 기준으로 합니다.

## 2. 논문과 코드의 차이 (D 목록)

| ID | 항목 | 논문 | 공식 코드 (파일) | 판정 |
|---|---|---|---|---|
| D1 | 예측·실행 지평 | 부록 A: H = 4, Nobs = 2, Nact = 3 (DP3와 DP 모두) | `config/dp3.yaml`과 `simple_dp3.yaml`: horizon 16, n_action_steps 8. 과제별 yaml에 재정의 없음 | **어긋남.** 기본값은 논문 표 XV의 "long horizon" 설정입니다 |
| D2 | 학습 에폭 | MetaWorld 1000, 나머지 3000 (§III-C) | 모든 과제 `num_epochs: 3000`. `task/*.yaml`과 `scripts/`에 재정의 없음 | **어긋남** (MetaWorld 1000 미반영) |
| D3 | 옵티마이저·스케줄 | 논문에 없음 (배치 128만 적혀 있음) | AdamW, lr 1e-4, betas (0.95, 0.999), weight decay 1e-6, cosine, warmup 500, EMA (max 0.9999, power 0.75) | **논문 누락** |
| D4 | 시연 수 | 시뮬 10회 (DexArt, HORA 100회), 실로봇 40회 (표 III) | `max_train_episodes: 90`. 데이터셋이 학습 에피소드 중 최대 90개를 무작위(seed 42)로 고름. zarr 파일은 저장소에 없음 | **확인 불가.** 실제 시연 수는 zarr 에피소드 수에 달려 있음 |
| D5 | 평가 지표 | 20 에피소드씩 200 에폭마다 평가, 상위 5개 평균 (§IV-A) | `eval_episodes: 20`, `rollout_every: 200`. `LargestKRecorder(K=5)`로 `SR_test_L5` 계산 (`env_runner/*_runner.py`) | 일치. **선택 편향 있음.** 평가 점수를 골라 쓰므로 낙관적 |
| D6 | 점군 샘플링 | FPS, 512 또는 1024점 (§III-B) | 시뮬(`gym_util/mjpc_wrapper.py`): `sample_farthest_points`(pytorch3d) 사용, `.cuda()` 고정. 'uniform'도 있음. **점이 모자라면 0으로 채움**(패딩) | 일치. 패딩과 CUDA 고정은 논문에 없음 |
| D7 | 자르기 상자 | 작업공간 상자로 자름 (§III-B), 경계값 없음 | 환경별 `min_bound`, `max_bound`가 코드에 하드코딩 (`mjpc_wrapper.py` 등). 실로봇 자르기는 `scripts/convert_real_robot_data.py`와 README 안내에 맡김 | **재현 정보 부족** |
| D8 | 색 | 색 미사용 (§III-B) | `use_pc_color: false`. 인코더는 xyzrgb도 지원 (`PointNetEncoderXYZRGB`) | 일치 |
| D9 | 점 인코더 | 3층 MLP (3→64→128→256), LayerNorm, 최대 풀링, 투영 64 (§III-B, 부록 A 코드) | `PointNetEncoderXYZ`의 `block_channel = [64, 128, 256]`, 투영 64, `final_norm: layernorm` | 일치 |
| D10 | 로봇 상태 MLP | 64차원 (부록 A) | `state_mlp_size (64, 64)`. 출력 128 = 점 64 + 상태 64 | 일치 |
| D11 | 확산 설정 | DDIM, 학습 100, 추론 10, 샘플 예측 (§III-C) | `DDIMScheduler` 100 / 10, `prediction_type: sample`, `clip_sample: True`. U-Net 채널 512/1024/2048 (논문에 U-Net 크기 없음) | 일치 (U-Net 크기는 논문 누락) |
| D12 | 기준선 DP | "공식 PyTorch 구현 [10]" 사용 (부록 A) | 이 저장소에는 2D DP 구현이 없음 | 확인 불가. 기준선 설정이 DP3와 같은지 알 수 없음 |
| D13 | Simple DP3 | 본문에 설계 없음, "UNet 백본 단순화로 2배" (부록 D) | `simple_dp3.yaml`은 U-Net 채널만 128/256/384로 줄임. 나머지 설정은 `dp3.yaml`과 같음 (diff 확인) | 일치. 논문 본문에는 없는 정보 |
| D14 | 실로봇 제어 | Franka와 Allegro, RealSense L515, 원격조작 (§V-A) | 저장소에 제어 코드와 원격조작 코드 없음. `realdex_*.yaml`은 zarr 데이터셋을 가리킴. 데이터는 Google Drive | **재현 불가 (저장소 밖)** |
| D15 | 시연 생성 | RL 전문가 (VRL3, PPO), MetaWorld 스크립트, DexDeform 사람 원격조작 (§IV-A) | `scripts/gen_demonstration_{adroit,dexart,metaworld}.sh`와 `third_party/VRL3`. DexDeform 원격조작 코드 없음 | 부분 일치 |
| D16 | 시드 | 0, 1, 2 (§IV-A) | `scripts/train_policy.sh`가 seed 인자를 받음. 기본값 강제 없음 | 일치 |
| D17 | 점 인코더 비교 | PointNet++, PointNeXt, Point Transformer 등 (표 V) | 공식 저장소에 PointNet++, PointNeXt, Point Transformer 구현과 사전학습 설정이 없음 (파일명과 코드 검색으로 확인). 표 V의 비교 코드는 저장소 밖 | 확인 불가 |
| D18 | 장치 | 논문에 명시 없음 | 기본 `training.device: cuda:0`, `train_policy.sh`가 `CUDA_VISIBLE_DEVICES`를 설정. 점 샘플링도 CUDA 고정 | **MPS에서 바로 안 돌아감** |

요약하면 **D1, D2 (설정 불일치), D3, D7, D14 (재현 정보 부족), D4 (시연 수 확인 불가)** 가 중요합니다. 특히 D1은 논문 주 설정과 저장소 기본 설정이 다르다는 뜻이라, 표 I과 표 II를 어떤 설정으로 냈는지 확인해야 합니다.

## 3. 학습 루프와 평가

- `train.py`: 200 에폭마다 `rollout_every`로 환경 평가를 하고, 그 사이에는 학습 손실로 `test_mean_score`를 대신합니다(`if env_runner is None: step_log['test_mean_score'] = -train_loss`). 환경 평가가 있는 과제에서는 평가 성공률이 쓰입니다.
- 체크포인트는 기본으로 저장하지 않습니다(`checkpoint.save_ckpt: False`). 최고 점수 체크포인트를 고르는 `TopKCheckpointManager`는 저장을 켰을 때만 동작합니다.
- 공식 저장소에 학습된 정책 체크포인트가 없어서, 논문 표를 추론만으로 다시 확인할 수 없습니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 논문 표를 이 맥에서 재현하는 것은 불가능합니다. 재계산과 정적 대조는 끝냈습니다. 인코더와 결정부의 소형 재구현은 가능하지만 설치 승인이 먼저 필요합니다.**

| 작업 | 가능 여부 | 근거 |
|---|---|---|
| 표 재계산과 검정 | 가능, **완료** | `03`, `analysis/recompute.py` |
| 공식 코드 설치와 실행 | **이 맥에서는 어려움** | INSTALL.md가 Python 3.8, CUDA 11.7 등을 요구. mujoco-py 2.1(MuJoCo 2.1 바이너리), gym 0.21 특수 버전, pytorch3d_simplified(CUDA 확장). 설치하지 않았음 |
| 인코더 forward 확인 (무작위 점군) | 조건부 | torch 설치와 공식 코드 import가 필요. 승인 전 보류 |
| 인코더와 결정부 torch 재구현 | 가능 (MPS) | 구조가 본문과 부록 A 코드로 충분히 정해짐. 노력 3–5일 (추정) |
| 소형 학습 (과제 1개, 시연 10회) | 조건부 | MPS 속도는 측정 전. 설치 승인 필요 |
| 72 과제 학습 (3000 에폭, 시드 3) | **불가** | GPU 시간을 수백 시간 이상으로 추정 (측정 전) |
| 실로봇 4 과제 | **불가** | Franka, Allegro, L515, 원격조작 장치, 과제당 40회 시연 |
| 시연 데이터 다운로드 | 보류 | OneDrive와 Google Drive 링크. 크기 미확인, 받지 않음 |

- **노력 추정 (측정 전):** 인코더와 결정부 재구현 3–5일, 소형 과제 학습 1주, 공식 코드 이식 (CUDA 고정 제거, 패딩과 자르기 설정 맞춤) 1–2주입니다. 이 추정은 측정하지 않은 값입니다.
- 구현 우선순위: (1) D1과 D2를 해결하지 않으면 표 I을 다시 맞출 수 없습니다. (2) D7의 자르기 상자가 없으면 실로봇 일반화 실험을 다시 할 수 없습니다. (3) D14는 저장소 밖 장비가 있어야 합니다.

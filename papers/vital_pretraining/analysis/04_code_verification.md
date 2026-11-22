# 04. 공식 코드 대조 (정적 읽기)

대상: [Abraham190137/TactileACT](https://github.com/Abraham190137/TactileACT), MIT 라이선스, 커밋 `5a533f76f15c54dbb66e19fbe30b7318b2281211`(2024-08-23), 얕은 복제 `repo/`. **코드를 실행하거나 설치하지 않았습니다.** 모든 판단은 파일 읽기로 했고, 파일 경로와 줄 번호는 `repo/` 기준입니다. 대조 기준은 논문 v2입니다(`01` §2).

코드 시점은 논문 v2(2024-09-26)보다 앞섭니다(마지막 커밋 2024-08-23). 불일치 중 일부는 버전 차이일 수 있습니다. 그래도 논문에 적힌 값과 코드 값이 다르면 "차이"로 기록합니다.

## 1. 대조 결과 (D1–D18)

판정: **일치**, **차이**(논문에 값이 다르거나 코드만 다른 설정), **미기재**(코드에는 있고 논문에 설명 없음), **확인 불가**(데이터나 체크포인트가 없어 검증 불가).

| ID | 항목 | 논문 (v2) | 코드 (`repo/`) | 판정 |
|---|---|---|---|---|
| D1 | 시점 샘플 수 `n` | 7 (§III-A, "we used 7") | `clip_pretraining.py:608` `n_clip_images = 5` | **차이** |
| D2 | 시점 최소 간격 `Δt_min` | 10 스텝 ≈ 1초 (§III-A) | `clip_pretraining.py:609` `min_distance = 20` (정의부 기본값은 `:126`의 5) | **차이** (두 배) |
| D3 | 대비 손실 온도 `τ` | 값 없음, `exp(sim/τ)` | `clip_pretraining.py:359`의 `logit_scale = 1.0`, 온도 1. 학습 때 호출 시 기본값 유지 | **미기재** (온도 1의 효과는 `03` §7) |
| D4 | 손실 정규화 | `1/(2n) Σ_c` (카메라 합) | `clip_loss` 끝: 두 방향 평균 후 카메라 평균 (`:391`) | **차이** (상수 배율, 카메라 수 6의 역수 배) |
| D5 | ResNet-18 정규화 | 언급 없음 | `clip_pretraining.py:96–111`에서 BN을 GroupNorm(그룹당 16채널)으로 바꿈 | **미기재** |
| D6 | 학습률, 옵티마이저, 에폭 | 언급 없음 | 대비: Adam, `resnet_lr 1e-5`, `projection_lr 1e-4`, 배치 3, 에폭 1501 (`:410–411`, `:651`). ACT: `lr 1e-5`, `weight_decay 1e-4`, `kl_weight 10`, `z 32`, `hidden 512`, `enc 4`/`dec 7`, `dropout 0.025`, `chunk 20` (`base_config.json`). README 예시는 `chunk 30` | **미기재** |
| D7 | 촉각 인코더 동결 (ACT) | 언급 없음 | `imitate_episodes.py:31` `FREEZE_TACTILE = True`, `:107–109`에서 촉각 백본 `requires_grad_(False)` | **미기재** (결정적) |
| D7' | 촉각 인코더 동결 (DP) | 언급 없음 | `diffusion/train_policy.py`에는 동결 코드가 없음. 인코더 전부 학습 | **차이** (ACT와 DP가 다르게 처리) |
| D8 | 정지 마찰 보정 (적분항) | 언급 없음 | `robot_operation.py:483–487`. 움직임이 0.0025 m 미만이면 적분항 누적, ±0.015 m 클리핑, z 음수 방향 차단 | **미기재** |
| D9 | 추론 잡음 2.5 mm | "표준편차 2.5 mm" (§IV-A) | `robot_operation.py:352–353`, `:500`, `noise_std = 0.0025` | **일치** |
| D10 | 시간 앙상블 `k = 0.25` | `w_i = e^{-k i}`, `k = 0.25` (§III-B1) | `robot_operation.py:354`, `:459–461` `confidences[t] *= np.exp(-K)` | **일치** |
| D11 | 상대 위치 목표 | `goal − pos_t` (§III-B1) | `utils.py:51–52`에서 `delta = action − qpos` (그리퍼 제외), 추론 때 현재 위치를 다시 더함 (`utils.py:70–72`) | **일치** |
| D12 | 카메라 크롭과 배경 마스크 | 언급 없음 | `process_data_cage.py:12–22` `CROP_PARAMS`, `MASK_VERTICIES` (카메라 5, 6). `robot_operation.py:39–42`에서 크롭과 90도 회전, `:122` 마스크 적용 | **미기재** |
| D13 | 촉각 입력 해상도 240×320 | 240×320 변형률 지도 (§III-C) | 키 `observations/gelsight/depth_strain_image`, 해상도 상수는 HDF5 속성에서 읽음 (`clip_pretraining.py:162`, `:195`). 240×320 값은 코드에 없음 | **확인 불가** (데이터 형식 미공개) |
| D14 | 촉각 정규화 통계 | 언급 없음 | `gelsight_mean`, `gelsight_std`를 `norm_stats`에서 가져옴 (`clip_pretraining.py:143`) | **미기재** |
| D15 | 80/20 학습·검증 분할 | 80/20 (§IV-A) | 에피소드 단위 80/20 무작위 분할 (`clip_pretraining.py:612–616`, 100 에피소드) | **일치** |
| D16 | 카메라 6대 | 6 RealSense (§III-C) | `camera_names = ['1'..'6']` (`clip_pretraining.py:607`), 촉각은 별도 카메라 | **일치** |
| D17 | 평가 중단 300 스텝 | 300 스텝 (§IV-A) | `robot_operation.py:526` `if i >= 300` 뒤 수동 확인 | **일치** |
| D18 | DP 지평과 스케줄 | 관측 1, 행동 20, 실행 8, DDPM 100/10 (§III-B2) | 관측 1 (`train_policy.py:41`), 예측 20 (`train_args.py:33`), 실행 `actions[:8]` (`robot_operation.py:376`), DDPM 100 (`train_policy.py:202`), 추론 10 (`predict_robot_actions.py:87`) | **일치** |
| D19 | DP 학습률과 스케줄 | 언급 없음 | `lr=1e-4`, `weight_decay=1e-6`, 워밍업 500 (`train_policy.py:245–252`), EMA 주석 처리 (`:235`, `:449–450`) | **미기재** |
| D20 | DP 시각 인코더 초기화 | 카메라마다 같은 가중치로 시작, 별도 미세조정 (§III-B2) | 같은 `image_weights`로 카메라별 인코더 생성 (`train_policy.py:50–56`) | **일치** |
| D21 | 블록 쌓기 과제 설정 | 1.5 cm 직사각형, 3 cm 정육면체, 시작 위치 무작위 (§III-C) | 과제 설정이나 데이터 처리 코드를 찾지 못함 (`stack`, `cube` 검색은 일반 단어만) | **확인 불가** |
| D22 | 촉각 전용 실험 | 있음 (§IV-A) | 시각 카메라를 빼는 설정 흔적 없음. 촉각+위치 전용 실행 설정은 저장소에 없음 | **확인 불가** |
| D23 | 위치 조건 제거 변형 | 언급 없음 | `clip_pretraining_no_pos.py` (투영 헤드 조건 주석 처리, `:429`) | **미기재** (저장소에만 있음) |
| D24 | 데이터, 체크포인트, 기준 학습 실행 로그 | 데이터 링크 없음 | 저장소에 없음. README는 `data/clip_models/*.pth`와 `data/data_dir`을 가정 | **확인 불가** |
| D25 | README 게재 상태 | arXiv 댓글: ICRA 2025 제출 | README "As the paper is under review" (2024년 기준 문구) | **차이** (최신 게재 상태 미반영) |

## 2. 가장 중요한 세 차이

1. **대비 사전학습의 시점 설정이 논문과 다릅니다 (D1, D2).** 논문 본문은 7개 시점과 1초 간격이라고 쓰지만, 실행 스크립트는 5개 시점과 2초(20스텝) 간격입니다. 논문 그림 7의 결과가 어느 설정으로 나왔는지는 알 수 없습니다. 재현할 때는 두 값을 모두 기록하고 어느 쪽을 쓸지 정해야 합니다.
2. **ACT와 DP가 촉각을 다르게 씁니다 (D7, D7').** ACT에서는 사전학습 촉각 인코더를 고정하고 쓰지만, DP에서는 미세조정합니다. 논문은 두 정책을 같은 방식으로 기술하므로, 두 결과를 같은 척도로 비교하기 어렵습니다.
3. **학습 손실의 절댓값이 해석에 쓰이기 어렵습니다 (D3).** 온도 1 때문에 정답 확률에 상한이 있고, 교차엔트로피 하한이 n = 5에서 0.433, n = 7에서 0.594입니다(`03` §7). 로그 손실을 보고 정렬이 되었다고 판단하면 안 됩니다.

## 3. 실행 흐름 확인 (정적)

- 사전학습: `run_clip_pretraining()` → `ClipDataset`(에피소드당 `n_clip_images`개 시점, 최소 간격 `min_distance`) → `clip_pretraining()`(두 ResNet-18 + 투영 헤드, 두 방향 대비 손실, Adam). 체크포인트는 `vision_encoder.pth`, `gelsight_encoder.pth`로 저장됩니다.
- ACT 학습: `imitate_episodes.py`가 사전학습 가중치를 읽음 (`:101–103`). 두 가중치 경로를 모두 지정해야 함 (`:104–105`).
- ACT 추론: `robot_operation.py`의 시간 앙상블, 적분 보정, 잡음 주입, 300 스텝 제한.
- DP 학습: `diffusion/train_policy.py`, 추론 `diffusion/robot_operation.py`.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**저장소 그대로 재현은 불가능합니다.** 데이터, 체크포인트, 로봇 스택이 공개되지 않았습니다. 손실과 샘플러를 다시 구현하는 것은 가능합니다.

- **공식 코드 실행 (사전학습, ACT, DP 학습): 불가 (데이터 없음).** 데이터와 체크포인트가 없습니다(D24). 저장소 스크립트에는 CUDA 대신 CPU 대체 경로가 있지만(`clip_pretraining.py:610`), 데이터가 없으면 시작할 수 없습니다.
- **손실, 샘플러, 시점 설정의 재구현 (합성 데이터): 가능, CPU 또는 MPS.** 논문 식과 D1–D4 설정을 모두 구현하고 합성 시퀀스에서 정렬 지표를 비교할 수 있습니다. 실로봇 데이터는 필요하지 않습니다. 새 설치가 필요하므로 설치 전에 무엇을 설치할지 보여 준 뒤 진행해야 합니다.
- **ACT 소형 학습 (합성 또는 시뮬레이션 데이터): 조건부.** 아키텍처는 논문보다 작게 가능합니다. 결과가 논문 수치와 비교되려면 실제 USB 데이터가 필요합니다.
- **실로봇 평가 (그림 7의 조건들): 불가.** Franka Panda, GelSight Mini, RealSense 6대, 원격조작 장비가 필요합니다.
- **노력 추정 (측정 전):** 손실, 샘플러, 시점 설정 재구현 1–2일. 합성 데이터에서 엔드투엔드 파이프라인 1주. 실로봇 재현은 장비 조달이 먼저입니다.

## 5. 라이선스와 권리 (정적 확인)

- 코드: MIT, `LICENSE` 파일 확인 (저작권 2024 Abraham George).
- 논문: arXiv abs 페이지의 "view license"가 `creativecommons.org/licenses/by-nc-sa/4.0/`를 가리킵니다. CC BY-NC-SA 4.0입니다. 그림은 게시본에 넣지 않습니다. 자세한 근거는 `README.md`에 있습니다.

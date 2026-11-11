# 04. 코드 검증: 논문 ↔ 공개 저장소

**방법.** 정적 읽기만 했습니다. 코드를 실행하거나 패키지를 설치하지 않았습니다.
- 저장소: `repo/volumedp_maniskill/` (`yzc0731/volumedp_maniskill`, 얕은 클론, 커밋 `d2ec13a`, 2026-06-22, Apache-2.0, 스타 0). 단일 커밋 "volumedp"이며 ManiSkill 3.0.0b19 구조입니다(`mani_skill/__init__.py`의 `__version__ = "3.0.0b19"`).
- 추적 파일 1,111개 중 VolumeDP 고유 구현 여부를 검색했습니다(`volumetric`, `voxel`, `deformable`, `TokenLearner`, `adaLN`, `DeformAttn`, `LIBERO`, `VolumeDP`, `volume_dp`를 단어 단위로 검색 → **0건**).
- 프로젝트 페이지(`paper/project_page.html`)의 코드 버튼은 "Code (Coming soon)"이며 `ZTX2021/LIBERO_openvla`를 가리킵니다. 이 저장소는 **404**입니다(2026-11-11 `gh api`).

**한 줄 결론:** 공개 저장소에는 **논문의 방법 구현이 없습니다.** 들어 있는 것은 ManiSkill 상류의 확산 정책 기준선(DP)과 다른 기준선 예제뿐입니다. 논문 ↔ 코드 대조는 "구현 없음"이 대부분이며, 기준선 설정만 대조할 수 있습니다.

## 1. 저장소 구성 (실제로 들어 있는 것)

| 경로 | 내용 | 논문과의 관계 |
|---|---|---|
| `mani_skill/` | 시뮬레이터 본체(상류 3.0.0b19) | 환경 기반. 논문 과제 중 PokeCube, PlaceSphere, PullCubeTool, PegInsertionSide 등록 확인 |
| `examples/baselines/diffusion_policy/` | 상류 DP 기준선 (`train.py` 상태 기반, `train_rgbd.py` 영상 기반) | 논문 기준선 DP와 설정 대조 가능 (D5–D9) |
| `examples/baselines/{bc, act, ppo, sac, tdmpc2, rlpd, rfcl, stable_baselines3}` | 상류 기준선 | 논문과 무관 |
| `docs/`, `tests/` | 문서와 테스트 | 무관 |

## 2. 논문 항목별 대조 (D-items)

범례: **구현 없음** = 공개 코드에서 찾지 못함, **설정 일치** = 기준선 설정이 논문 서술과 맞음, **불일치** = 서술과 다름.

| 항목 | 논문 (원문 위치) | 공개 코드 | 판정 |
|---|---|---|---|
| **D1 볼륨 구성** | 큐브 범위 r, 40³ 복셀, 채널 32, 복셀 중심 식 1, 카메라 투영 식 2 (§III-A) | 해당 코드 없음 (검색 0건) | **구현 없음** |
| **D2 볼륨–이미지 교차 주의** | deformable attention, 식 3 (§III-A) | 없음. `DeformAttn`류 검색 0건 | **구현 없음** |
| **D3 공간 토큰 생성** | goal-aware 볼륨(텍스트 토큰 결합), TokenLearner식 softmax 가중치, N = 200 (§III-B) | 없음 | **구현 없음** |
| **D4 보조 감독** | 말단·그리퍼 개폐 BCE 마스크, proprio에서 얻는 신호 (§III-B) | 없음 | **구현 없음** |
| **D5 다중 토큰 디코더** | adaLN-Zero DiT 블록, 202 토큰 (§III-C) | 없음. 상류 DP는 1D UNet(`conditional_unet1d.py`, 약 4.5M 파라미터)과 PlainConv 인코더 | **구현 없음** (DiT 블록 없음) |
| **D6 디퓨전 목표와 샘플링** | DDPM, ε 예측 MSE, 샘플링 100 단계 (식 4, §IV-A) | `train_rgbd.py`: `DDPMScheduler(num_train_timesteps=100, prediction_type="epsilon")`, `num_diffusion_iters = 100` | **설정 일치** (기준선 DP 기준) |
| **D7 입력 프레임 수** | 두 프레임 (t, t−1) (§IV-A) | `obs_horizon = 2` (기본값) | **설정 일치** (기준선 DP 기준) |
| **D8 기준선 DP의 관측 모드** | **RGB만 사용** (§I, §III-A, 저자 주장) | `train_rgbd.py`의 기본 `obs_mode = "rgb+depth"` (88행). 깊이가 관측에 있으면 깊이 채널이 `PlainConv` 입력 채널에 더해짐 (264–273행) | **불일치 가능성.** 논문이 쓴 설정이 `rgb`인지 확인되지 않음. 기본값대로 돌렸다면 기준선이 깊이를 사용 |
| **D9 학습 설정 (기준선 DP)** | "uniformly trained" (§IV-A). 시연 수는 과제당 약 150(ManiSkill) | `total_iters` 기본 1,000,000, `batch_size` 256, `lr` 1e-4, `act_horizon` 8, `pred_horizon` 16, `unet_dims` [64,128,256]. 예시 스크립트(`baselines.sh`)는 `num_demos` 100, `total_iters` 30,000–100,000 | **논문에 값 없음.** 저장소 예시와 논문 실험 설정의 일치 여부는 확인 불가 |
| **D10 과제 목록** | Poke Cube, Place Sphere, Hook Cube, Stack Pyramid, Insert Peg (본문) / Fig. 3은 Pull Cube Tool 표시 | PokeCube-v1, PlaceSphere-v1, PullCubeTool-v1, PegInsertionSide-v1 **등록됨**. HookCube, StackPyramid **등록 없음** | **불일치.** 본문 과제명 두 개가 공개 환경에 없음. Fig. 3과 본문도 어긋남 |
| **D11 평가 에피소드** | 시뮬 평가 횟수는 원문에 없음. 실물은 과제당 20회 | `num_eval_episodes` 기본 100, `num_eval_envs` 10 | **원문과 비교 불가** (시뮬 횟수 미보고) |
| **D12 시드** | 시뮬 3 시드 (§IV-A) | 시드 인자 `seed` 기본값 1 (43행). 논문의 3 시드 실행 목록은 공개되지 않음 | 설정 존재 확인, 실행 기록은 없음 |
| **D13 LIBERO 학습·평가** | 4 suite, 과제당 50 시연, no-op 제거 (§IV-A) | 해당 코드 없음. LIBERO 저장소 참조 0건 | **구현 없음** |
| **D14 LIBERO-Plus 교란 평가** | 7종 교란 (§IV-A) | 없음 | **구현 없음** |
| **D15 실물 로봇 코드** | Galaxea R1 Lite, isomorphic teleoperation (§V) | 없음 | **구현 없음** (하드웨어 필요) |
| **D16 카메라 보정** | 식 2의 외부·내부 행렬 필요 (§III-A) | 볼륨 관련 코드 없음 | **구현 없음** |

## 3. 기준선 설정에서 드러난 공정성 문제 (D8, D9)

- **관측 모드 불일치 가능성(D8).** 논문은 "RGB만 사용"이 핵심 차별점이라고 주장합니다(§I). 그런데 공개된 상류 RGB-D 기준선의 기본값은 `rgb+depth`입니다. 원문 DP의 ManiSkill 결과(Table II)가 이 기본값으로 돌렸다면, 기준선은 깊이를 쓰고 제안 방법은 쓰지 않는 셈입니다. 그러면 비교는 "깊이 있는 DP 대 RGB 전용 VolumeDP"가 되어 방향이 바뀔 수 있습니다. 원문으로는 확인할 수 없습니다.
- **학습 설정 미공개(D9).** 논문은 학습 epoch나 배치를 보고하지 않습니다. 상류 기본값(total_iters 1M, batch 256)과 예시(30k–100k)가 서로 크게 다르므로, 기준선 설정에 따라 결과가 크게 달라질 수 있습니다.
- **LIBERO 행(D13).** 코드가 없으므로 LIBERO 기준선 설정을 대조할 수 없습니다. `02` §2의 출처 문제와 합쳐 보면, LIBERO의 +16.4%p는 검증 불가입니다.

## 4. 구현 가능성 판정 (이 맥 기준, 정적 판단)

**판정: 논문 방법은 공개 코드로 재현할 수 없습니다(구현 없음). 공식 재현 경로는 세 가지 중 하나도 현재 열려 있지 않습니다.**

| 재현 경로 | 공개 여부 | 이 맥(Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음, torch 미설치)에서 | 필요한 자원 | 노력 |
|---|---|---|---|---|
| **공식 ManiSkill 재현** | 기준선 DP만 공개, VolumeDP 방법 없음 | **불가능.** ManiSkill README는 macOS를 지원하지 않음(GPU 시뮬 Linux 전용). CPU 시뮬도 macOS에서 불가로 명시 | Linux + NVIDIA GPU(GPU 시뮬), Vulkan 렌더링 | 방법을 직접 구현해야 함(큼) |
| **공식 LIBERO 재현** | 코드 없음(404) | **불가능.** 코드 없음. LIBERO 자체는 MuJoCo 기반이라 헤드리스 렌더링 확인 필요 | 학습 GPU, 데이터 | 구현 필요, 기준선 재학습 필요 |
| **실물 재현 (Galaxea R1 Lite)** | 코드 없음 | **불가능.** 하드웨어 없음 | 로봇, 텔레오퍼레이션 장비 | 불가 |
| **아이디어 재구현 (볼륨 + 토큰 + DiT)** | 해당 없음 | **중간~높은 난이도.** 모델 구현 자체는 PyTorch로 가능하나 MPS에서 40³×32 볼륨 학습은 메모리·시간 면에서 제한적(추정, 미확인). torch 설치 승인 필요 | 기존 DP 기준선 코드 + 구현 | 큼 |

- **노력 추정 (구현 가능성):** 공개된 것은 DP 기준선뿐이므로 방법 전체를 새로 짜야 합니다. 볼륨 리프팅(식 1–3)과 토큰 생성(§III-B), DiT 디코더(§III-C) 세 모듈입니다. 보조 감독(§III-B)과 카메라 투영 정확성을 검증하려면 대조 실험도 필요합니다.
- **하드웨어:** 학습에는 GPU가 필요합니다. 이 맥에서는 토이 수준 검증까지만 CPU로 가능합니다(`03` §6).

## 5. 재현 전 확인할 것 (저자 또는 공개 자료로)

1. 기준선 DP가 RGB로 학습되었는지(D8).
2. LIBERO 코드와 기준선 설정(공개 여부).
3. 실물 시험의 초기 배치 무작위화 방식.
4. 볼륨 생성에 쓴 카메라 보정 값과 과제별 r의 값(부록 없음).

## 6. 파일

- 저장소 클론: `repo/volumedp_maniskill/` (`.gitignore`에 포함, 재배포 안 함)
- 상세 검색 결과는 이 문서의 §2 판정에 요약했습니다. 재실행이 필요하면 `grep -rnwi`로 같은 용어를 다시 찾을 수 있습니다.

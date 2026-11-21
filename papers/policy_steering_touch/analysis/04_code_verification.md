# 04. 공식 코드 대조 (정적 읽기): 논문 ↔ 코드 차이 D1–D17

**대상.** `github.com/yilin-wu98/vital`, 커밋 `147bdb9` (2026-10-27), 얕은 복제 `repo/` (gitignore). **실행, 설치, import, 테스트 실행은 하지 않았습니다.** 읽은 파일은 `latent-tactile/dino_wm/`(학습·평가·데이터·모델), `anytouch-reward/anytouch_reward/`(보상), `liquid-tactile-control/liquid_tactile_control/`(run.py 프리셋, drivers/two_step.py), 설정 파일, README와 DATASETS.md, VALIDATION.md, DATASET_RELEASE.json, `latent-tactile/docs/recovered_runs.json`입니다. 복제물은 신뢰하지 않는 데이터로 다뤘습니다.

**패키지 구성.** 공식 저장소는 `latent-tactile`(세계 모델), `anytouch-reward`(촉각 보상), `liquid-tactile-control`(로봇 드라이버)의 세 패키지입니다. 논문의 코드 이름은 "ViTaL"이고, 패키지 안의 이름은 "liquid"입니다. 공식 릴리스는 **피펫(액체 옮기기) 과제만** 드라이버와 세계 모델 설정을 제공합니다(README).

## 1. 라이선스와 공개 범위

| 구성 요소 | 라이선스 (파일 기준) | 비고 |
|---|---|---|
| `latent-tactile` | MIT (`latent-tactile/LICENSE`) | |
| `anytouch-reward` | MIT (`anytouch-reward/LICENSE`) | 번들된 CLIP 토크나이저 등은 `THIRD_PARTY_NOTICES.md` |
| `liquid-tactile-control` | Apache-2.0 (`liquid-tactile-control/LICENSE`) | 드라이버는 Deoxys 로컬 작업 트리에서 가져왔다고 적힘 (provenance) |
| 번들 정책 백본 | MIT (`.../policy_backend/LICENSE.md`) | |
| 데이터셋 (HF) | CC BY 4.0 (`DATASET_RELEASE.json`, README) | 받지 않음 |
| 사전학습 가중치 | 각 원 프로젝트 조건 (README: "retain those projects' terms") | DINOv3, AnyTouch2, CLIP, ROBOMETER |

- README 배지는 "MIT / Apache-2.0"입니다. GitHub API는 `NOASSERTION`을 줍니다. 구성 요소별 라이선스가 맞으므로 **저장소 전체를 한 라이선스로 취급하면 안 됩니다**.
- 논문 라이선스(CC0)와 공식 코드 라이선스는 다릅니다. 이 폴더의 공개 자료에는 공식 코드를 넣지 않았습니다(코드 자체는 복제만 했고 게시 대상이 아님).

## 2. 논문과 코드의 차이 (D1–D17)

판정 기호: **불일치** = 논문 값과 코드 값이 다름, **미구현** = 논문에 있는데 코드에서 찾지 못함, **미공개** = 릴리스에 없음, **일치** = 확인함.

| 번호 | 항목 | 논문 | 코드 (파일:위치) | 판정 |
|---|---|---|---|---|
| D1 | 세계 모델 홀드아웃 | 본문 B.2: "holds out 30 trajectories". 부록 A.1: 검증 30 궤적 | `configs/liquid.yaml` `num_test: 50`, DATASETS.md "first 50 trajectories held out per task". Table 6 "Test split trajectories 50"과 같음 | **불일치** (논문 본문 내부 불일치) |
| D2 | 행동 마스크 확률 상승 | B.2: "linearly-ramped masking probability (reaching 0.5 at the final epoch)" | `train_dino_wm_tactile_k_step.py:101` `chance_of_mask = min(0.5, epoch_idx / total_epochs)`. 0.5에 **학습 중간**(약 절반 에폭)에 도달하고 그 뒤 고정 | **불일치** |
| D3 | 디코더 학습 반복과 설정 | Table 7: 5,000 반복, λ_LPIPS = 1.0 | 기록된 실행 `d5gwxcla`: `iters: 15000`, `batch_size: 64`, `num_test: 20`, `loss: rae`. `lpips_weight` 1.0 (RAE 경로에서 사용). `perceptual_weight` 0.1은 RAE에서 쓰이지 않음 | **불일치** (반복 수). λ_LPIPS, 판별기 워밍업 600/800은 일치 |
| D4 | 행동 차원 | A.1: 8차원 (위치 3, 쿼터니언 4, 그리퍼 1). 표 2와 B.2: 7차원 (6 DoF + 그리퍼) | 세계 모델 `frameskip * 7` (`dino_models_k_step.py`), 상태 8차원, 행동 블록 8 스텝 (`latent-tactile/README.md`) | **불일치** (A.1만 다름) |
| D5 | 촉각 보상 형태 | Sec. 4.3, 식 (8)–(10): `R^τ = cos(ẑ^τ, E(ℓ^τ))`. 양성 문장 하나 | `liquid-tactile-control/.../two_step.py` `make_tactile_reward_fn`: `reward = cos(pos) − cos(neg)` (`text_embeds_neg`가 있을 때). 두 단계 프리셋 `mixed_reward: True`는 음성 문장을 씀 (1단계 양성 "grasp lightly", 음성 "grasp heavily"; 2단계는 반대). `anytouch_reward/reward.py`의 `score_patch_embeddings`도 같은 음성 항을 가짐 | **불일치** (주 방법이 논문 식과 다른 보상을 씀) |
| D6 | 1→2단계 시각 전환 임계값 | Table 8과 A.3: 0.70 | `run.py` 두 단계 프리셋에 `phase_threshold` 없음. 드라이버 기본값 `two_step.py:153` 0.64. 다른 프리셋은 0.70 (vision-sampling), 0.63 (combined), 0.50 (tactile-guidance) | **불일치** (주 방법에 0.64가 적용됨. 정확한 실행 값은 로봇 설정 파일에 따라 달라짐) |
| D7 | 촉각 목표 40 스텝 유지 | A.3: 시각 보상이 0.7을 넘으면 촉각 목표를 바꾸고 40 스텝 유지 | `two_step.py` 약 1063–1100행: 전환 즉시 `text_embeds_active = phase2`. 40 스텝 유지 로직 없음. 대신 `t_count > 4` 가드가 있음 (논문에 없음) | **미구현** (40 스텝), **추가** (4 스텝 가드) |
| D8 | 2단계 시각 목표 문장 | A.3: "put the dropper back to the red cup" | `run.py` `VISUAL_TEXT`: "move the dropper back to the red cup" | 표현만 다름 (사소) |
| D9 | 시각 보상 카메라 | Table 8: front-view | `two_step.py` `--vision_reward_camera` 기본 `camera_0`. camera_0이 정면인지 문서에 없음 | **확인 불가** |
| D10 | 기준선별 목표 컵 | 논문은 목표 컵이 노란 컵 또는 파란 컵이라고만 씀 (A.3). 기준선 배정은 설명 없음 | `run.py`: 주 방법은 노란 컵, 단순 결합 기준선(`combined-sampling`)은 **파란 컵**. README가 이를 명시하고 "intentionally retain differences" | **불일치 가능** (기준선 비교의 교란. 방법 간 과제 조건이 다름) |
| D11 | 기준선 하이퍼파라미터 | Table 9은 주 방법만 보고. 기준선 설정은 없음 | 촉각 가이던스: 임계 0.50, `guidance_start_step 12`. 샘플링류: `action_select argmax` | **보고 안 됨** (재현에 필요) |
| D12 | 주 방법의 확산 편집 설정 | λ = 10, K = 4, ss = 1, 단위 노름 기울기, LPB 스케일, N = 10, h = 8, H = 16 (Table 8–9) | `run.py` two-step 프리셋: `guidance_scale 10`, `normalize_grad True`, `lpb_scale True`, `ss 1`, `n_samples 10`. `renoise_steps` 기본 4. `ACT_HORIZON = 8`, 두 덩어리로 H = 16 | **일치** |
| D13 | 세계 모델 구조와 학습 | 깊이 6, 헤드 16, 헤드 차원 64, MLP 2048, 드롭아웃 0.1, 공통 차원 384, 롤아웃 k ∈ {3,4,5}, Geometric p = 0.05, DDIM 100, 배치 16, 100k 반복, 학습률 5e-5/5e-4 (Table 5–6) | `dino_models_k_step.py` 기본값과 `liquid.yaml`, `train_dino_wm_tactile_k_step.py`의 옵티마이저 그룹이 모두 같음. AMP, `torch.compile` 사용 | **일치** |
| D14 | 기준선 코드 (SysID, PhysPred 등) | 해당 없음 (이 논문은 기준선을 자체 구현으로 비교) | 드라이버: 시각 샘플링, 촉각 샘플링, 촉각 가이던스, 단순 결합, 기본 정책 (`run.py` METHODS). 기본 정책 단독 실행 포함 | **일치** (기준선 드라이버는 있음) |
| D15 | 기본 정책 가중치와 학습 | B.1: 확산 U-Net, 80k 반복 | 릴리스에 `itr_23999.ckpt` 등이 없음. README: "base-policy training is outside this repository" | **미공개** |
| D16 | 과제 범위 | 세 과제 (닦기, 구멍 끼우기, 피펫) | README: "the documented active world-model recipe and bundled robot drivers are for pipette/liquid transfer. Insertion and wiping require their own trained checkpoints and decoders." | **부분 공개** (피펫만 드라이버와 레시피 있음) |
| D17 | 로봇과 보상 서버 | Franka, Robotiq, GelSight, ZED, RealSense, ROBOMETER-4B | README: Deoxys, deoxys_vision, 카메라 스트림, 호환되는 Robometer 서버가 필요하다고 적음. 서버는 포함되지 않음 (`robometer_client.py`는 HTTP 클라이언트만) | **미공개** (재현 불가 핵심) |

**추가로 확인한 점 (D 항목 밖).**

- (추가 확인, D 목록 밖) 세계 모델 학습 코드는 CUDA를 전제합니다. `torch.autocast(device_type="cuda")`, `torch.cuda.amp.GradScaler`, `device: cuda:0`, `torch.compile(mode="reduce-overhead")`(CUDA 그래프)입니다. **이 맥(MPS)에서 바로 돌릴 수 없고 수정이 필요합니다.**
- 기록된 학습 설정에 `gripper_change_weight: 5`가 있지만, 릴리스 레시피는 `segment_sampling: uniform`이라 가중치 샘플링을 쓰지 않습니다(`tactile_dataset.py`의 가중치는 가중 샘플링에서만 쓰임). 논문에는 언급이 없습니다.
- 촉각 토큰은 AnyTouch 2 패치를 `mean_patch`로 평균하고 L2 정규화합니다(`reward.py`). 논문은 풀링 방식을 적지 않습니다. 세계 모델은 예측 패치 196개 중 앞 196개를 씁니다(DATASETS.md). 저장된 임베딩은 392개입니다.
- `recovered_runs.json`은 학습 실행 두 개를 복구한 기록입니다. 세계 모델 실행 `xrjdyidg`(2026-05-03)와 디코더 `d5gwxcla`(2026-05-02)입니다. 두 실행 모두 같은 git 커밋 `c6cd627`에서 나왔습니다.

## 3. 주장과 코드의 일치 (확인한 항목)

- 세계 모델 구조와 학습 설정(D13)은 논문과 같습니다. 이 부분은 구현 자체는 읽기 쉽습니다.
- 확산 편집의 핵심 수식(λ = 10, K = 4, 단위 노름 기울기, LPB 스케일, D12)은 프리셋에서 같습니다.
- 촉각 보상의 임베딩 경로(예측 패치 → 평균 → 정규화 → 텍스트 임베딩과 내적)는 논문 식과 같습니다. 차이는 음성 항(D5)뿐입니다.
- 주 방법의 2단계 구조(H = 16 = 두 덩어리 × 8)는 논문과 같습니다.

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 이 맥에서 논문 결과를 재현할 수 없습니다. 알고리즘 구조는 소형 환경에서 옮길 수 있습니다.**

| 무엇 | 필요 자원 | 이 맥 | 근거 |
|---|---|---|---|
| 실로봇 성공률 (Fig. 2, Table 1 실로봇 부분, Fig. 8) | Franka 7-DoF, Robotiq 2F-85, GelSight Mini, ZED, RealSense, Deoxys, Robometer 서버, 기본 정책 가중치 | **불가** | D15, D16, D17 |
| 세계 모델 학습 (100k 반복, 세 카메라) | CUDA GPU, pipette 데이터 361 GB (DATASET_RELEASE.json), 임베딩 캐시 | **불가에 가까움** (CUDA 전제, 저장 공간, 학습 시간 미측정) | 위 "추가 확인", DATASET_RELEASE.json |
| 세계 모델 추론 (공개 체크포인트가 있다면) | 체크포인트 (릴리스에 위치 없음), CUDA 경로 | **조건부** (체크포인트 공개 여부 미확인, 포팅 필요) | D17, README |
| 보상 검증 (텍스트-촉각 코사인, AnyTouch 2) | AnyTouch 2 가중치 (외부), CLIP 텍스트 인코더 | **조건부** (가중치 다운로드는 사용자 승인 필요. 추론은 CPU 가능, 속도 미측정) | `score_liquid.sh` |
| 시각 보상 (ROBOMETER-4B, VLM) | 4B VLM 서버 | **불가에 가까움** (서버 미공개, 4B 모델을 로컬에서 돌리는 것은 측정 전) | D17 |
| 알고리즘 구조 (이중 최적화, 촉각 확산 편집, 음성 항 유무 비교) | 소형 시뮬 (예: numpy 토이, MuJoCo), 작은 확산 정책 | **가능** (CPU/MPS, 새 설치 승인 필요) | 방법 구조는 Sec. 4, 부록 B.3–B.4에 있음 |
| 공식 코드의 정적 검증 | 없음 | **완료** | 이 문서 |

**노력 추정 (측정 전, 추정).**

- 공식 세계 모델 학습·추론을 MPS로 옮기기: 약 1–2주. 자동 혼합 정밀도, `torch.compile` 경로, 임베딩 입력 형식을 고쳐야 합니다.
- 피펫 데이터 일부 (40 에피소드, 약 1.1 GB/에피소드로 계산해 약 45 GB 다운로드): 수 시간에서 하루.
- 소형 토이로 이중 최적화와 촉각 보상 구조를 옮기기: 1–2일 (추정).
- 논문 과제 재현 (로봇, 시각 보상 서버): **불가** (로봇이 없음).

## 5. 문제점과 확인하지 못한 점

1. **공식 코드는 논문과 다른 설정을 씁니다 (D1–D7).** 보고된 수치(Fig. 2, Table 1)가 어떤 설정에서 나왔는지 확인할 수 없습니다. 특히 주 방법의 임계값(D6)과 음성 항(D5)은 결과에 직접 영향을 줍니다.
2. **기준선 설정이 주 방법과 다릅니다 (D10, D11).** 목표 컵과 임계값이 다르면 "단순 결합 대비 우위"의 근거가 약해집니다.
3. **릴리스에 없는 것.** 기본 정책 가중치와 학습 코드, 로봇 드라이버 대부분, 로봇 서버, 세계 모델 체크포인트 위치, 실행 기록입니다. 릴리스가 "모든 과제를 재현할 수 있다"는 뜻은 아닙니다(README가 명시).
4. **확인하지 못한 것.** 공개 릴리스의 테스트 29개(VALIDATION.md)는 실행하지 않았습니다. 세계 모델 체크포인트 크기는 확인하지 못했습니다. HF 데이터셋 페이지를 직접 열어 보지 않았습니다(README 기준).
5. **라이선스 주의.** 저장소 전체를 MIT로 적으면 Apache-2.0 구성 요소와 맞지 않습니다. 코드를 다른 곳에 가져갈 때는 구성 요소별로 확인해야 합니다.

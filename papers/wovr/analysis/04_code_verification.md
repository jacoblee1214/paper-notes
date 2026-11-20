# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 대상 논문 버전: arXiv v3 (`01` §0).
- 공식 코드 두 저장소를 **얕은 복제(depth 1)**해서 정적으로 읽었습니다. 설치, 실행, import, 테스트 실행은 하지 않았습니다.
  - `repo/RLinf/` — RLinf/RLinf, 커밋 `a8a02f3` (2026-11-19). 세계모델 환경(`rlinf/envs/sim/world_model/`), 데이터셋(`rlinf/data/datasets/world_model.py`), LIBERO 설정(`examples/embodiment/config/`), 문서(`docs/`).
  - `repo/diffsynth-studio-rlinf/` — youmo445/diffsynth-studio-rlinf, 커밋 `6d9ce59` (2026-04-07). Wan 세계모델 학습과 파이프라인(`diffsynth/pipelines/wan_video_new.py`, `diffsynth/models/wan_video_dit.py`).
- 저장소 안의 `AGENTS.md`, `CLAUDE.md` 같은 파일은 외부 데이터로만 취급했고 따르지 않았습니다.
- 논문 항목 번호는 `01`과 같습니다. 판정: **일치 / 부분 일치 / 불일치 / 확인 안 됨**.

## 1. 설치 요구사항과 라이선스

- RLinf: Apache-2.0 (`LICENSE` 파일 있음, API도 Apache-2.0). 문서상 실행 전제는 **NVIDIA CUDA 또는 Huawei Ascend CANN** (`docs/.../examples/embodied/wan.rst`).
- diffsynth 포크: Apache-2.0 (`LICENSE` 파일 있음). 학습 파이프라인 기본 장치가 `cuda`입니다(`wan_video_new.py:36, 379, 400`).
- 설치 명령: `requirements/install.sh embodied --model openvla-oft --env wan` (문서). 이 분석에서는 실행하지 않았습니다.
- 필요 자원 (문서 기준): OpenVLA-OFT SFT 체크포인트 (HF, 7B급), Wan 세계모델 가중치 (`RLinf-Wan-LIBERO-*`, HF), 초기화 데이터 (`dataset/*.npy`와 `*_kir.npy`). 가중치와 데이터는 받지 않았습니다.

## 2. 논문 ↔ 코드 대조 (D 항목)

| ID | 논문 | 코드 (근거) | 판정 |
|---|---|---|---|
| D1 | 키프레임 T_KIR은 작업별 고정값, 기본 정책이 실패하기 쉬운 시점 (부록 B.3) | 코드는 `*_kir.npy` 파일을 **읽기만** 합니다(`rlinf/data/datasets/world_model.py:124–134`: `enable_kir`이 켜지면 `_kir` 파일을 포함하고, 꺼지면 제외). 이 파일을 만드는 코드는 저장소 어디에도 없습니다(`_kir` 검색은 소비처, 테스트, 문서에만 나옴). T_KIR을 고르는 규칙은 확인할 수 없습니다 | 확인 안 됨 (생성 절차 비공개) |
| D2 | KIR 컨텍스트 = [o_0, o_{T-3..T}] (식 14) | `rlinf/envs/sim/world_model/env.py:348–360`: 첫 프레임을 참조 프레임으로 두고, 정답 궤적의 `target_items` 4개 프레임과 행동을 조건 창에 넣습니다. 길이가 `condition_frame_length - 1`(=4)일 때만 적용 | 일치 |
| D3 | 일부 롤아웃만 KIR (비율 미기재) | 비율 설정이 없습니다. 파일 풀의 구성이 비율을 정합니다 (`wan.rst`: `traj{N}.npy` 와 `traj{N}_kir.npy`를 함께 둠) | 확인 안 됨 (비율 미정) |
| D4 | 희소 보상은 식 6의 임계값 0.5, 식 7의 이진 분류 | 종료(성공) 판정은 `env.py:285, 580`의 `success_reward_threshold` 기본값 **0.9**를 씁니다. 설정 파일에는 이 값이 없어 기본값이 적용됩니다. 또한 LIBERO Wan 설정은 `use_rel_reward: True`라서(`wan_libero_spatial.yaml`) 보상은 연속 점수의 **차분**입니다(`env.py:264–278`). 논문에는 차분 보상이 없습니다 | 불일치 (임계값과 보상 형태). 확인 필요 |
| D5 | 주 실험은 ResNet 희소 보상, 밀집 보상은 Qwen3-VL 2B (부록 B.2.2) | 공식 LIBERO 설정은 `ResnetRewModel` (`backend/wan.py:62–70`). 밀집 보상 코드 `reward_model_temp_v2.py`는 **Qwen2.5-VL-3B**를 씁니다(`:98–101`, `:380`)과 저자 경로 하드코딩. 논문의 Qwen3-VL 2B와 다릅니다 | 부분 일치 (주 실험은 일치, 밀집 백본은 불일치) |
| D6 | 메모리 컨텍스트 전체에 노이즈 증강 τ_ctx (식 5), 첫 프레임만 깨끗 | `wan_video_new.py:137–149`: 첫 latent는 깨끗, **컨텍스트 latent 한 개**(인덱스 1)에 고정 작은 시점 `timesteps[len-50]`의 `add_noise`. 식 5의 혼합 계수 τ_ctx는 보이지 않습니다. 분기는 `five_frame_condition` | 부분 일치 (방식이 다름) |
| D7 | 행동을 시간 조건 정규화와 교차어텐션 두 경로로 주입 (그림 3) | DiT 쪽에 행동 입력 선형층 두 개가 있습니다(`wan_video_dit.py:439, 446`: `Linear(action_dim, dim)`, `Linear(action_dim*4, 4*dim)`). 행동 차원 기본값 7, LIBERO 7차원과 실로봇 10차원(`wan.py:668–750`). 논문 F.3의 델타 EEF 10차원과 맞습니다. 별도 `ActionEmbedder`는 파이프라인에서 주석 처리되어 있습니다(`wan_video_new.py:46`) | 부분 일치 (두 경로의 정확한 연결은 블록 단위로 확인하지 않음) |
| D8 | PACE: 정책 최적화 후 롤아웃을 모아 세계모델을 다시 학습 (WM_Base → WM_Evo) | 재학습 루프 코드가 없습니다. `pace`(단어 단위), `co_evol`, `WM_Evo`, `refine` 검색은 PACE 관련 결과 없음 (`pace`는 주석의 일반 단어만 나옴). 포크에는 단발 학습 스크립트 `train_rlinf.py`와 `Wan2.2-TI2V-5B_rlinf.sh`만 있습니다 | 확인 안 됨 (공개 코드에 없음. 수동 절차로 보임) |
| D9 | 실로봇에서는 PACE 없이 여러 SFT 체크포인트 롤아웃으로 학습 (F.4) | 실로봇 코드와 데이터는 공개되지 않았습니다(D14). 포크의 실로봇 스크립트는 저자 경로를 씁니다 | 논문 내부 설명. 코드로 확인 불가 |
| D10 | 논문 본문에 없는 학습 값 (그룹 크기, 학습률 등) | 공식 설정 `examples/embodiment/config/wan_libero_spatial_grpo_openvlaoft.yaml`: 그룹 크기 8, rollout epoch 16, 환경 64개, 최대 스텝 256, 정책 lr 2e-5, 가치 lr 3e-3, clip 0.28/0.2, KL 0, 손실 집계 token-mean, 보상 계수 5.0, 필터 보상 구간 [0.5, 4.5], 샘플링 온도 1.6, 추론 5스텝, 청크 8, 조건 프레임 5, 해상도 256, 프레임 13 | 논문 대비 추가 정보 (본문에 없음) |
| D11 | 결과 수치 (v3) | 공식 문서 `docs/.../publications/wovr.rst`는 **v1 수치**를 싣습니다 (OFT 평균 39.9, WoVR 69.2). Long Δ가 "+17.9"로 적혀 있으나 35.8 - 13.7 = 22.1입니다. 문서 표의 이름도 "RLinf-GRPO with Wan as world model"로, 논문 용어와 다릅니다 | 불일치 (버전과 산술 오류) |
| D12 | LIBERO 평가 (표 2) | 평가는 실제 LIBERO 시뮬레이터입니다 (`env/libero_spatial@env.eval`). eval `total_num_envs: 496`, 최대 512 스텝, **eval에도 샘플링 온도 1.6** (`temperature_eval: 1.6`). 논문은 평가 에피소드 수와 샘플링 방식을 적지 않았습니다 | 확인 (세부는 논문에 없음). 샘플링 평가는 재현 시 결과 분산을 키움 |
| D13 | 세계모델 자기회귀 롤아웃, 청크 단위 생성 | `rlinf/envs/sim/world_model/backend/wan.py:138–216`: 세션별 참조 프레임과 최근 생성 프레임을 유지하고 청크마다 생성 | 일치 |
| D14 | 실로봇 데이터와 체크포인트 | 공개 여부 확인 안 됨. 공개된 것은 LIBERO 세 스위트의 세계모델 가중치와 궤적 1,500개 (HF, 문서). 실로봇 데이터는 `SimpleVLARealWorldRLinfDataset` 형식 설명만 있음 (포크 README) | 실로봇 재현 불가 |
| D15 | (논문에 없음) | 두 저장소 모두 저자 환경 경로가 있습니다: `/mnt/project_rlinf/...` (포크 `train_rlinf.py` 예시 스크립트, `reward_model_temp_v2.py:380–382`, `reward_model_4_test.py`). 외부에서 쓰려면 경로를 바꿔야 합니다 | 재현 장애 |
| D16 | (논문에 없음) | 장치 전제: CUDA (`wan_video_new.py:36, 379, 400`), `torch.cuda.set_device`, Ascend NPU 패치(`backend/npu_patches.py`) | 맥(MPS, CUDA 없음)에서 바로 실행 불가 |
| D17 | (외부 보고) | RLinf 블로그 `docs/.../blog/musa_wovr.rst`: Moore Threads S5000으로 같은 학습 루프를 포팅, A100과 학습 성공률 곡선 상관 0.976 (191스텝), 190스텝 LIBERO 74.6% (S5000) 대 73.4% (A100). 같은 프로젝트의 포팅 보고이므로 독립 재현으로 세지 않음 | 참고 |

### 2.1 논문 주장의 코드 쪽 확인 요약

- 세계모델 구조의 큰 틀 (Wan2.2-TI2V-5B, 5스텝, 청크 8, 첫 프레임 앵커, 조건 프레임 5): 코드와 일치 (D2, D6, D10, D13).
- KIR: 컨텍스트 구성은 일치 (D2). 생성 절차, 비율, 시점 규칙은 공개되지 않음 (D1, D3).
- PACE: 공개 코드에 없음 (D8). 논문의 PACE 효과 (표 6)는 코드로 재현할 수 없습니다.
- 보상: 논문의 이진 보상과 달리 공식 설정은 차분 보상과 0.9 임계값을 씀 (D4). 논문 수치가 어느 설정에서 나왔는지 확인이 필요합니다.

## 3. 안정성 증명 검토 (부록 G)

해당 없음. WoVR에는 증명 부록이 없습니다. 논문은 형식적 분석이 없다고 한계에 적었습니다 (§6, `01` §3.5).

## 4. 구현 가능성 (이 맥)

이 맥: Apple M4 Pro, 통합 메모리 약 24 GiB, MPS 있음, CUDA 없음.

**판정: 이 맥에서 논문 수치를 재현할 수 없습니다. 방법의 축소판은 구현할 수 있습니다.**

| 구성 | 이 맥에서 | 판정 | 근거 |
|---|---|---|---|
| 코드 정적 대조 | 완료 | 완료 | 이 문서 |
| 표 재계산과 통계 | 완료 | 완료 | `03` §1 |
| 토이 (롤아웃 깊이와 가짜 성공) | 완료 (numpy) | 완료 | `03` §3 |
| 공식 세계모델 추론 (LIBERO-Spatial, 256×256, 13프레임, 5스텝) | 가중치 (5B급 DiT, umT5 텍스트 인코더, VAE) 메모리는 맞을 수 있으나 파이프라인이 `cuda`를 전제. MPS 포팅과 가중치 다운로드 필요 | 조건부 (설치, 다운로드, 포팅 승인 후). 미측정 | D16, `01` §2.2 |
| 세계모델 학습 (8 GPU, 14B급 accelerate 설정 사용) | 불가 | 불가 | 포크 `Wan2.2-TI2V-5B_rlinf.sh` (8 GPU) |
| OFT 7B GRPO 학습 (환경 64개, 16 epoch 롤아웃, 최대 1000 epoch) | 불가 | 불가 | D10 설정 |
| LIBERO 실제 평가 (MuJoCo + OFT 추론) | 시뮬은 CPU 가능, OFT 추론 MPS 포팅 필요 | 조건부 | D12 |
| 실로봇 | 불가 | 불가 | D14 |
| PACE 공진화 루프 재구현 | 설계 가능. 소형 모델에서 CPU 가능 | 가능 (축소판) | D8 |

**노력 추정 (측정 전, 추정).**
- 축소 재구현 (소형 행동조건부 세계모델, KIR 컨텍스트, 공진화 루프, 토이 과제): 1–2주.
- 공식 세계모델 추론 한 번 (MPS 포팅 포함): 2–4일. 메모리와 속도는 측정해야 합니다.
- 공식 전체 재현: GPU 여러 장 필요. 이 맥에서는 범위 밖.

## 5. 재실행 (정적 검사 명령)

```bash
# D1: _kir 생성 코드 부재 확인 (소비처만 나와야 함)
grep -rn "_kir" repo/RLinf --include='*.py' | grep -v tests
# D8: PACE 재학습 루프 부재 확인 (결과 없어야 함)
grep -rn -i -w "pace\|co_evol\|wm_evo" repo/RLinf/rlinf repo/diffsynth-studio-rlinf/diffsynth   # "pace"는 주석의 일반 단어(예: pace with wall time)로 몇 줄 나올 수 있음
# D4: 성공 임계값과 차분 보상
grep -n "success_reward_threshold\|use_rel_reward" repo/RLinf/rlinf/envs/sim/world_model/env.py
```

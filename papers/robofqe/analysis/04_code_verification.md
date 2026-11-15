# 04. 논문 ↔ 공식 코드 정적 대조와 구현 가능성

공식 저장소는 `repo/`에 얕은 clone(`a40a3bd`, 2026-10-28 커밋 "Rewrite README for the official release, add MIT license")으로 받았습니다. **실행, import, 설치, 가중치 다운로드를 하지 않았습니다.** 읽은 파일은 `fqe/base.py`, `fqe/trainer.py`, `fqe/pretrained/algorithm.py`, `fqe/pretrained/networks.py`, `fqe/pretrained/encoders.py`, `fqe/scratch/*`, `fqe/data/libero.py`, `fqe/policy.py`, `fqe/rollouts.py`, `fqe/lrm.py`, `fqe/libero_patches.py`, `scripts/train_fqe_dinov3.py`, `scripts/train_fqe_scratch.py`, `scripts/collect_rollouts.py`, `scripts/finetune_lrm.py`, `scripts/label_rewards_lrm.py`, `README.md`, `environment.yml`입니다. 라이선스는 MIT입니다(`LICENSE`, 저장소 페이지도 MIT라고 표시). 가중치(DINOv3, LRM)는 각자 라이선스를 따릅니다.

## 1. 대조 범위

- **공개 범위:** README는 "This release covers the LIBERO experiments."라고 씁니다. CALVIN, 실로봇, 세계모델 기준선, 지표 계산, 표·그림 생성 스크립트는 저장소에 없습니다(§3 D10).
- **학습 데이터:** LIBERO 공식 데모(`*_demo.hdf5`)와 BC 체크포인트(`bc_ckpts/`)가 필요합니다. BC 체크포인트는 저장소에 없습니다. README는 학습된 BC 체크포인트를 가정합니다.

## 2. 대조표

| 논문 항목 | 코드 위치 | 판정 |
|---|---|---|
| 식 (1) 할인 반환 정의 | 정답 MC 평가 코드 없음 (README: `--gamma 1.0`이면 성공률) | 정의만 일치, 정답 계산은 미공개 |
| 식 (2)–(3) 궤적과 종료 라벨 r̂ (궤적 마지막 스텝) | `fqe/data/libero.py`(`rewards`를 마지막 스텝에만), `fqe/rollouts.py`(`dones[-1] = 1`) | 일치 |
| 식 (4) 제곱 손실 | `fqe/pretrained/algorithm.py` `update()` (`loss_type="mse"`) | 일치 |
| 식 (5) 목표: 비종료 γ E[Q̄(x', a')], 종료 r̂ | `algorithm.py`: `target = rewards + γ(1 − done) · Q̄(x', a')`, 한 번 샘플링 | **D1, D2** |
| 식 (6) 초기 상태 평균 V̂ | `algorithm.py` `estimate()`: 데이터의 t = 0 입력 평균, 행동 10개 샘플 평균 | 일치 (D9) |
| DINOv3 인코더 | `fqe/pretrained/encoders.py`(`dinov3_vits16` 기본, patch 16이라 128 px 그대로) | 일치 (D4) |
| 고유감각·그리퍼 입력을 임베딩 후 결합 (§3.4) | `networks.py` `lowdim_keys`, `scripts/train_fqe_*.py`: `("gripper_states", "joint_states")` | **D3** |
| 언어 인코더 (사전학습 텍스트 인코더) | `fqe/data/libero.py`: BERT 캐시로 `task_emb` | 일치 |
| LRM 종료 판정 (지도 미세조정 + 제로샷) | `fqe/lrm.py`(Qwen3-VL-8B 기반), `scripts/finetune_lrm.py`, `scripts/label_rewards_lrm.py` | 일치, 훈련 설계 **D7** |
| 롤아웃 25개 per task, 데모 50개 | `collect_rollouts.py` `--n-rollouts` 기본 25 | 일치 |
| LIBERO 최대 스텝 220/280/300/520 | `fqe/rollouts.py` `MAX_STEPS` | 일치 |
| 시드 (논문 "5 seeds", "three seeds") | 기본 `seed=42`, `--seed` 옵션 | 정의 불명 (D11) |
| 무작위 정책 (A.4) | `collect_rollouts.py` 정책 지정 `random` | 생성 가능 |
| ResNet-18 from scratch (A.1, 표 1) | `fqe/scratch/networks.py` (torchvision `resnet18(weights=None)`) | 일치 |
| 데모만 학습 (A.4 표 5) | `--demos-only` | 일치 |
| 풀링한 4개 LIBERO 데이터 (A.3) | 학습 스크립트는 `--suite` 하나만 받음 | **미지원**, 새 코드 필요 (D12) |
| 고유감각 정책 (A.1, 표 4) | 체크포인트 `cfg`에서 로드 가능. 체크포인트 미공개 | 코드 가능, 가중치 없음 |
| 지표 (ARE, MSE, r, MMRV, 부트스트랩 CI) | 없음 | **미공개** (D10) |
| 기준선 (WorldGym, Cosmos 2.5-AC, Ctrl-World) | 없음 (`rollouts.py`의 `world-model` 언급은 주석뿐) | **미공개** (D10) |
| CALVIN, 실로봇 SO-101 | 없음 | **미공개** (D10) |

## 3. 불일치와 빈칸

판정 기호: 불일치 = 논문과 코드가 다름, 빈칸 = 논문에 없음, 미공개 = 저장소에 없음.

**D1. 목표 네트워크 갱신 (불일치, 작음).** 논문 식 (5)는 "주기적으로 동기화된 복사본"이라고 씁니다. 코드는 `target_update_type="soft"`, `tau=0.005`로 매 스텝 Polyak 평균을 씁니다(`train_fqe_dinov3.py`, `train_fqe_scratch.py`). 주기 동기화는 `target_update_type="hard"`에 해당합니다. 논문 본문만으로는 어느 쪽인지 알 수 없습니다.

**D2. 목표 값 클리핑 [0, 1] (불일치, 작음).** `clip_target=True`, `clip_min=0`, `clip_max=1`이 기본입니다. 식 (4)–(5)에는 없습니다. 보상이 {0, 1}이면 이론상 Q ∈ [0, 1]이라 클리핑은 대개 작동하지 않지만, 초기 학습과 외삽 영역(A.4의 무작위 정책)에서는 값을 잘라냅니다. A.4의 무작위 정책 추정값이 [0, 1] 범위 안에 있는지, 클리핑이 그 값에 영향을 주었는지는 확인하지 못했습니다.

**D3. Q 입력에 고유 상태가 들어감 (불일치, 중요).** 두 학습 스크립트는 `lowdim_keys=("gripper_states", "joint_states")`를 기본으로 씁니다. 주된 실험의 정책은 `img128_noprop`, 즉 고유감각이 없는 정책입니다(README, `libero_patches.py`). 그래서 **RoboFQE의 Q 함수는 정책이 보지 않는 관절과 그리퍼 상태를 입력으로 받습니다.** 논문은 §3.2–3.3에서 Q의 입력을 "정책 입력 x_t = f(h_t)"로 정의하고, A.1에서는 "Q가 정책의 입력 모달리티를 상속한다"고 씁니다. 그런데 §3.4는 고유감각을 항상 임베딩한다고 쓰고, 주된 실험이 어떤 입력을 쓰는지는 밝히지 않습니다. 결과적으로 (a) 논문 설명과 코드가 어긋나고, (b) **표 1에서 RoboFQE만 정책보다 많은 정보를 씁니다.** 세계모델은 영상만 생성하므로 같은 정보를 쓸 수 없습니다. 비교가 공정한지는 이 점을 분리해야 판단할 수 있습니다(01 §5.4에서 다룸).

**D4. 하이퍼파라미터와 구조의 빈칸 (빈칸).** 논문 본문에는 DINOv3 크기, 프레임 스택, 미세조정 방식, 학습률이 없습니다. 코드 기본값은 다음과 같습니다.

| 항목 | 코드 기본 |
|---|---|
| DINOv3 크기 | ViT-S/16 (`dinov3_vits16`; 동결 모드에서 학습 파라미터는 약 1.5M이라고 스크립트 주석에 있음) |
| 미세조정 | `partial`: 상위 4블록 + 최종 LN, 백본 학습률 1e-5 (README는 "논문 설정"이라고 함) |
| 프레임 스택 | 10 (패딩은 첫 프레임 복제) |
| 헤드 학습률, 옵티마이저 | 1e-4, AdamW, weight decay 1e-5 |
| 기울기 클리핑 | 0.5 (partial) |
| 배치, 스텝 | 64, 50,000 |
| 시간적 트랜스포머, Q 헤드 | 2층, 4헤드, 토큰 256, 헤드 (256, 256) |
| 감마 | 0.99 |
| 이미지 | 128 x 128, 두 카메라, 인코더 공유 |

논문 본문에 이런 값이 없으므로 재현하려면 저장소 기본값을 써야 합니다. README가 "partial이 논문 설정"이라고 하지만, 논문 본문에서 그 근거를 찾을 수 없습니다.

**D5. 시간 인덱스가 없음 (빈칸, 이론적 한계).** Q 네트워크는 `Q(obs_seq, action, task_emb)`이며 시점 t나 남은 시간을 입력하지 않습니다(`networks.py` `forward`). 감마 할인 목표 γ^(T−1−t)는 시점에 따라 달라지므로, 같은 관측이 다른 남은 시간을 가지면 Q가 하나의 값만 낼 수 있습니다. 논문 §3.3은 Markov 조건을 말하지만 이 항은 다루지 않습니다.

**D6. 시간 제한 종료 (빈칸).** 실패 롤아웃은 최대 스텝에서 `dones=1`로 끝납니다(`rollouts.py`). 시간 제한을 종료로 보는 것은 식 (5)의 정의와 맞지만, 이 선택이 값을 어떻게 바꾸는지는 논의하지 않습니다.

**D7. LRM 미세조정 설계 (빈칸, 중요).** 논문 §4.1은 "LIBERO 시연 데이터로 미세조정했다"고만 씁니다. 코드(`finetune_lrm.py`)는 다음을 씁니다. 양성은 **시연의 마지막 5 프레임**, 음성은 **진행률 0.8 이하의 초기 프레임**이고, 다른 과제의 지시를 붙인 불일치 음성은 기본 0개입니다(LIBERO-Spatial은 검은 그릇이 접시 위에 있는 것으로 끝나 모호하다고 주석). 그러면 검출기는 "끝난 장면"과 "시작 장면"을 구분하도록 배우고, **"거의 성공했지만 실패한" 근접 실패 장면은 학습하지 않습니다.** 논문의 97.4%(LIBERO)는 롤아웃에서 측정된 값이라서 근접 실패에 대한 오류가 포함되어 있는지 알 수 없습니다. 클래스별 정확도도 보고되지 않습니다(03 §4).

**D8. 라벨 적용 범위 (일치, 확인).** 시연은 원래 정답 라벨을 쓰고, LRM 라벨은 증강 롤아웃에만 붙습니다(`label_rewards_lrm.py` 주석, `libero.py`의 "Demos always use 'rewards'"). 논문 §4.1과 맞습니다. 다만 시연 데이터는 모두 성공이므로(논문 §4.1, A.4), 실패 전이의 대부분은 LRM 라벨이 붙은 롤아웃에서 옵니다.

**D9. 초기 상태 (빈칸, 간접).** 기본 수집 설정은 `--init-states fixed`, 즉 LIBERO 고정 초기 상태를 에피소드 e mod 50으로 씁니다(`collect_rollouts.py`). 식 (6)의 초기 분포는 데이터의 t = 0 입력이므로, 이 설정에서는 **정답 MC 평가가 같은 고정 초기 상태를 쓴다면** 초기 분포 불일치 오류가 없어집니다. 논문은 정답 평가의 초기 상태를 밝히지 않았고, 정답 평가 코드도 없어서 이 가정은 확인하지 못했습니다.

**D10. 미공개 항목 (미공개).** 지표 계산(ARE, MSE, Pearson r, MMRV, 부트스트랩 CI), 세 가지 세계모델 기준선(WorldGym, Cosmos 2.5-AC, Ctrl-World), CALVIN(X-VLA), 실로봇(SO-101, LRM 제로샷, 세션 평가), 정답 MC, 표·그림 생성 스크립트가 없습니다. 따라서 **표 1–3의 숫자는 저장소로 재생성할 수 없습니다.** BC 체크포인트(`multitask_model_ep5/15/20/50`)도 없습니다.

**D11. 시드 정의 (빈칸).** 기본 시드는 42입니다. 논문의 "5 seeds"(LIBERO 절대값)와 "three seeds"(실로봇)가 학습 시드인지, 롤아웃 시드인지, 정책 체크포인트인지 본문에 없습니다.

**D12. 네 과제 풀링 (미지원).** A.3의 "all-LIBERO" 조건(3000 궤적, 569k 전이)은 학습 스크립트가 `--suite` 하나만 받으므로 저장소에서 바로 실행할 수 없습니다. 이 조건의 메모리(55.9 GB uint8, 03 §1.5)도 이 맥에서는 불가능합니다.

**D13. 데이터 로더의 메모리 (정보).** `fqe/data/libero.py`는 모든 궤적의 이미지를 `np.uint8`로 RAM에 올립니다. Spatial 10.0 GB, Object 11.9, Goal 11.2, Long 22.8 GB입니다(03 §1.5). 공식 파이프라인을 24 GiB 맥에서 돌리려면 로더를 고쳐야 합니다.

**D14. 평가 중 학습 데이터 사용 (정보).** `estimate()`는 학습에 쓴 궤적의 t = 0 입력으로 V̂를 계산합니다. 정답과 대조하는 용도이므로 누설은 아니지만, 추정이 학습 데이터에 맞춰져 있다는 뜻입니다. 학습 중 평가 간격은 500 스텝이고, 최종 값은 마지막 체크포인트입니다(선택 없음).

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 논문 수치의 재현은 이 맥에서 불가능합니다. 정적 재계산과 토이는 이미 끝냈고, 축소판(§4.3)만 조건부로 가능합니다.**

### 4.1 막히는 곳과 수치

| 단계 | 막힘 | 근거 |
|---|---|---|
| 환경 | `environment.yml`은 torch 2.7.0 + CUDA 12.8 휠, LIBERO가 mujoco 3.1.6을 고정. 로컬 Python 환경(torch 2.15 설치본)의 mujoco는 4.0.0이라 충돌 | `environment.yml`, 로컬 환경 확인 |
| 렌더링 | README는 `MUJOCO_GL=egl`(Linux). macOS 렌더링 경로는 확인하지 않음 | README |
| 장치 | 기본 `device = "cuda" if available else "cpu"`. MPS 경로가 없어 코드 수정 필요 | `base.py`, `trainer.py`, `pretrained/algorithm.py` |
| 데이터 | LIBERO 데모 (LIBERO 스크립트로 받음, 크기 미확인). 증강 롤아웃은 BC 체크포인트가 있어야 생성 | README, D10 |
| 가중치 | DINOv3 (HF gated, 토큰 필요), LRM (Qwen3-VL-8B, bf16 기준 약 16 GB로 추정, 미확인) | `encoders.py`, `lrm.py` |
| 메모리 | 데이터 10–23 GB (uint8, RAM 상주) + 학습 13.4 GiB (MPS 측정) | 03 §1.5, 03 §6 |
| 시간 | 한 스텝 3.21 s(인코더만, MPS). 50k 스텝 약 45 h | 03 §6 |

### 4.2 세부 판단

- **FQE 한 설정(LIBERO-Spatial, 시드 1, 50k 스텝):** 데이터 RAM 10.0 GB에 학습 13.4 GiB를 더하면 24 GiB를 넘습니다. 시간은 약 45 h(정책과 데이터 로딩 제외)입니다. 이 맥에서 안정적으로 끝낼 수 없습니다.
- **정답 MC(LIBERO 200 에피소드 x 정책 4개 x 과제 4개):** BC 체크포인트를 학습해야 합니다. 체크포인트는 미공개이고, 학습은 GPU 시간이 필요합니다. 시뮬레이션 시간은 측정하지 않았습니다.
- **LRM 라벨링(롤아웃 1000개):** 8B VLM의 추론. 가중치 다운로드와 메모리를 확인하지 않았습니다. 미세조정은 배치 1에 그래디언트 누적으로 짜여 있어 GPU에서 수 시간 이상으로 추정합니다.
- **세계모델, CALVIN, 실로봇:** 코드와 체크포인트가 없습니다. 맥에서 재현할 수 없습니다.

### 4.3 가능한 축소판 (조건부, 노력 추정)

- 같은 알고리즘(FQE + 사전학습 또는 소형 인코더 + 종료 라벨)을 MuJoCo 도달 과제 하나에서 돌립니다. 정답은 몬테카를로로 구하고, 라벨 오류를 주입해 D5–D7의 효과를 측정합니다.
- 새 설치는 필요 없습니다(mujoco 4.0, torch 2.15가 이미 있음, 승인됨). 자작 MJCF가 필요하고, 처리량은 측정하지 않았습니다.
- **노력 추정:** 축소판 1–2주(설계, 코드, 실험 세 번). 공식 파이프라인을 CUDA 기기에서 재현하려면 설치와 데이터 준비 1–2일, 세 시드 x 네 과제의 학습이 추가됩니다(CUDA 기기의 속도는 측정하지 않음). 이 맥의 전체 재현은 노력보다 **메모리와 시간**이 막습니다.

## 5. 저자에게 물을 질문

1. 표 1의 순위 상관은 정책 단위인가, 정책-과제 쌍 단위인가? 쌍이라면 같은 정책의 과제를 독립으로 셌는지.
2. Q 입력에 관절과 그리퍼 상태를 넣었는지(D3), 넣었다면 비교 공정성을 어떻게 보는지.
3. 목표 네트워크는 주기 동기화인지 Polyak인지(D1). 목표 클리핑이 결과에 영향을 주었는지(D2).
4. LRM 정확도의 클래스별 TPR과 TNR, 그리고 목표 정책 롤아웃에서 따로 잰 값. 정답 라벨을 쓴 RoboFQE와 LRM 라벨 RoboFQE의 ARE 비교(초록의 "최소 손실"의 근거).
5. 실로봇 표의 ±가 CI 반폭인지 SD인지(표 3 캡션과 A.2가 다름). LRM 83.3%와 정책 58.5%의 분모(250, 300)와 정확한 집계 방식.
6. CALVIN 정답의 샘플 수(0.96, 0.91, 0.87, 0.82, 0.76은 20의 배수가 아님).
7. 표 1 ARE가 Long의 세계모델 세 셀에서 V^π 캡션 값과 맞지 않는 이유(RMS 상한 초과, 03 §1.1), 표 5 데모만 행의 같은 문제.
8. "5 seeds"의 정의와 부트스트랩 CI가 리샘플링한 단위.
9. 정답 평가의 초기 상태가 학습 데이터와 같은 고정 초기 상태인지(D9).
10. A.3의 풀링 실험 코드와 CALVIN, 실로봇, 기준선 코드 공개 계획.
11. 표 1 Ctrl-World 행의 MMRV "+0.09"는 오타인지.

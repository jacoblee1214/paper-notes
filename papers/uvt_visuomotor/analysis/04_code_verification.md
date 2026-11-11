# 04. 논문 ↔ 공식 코드 대조 (UVT, arXiv:2608.03563)

**대상 저장소:** `https://github.com/DeFisch/UVT` (프로젝트 페이지의 Code 버튼이 가리키는 저장소). 얕은 복제는 `uvt_visuomotor/repo/`(gitignore)에 있고, 확인 시점의 HEAD는 `4c46246`(2026-03-11)입니다. 저장소에는 커밋 3개(2026-03-10 `0725777` 초기 커밋, 2026-03-11 `1fc1f2b`, `4c46246`)만 있고 태그와 릴리스가 없습니다.
**방법:** 정적 읽기만 했습니다. 코드를 실행하지 않았고, 패키지를 설치하지 않았고, 데이터와 가중치를 받지 않았습니다. 인용은 `파일:줄` 형식입니다.
**라이선스 주의:** 저장소에는 LICENSE 파일이 없습니다. 그래서 코드는 복사하지 않고 파일명과 줄 번호로만 인용합니다(권리 판정은 `01` 머리말 참조).

---

## 1. 요약

| 항목 | 판정 |
|---|---|
| MVAE 구조(두 인코더, 정밀도 가중 PoE, 두 디코더, KL) | **일치.** 논문 §III-C의 식과 같은 구조입니다. 다만 파라미터화 세부(두 층 MLP 256, 고정 초기 분산)는 논문에 없습니다 |
| MVAE 손실 가중치 | **불일치.** 논문은 가중치 1, 1, 1(β 없음)이고, 코드는 CE에 0.5, KL에 0.01과 워밍업을 씁니다 (D3) |
| VLA 정렬 손실 | **불일치.** 논문은 제곱 L2, 코드 기본값과 공개 명령은 L1입니다 (D1). 디코딩 손실 가중치도 논문 0.5, 공개 명령 1.0입니다 (D2) |
| 디코더를 VLA 학습 중 갱신 | **일치.** 학습 가능으로 불러옵니다 (`vla-scripts/finetune.py:1205–1221`) |
| 동역학 코드 | **일치하지만 세부가 빠짐.** 4개 코드 × 16 범주, UniVLA 2단계 LAM입니다. 논문 본문은 주 실험에 쓴 LAM을 밝히지 않았습니다 (D9) |
| LIBERO 평가 설정 | **부분 일치.** 과제 최대 스텝과 시드 7은 코드에 있습니다. 시행 수는 코드 기본 50, README 예시 20으로 서로 다르고, 논문은 밝히지 않았습니다 (D15) |
| π0.5 + UVT | **재현 불가.** 저장소에 π0.5 코드가 없습니다 (D13) |
| 실물 실험 | **재현 불가.** 실물 제어 코드, 양팔 설정, 원격 조작 코드가 없습니다 (D14) |
| 공개 명령의 실행 가능성 | **불일치 다수.** README와 `run_full_pipeline.sh`의 인자 이름이 스크립트와 맞지 않습니다 (D19). 정적 확인 기준으로 1단계부터 막힐 가능성이 큽니다 |
| **promotion 조건: 동역학 없는 행동 전용 대조** | **저장소와 논문 모두 답하지 않습니다.** §3 참조 |

---

## 2. 논문 항목과 코드 위치 대조

| 논문 항목 (절) | 코드 위치 | 판정 |
|---|---|---|
| 행동 청크 56차원 = 8 × 7 (§III-A, Fig. 5) | `scripts/fusion_mvae/configs.py:21`(chunk 8), `:22`(action 7), `:63–64`의 `input_action_dim` 계산 | 일치 |
| 동역학 코드 z_t = LAM(v_t, v_{t+k}) (§III-A) | `scripts/precompute_lam_indices.py:155–182`(첫 프레임과 `step_start + window − 1` 프레임으로 창 생성), `:293`(창 기본 8) | 일치. 창 길이와 행동 청크 길이의 관계는 명시되지 않음 (D10) |
| 인코더 두 개가 같은 32차원 가우시안을 냄 (§III-C) | `scripts/fusion_mvae/model.py:19–48`(행동), `:50–82`(LAM 코드, 원-핫 64차원 입력) | 일치. 구조 세부는 논문에 없음 |
| 정밀도 가중 PoE: τ = α_a τ_a + α_z τ_z, μ = (α_a τ_a μ_a + α_z τ_z μ_z)/τ (§III-C) | `model.py:158–181`(`poe_fuse`) | 일치. 단, `T_a = α_a/var_a`, `T_z = α_z/var_z`로 정밀도 가중이 맞습니다. 파라미터는 스칼라 로그 α 두 개(`model.py:151–152`) |
| α는 전역 학습 스칼라 (§III-C) | `model.py:151–152` | 일치. 다만 `model.py:172`의 주석 "(D,)"는 실제 형상(1)과 어긋납니다 (사소) |
| 고정 분산 초기값 (논문 없음) | `configs.py:39–40`, `model.py:36`, `:69` | **논문에 없음.** 행동 로그분산 −2, LAM 로그분산 +1 (D5) |
| 융합 평균 μ_t를 목표 u_t로 씀 (§III-C) | `model.py:204`(샘플링), `:238–239`(결정적 `pd_mu` 사용), `scripts/compute_u_for_subset.py:204` | 일치. 학습 목표는 샘플링이 아니라 평균입니다 |
| 디코더 g_a, g_z (§III-C) | `model.py:85–130` | 일치 |
| L_ACT = ‖a − â‖₁ (§III-C) | `scripts/fusion_mvae/train.py:38`(`F.l1_loss`, 평균) | 일치하지만 평균(mean)입니다. 논문의 ‖·‖₁가 합인지 평균인지 불명확 (D3) |
| L_DYN = CE(z, ẑ) (§III-C) | `train.py:42–45`(네 코드 평균 CE) | 일치 |
| L_KL = KL(N(μ,σ²) ‖ N(0,I)), 융합 후 분포 (§III-C) | `train.py:47–50`(융합 후 μ, logvar에 대해 계산) | **일치** |
| L_UVT = L_ACT + L_DYN + L_KL, 가중치 없음 (§III-C) | `train.py:52`: L1 + λ_z·CE + β·KL. `configs.py:34`(λ_z = 0.5), `:35`(β_max = 0.01), `:36`(워밍업 2000 스텝) | **불일치** (D3) |
| VLA 목표 ‖û − u‖²₂ (§III-D) | `vla-scripts/finetune.py:157`(`ut_loss_type = "l1"` 기본), `:473`(L1 사용), `:479`(MSE 분기) | **불일치** (D1). 공개 명령도 `l1` |
| 디코딩 손실 γ = 0.5 (§IV-A-d) | `finetune.py:159`(`ut_decode_loss_weight` 기본 0.0), 공개 명령 `1.0` (README, `run_full_pipeline.sh:126–127`) | **불일치** (D2) |
| 디코더 학습 중 갱신 (§III-D) | `finetune.py:1205–1221`(`requires_grad(True)`), `:1213`(MVAE 설정으로 재구성) | 일치 |
| 학습 시 행동은 복원 손실로만 사용 (§III-D) | `finetune.py:506–511`(디코딩 후 L1), `:506–508`(LIBERO만 그리퍼 변환 `(x+1)/2`) | 일치. 그리퍼 변환은 논문에 없음 (D18) |
| 해시 실패 창 ~6%는 손실에서 제외 (논문 없음) | `prismatic/vla/datasets/datasets_lam_precomputed.py:195–205`(0 벡터 대체), `finetune.py:463–470`(마스크) | **논문에 없음** (D17) |
| u 사전계산 후 VLA 학습 (§III-D) | `scripts/compute_u_for_subset.py:188–238`(인코딩, `u_vectors.npy`, `decoder_weights.pt`) | 일치 |
| 표준 LIBERO 평가: 과제 10개, 실행 한 번에 최대 스텝 (§IV) | `experiments/robot/libero/run_libero_eval.py:73–79`(suite별 최대 스텝 220/280/300/520), `:120`(시행 50), `:135`(시드 7) | 부분 일치 (D15) |
| 10k·100k 스텝 학습, 동일 데이터 분할 (§IV-A-d) | README 예시는 100k 스텝, 공개 명령의 `--max_steps` 값은 1만 또는 10만. 시드 설정이 `finetune.py`에 없음 | 부분 일치. 학습 시드 미설정 (D15) |

---

## 3. promotion 조건: 동역학 없는 행동 전용 대조 (핵심 질문)

**질문:** 같은 32차원 목표 파이프라인에서 LAM 동역학 코드만 뺀 대조(행동 전용)를 두면, 이득이 사라지는가? 즉 "동역학 신호 자체"가 도움이 되는가?

**결론: 답할 수 없습니다. 조건은 아직 열려 있습니다.** 근거는 세 가지입니다.

1. **논문에 없습니다.** Table III은 기준선(VLA-Adapter 회귀 헤드, 행동 L1)과 "동역학을 별도 헤드로 예측하는" 두 변형(Villa-X, UniVLA LAM 헤드)과 UVT만 비교합니다. 동역학 없이 행동만 32차원 MVAE로 압축한 대조가 없습니다. 그래서 UVT와 기준선의 차이는 (a) 동역학 정보, (b) 32차원 병목과 디코더, (c) 디코딩 손실과 L1·L2 차이, (d) 정렬 손실 가중치를 분리하지 못합니다.
2. **저장소에 스위치가 없습니다.** `scripts/fusion_mvae/model.py`의 MVAE는 항상 LAM 인코더를 만듭니다. `train.py`의 `lambda_z`를 0으로 두어도 LAM 인코더는 PoE 융합에 계속 들어가므로 z가 μ에 섞입니다. 그래서 `λ_z = 0`은 깨끗한 대조가 아닙니다. 동역학을 완전히 빼려면 (i) `poe_fuse`에서 z 정밀도 T_z를 0으로 두고, (ii) CE 항과 LAM 디코더를 제거해야 합니다. 이 변경은 저장소에 없습니다.
3. **학습 시 LAM 디코더 항이 u에 z 정보를 넣는 주된 경로입니다.** PoE 가중치만으로는 z 정보가 u에 들어가지 않습니다. 초기값에서 행동 가중치는 약 0.953(§4의 계산)이라 z의 직접 기여는 5% 정도입니다. 나머지 z 정보는 "u에서 z를 복원해야 한다"는 CE 항에서 옵니다. 따라서 올바른 대조는 **T_z 제거와 CE·LAM 디코더 제거를 함께** 하는 것입니다.

**코드 안에서 얻을 수 있는 간접 증거.** 저장소의 `train.py:141–180`은 `train/ce_lam`, `precision/ratio_a_over_z`(T_a/T_z) 같은 진단값을 기록합니다. 이 값들이 학습 중 어떻게 움직이는지는 W&B 기록이 없어서 확인할 수 없습니다. 특히 학습된 α와 분산이 초기 비율 20:1에서 얼마나 바뀌는지가 중요합니다. 논문은 이 값을 보고하지 않습니다.

**판정:** 조건 미충족. 노트의 "조건부 승격"은 그대로 유지합니다. 첫 실험은 이 대조입니다. 선형 toy(`analysis/uvt_toy.py`, `03` §6)는 이 질문에 대해 조건부 답을 줍니다. 동역학 고유 이득은 예측도 ρ가 크고 압축 차원 r이 신호 차원(4) 이상일 때 양수이며, r이 작고 데이터가 많으면 음수가 됩니다. 실제 LIBERO에서 같은 결과가 나올지는 실험으로만 알 수 있습니다.

---

## 4. 구현 가능성 판정

**질문 1: 공식 재현이 가능한가?** 조건부로 가능합니다. 다만 이 맥에서는 불가능합니다.

| 요소 | 상태 | 근거 |
|---|---|---|
| 학습 코드 | 있음. 다만 CUDA 전제 | `finetune.py`는 `torchrun`, DDP, `device_id`를 씀(`:250–262`). MPS 경로 없음 |
| LIBERO 데이터 | 외부 다운로드 필요 | README: `modified_libero_rlds`(git-lfs). 용량 수십 GB로 추정(확인 안 함) |
| 기반 VLM | 외부 다운로드 필요 | Prismatic Qwen2.5-0.5B + DINOSigLIP, 약 3 GB(README) |
| LAM 체크포인트 | 외부 다운로드 필요 | UniVLA stage-2 LAM, 약 200 MB(README) |
| UVT 체크포인트 | **공개되지 않음** | README의 체크포인트 표에 UVT 항목이 없습니다 |
| LIBERO 시뮬레이터 | 설치 필요 | README 설치 4단계. MuJoCo와 EGL 설정 |
| 실행 시간 | 10k 스텝 약 4–5시간(A100 80GB), 10k 스텝 RTX 4090 약 8–10시간(README 추정). 100k 스텝 40–50시간 | README 표. 논문은 A6000 기준 10k 스텝 약 8시간 |
| π0.5 대조 | **불가** | 저장소에 openpi나 π0.5 코드가 없습니다(검색 결과 0건) |
| 실물 | **불가** | 제어 코드와 하드웨어 설정이 없습니다 |

**공개 명령의 결함 (D19–D20):** 현재 상태로는 README 1단계부터 인자 불일치로 막힐 가능성이 큽니다(정적 확인이며 실행하지 않았습니다). 고치는 데 코드 수정 몇 줄이 필요하고, `predict_u` 메서드는 없습니다.

**질문 2: 이 맥(Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음, torch 미설치)에서 할 수 있는가?**

- **VLA 학습과 LIBERO 평가: 불가(현실적으로).** CUDA 전제 코드입니다. MPS로 포팅해도 0.5B 모델을 10k 스텝 LoRA 학습하는 것은 실용적이지 않습니다. 추정이며 측정하지 않았습니다.
- **MVAE 학습: 가능.** 약 31만 파라미터(§4 계산)라서 CPU로 몇 분에서 수십 분이면 됩니다. 단, 입력 데이터(LAM 코드)를 만들려면 LAM 추론과 RLDS 읽기가 필요합니다. 그 부분은 CPU로 될지 확인하지 않았습니다.
- **선형 또는 소형 toy: 가능.** 이미 `analysis/uvt_toy.py`로 했습니다.

**질문 3: 노력과 필요한 자원.** LIBERO-Spatial에서 기준선과 UVT를 10k 스텝으로 비교하는 최소 재현은 NVIDIA GPU 1장(24 GB 이상)과 1–2주 정도의 작업(인자 수정, 데이터 준비, 시드 여러 개)이 필요하다고 추정합니다. 이것은 추정이며 확인하지 않았습니다.

**최종 판정:** 공식 재현은 **조건부 가능**(NVIDIA GPU 필요, 공개 명령을 고쳐야 함, π0.5와 실물은 불가). 이 맥에서는 **학습 재현 불가, MVAE와 toy 검증은 가능**.

---

## 5. 설정과 시드 (논문 대 코드)

| 항목 | 논문 | 코드 기본값 | 공개 명령 |
|---|---|---|---|
| MVAE 잠재 차원 | 32 | 32 (`configs.py:19`) | 32 |
| MVAE 은닉 차원, 층 수 | 없음 | 256, 두 층 (`configs.py:20`, `model.py`) | 기본값 |
| MVAE 배치, 학습률, 에폭 | 없음 | 256, 1e-3에서 1e-5 코사인, 200 (`configs.py:27–30`) | 500 에폭(`run_full_pipeline.sh:77`) |
| MVAE 시드 | 없음 | 42 (`configs.py:45`), 검증 분할 10% | 기본값 |
| MVAE 손실 가중치 | 1, 1, 1 | λ_z 0.5, β 0.01, 워밍업 2000 (`configs.py:34–36`) | 기본값 |
| VLA 정렬 손실 | 제곱 L2 | L1(`finetune.py:157`) | L1 |
| VLA 디코딩 가중치 | γ = 0.5 | 0(`finetune.py:159`) | 1.0 |
| VLA 학습률, 배치 | 논문 본문에 없음 | 명령에서 지정 | README: 1e-4, 배치 8 × 누적 2. 파이프라인: 1e-4, 배치 4 × 누적 4 (효과 배치 16으로 같음) |
| LoRA 랭크 | 없음 | 명령에서 지정 | 64 |
| 입력 이미지 수 | 없음 | 명령에서 지정 | README 2, 파이프라인 3 (서로 다름) |
| VLA 학습 시드 | 없음 | 설정 없음(`finetune.py`에 `set_seed` 호출 없음) | — |
| LIBERO 평가 시행 | 없음 | 과제당 50(`run_libero_eval.py:120`) | README 예시 20 |
| LIBERO 평가 시드 | 없음 | 7(`:135`) | — |
| 동역학 코드 | "discrete code" | 4개 × 16 범주(UniVLA 2단계 VQ, `latent_action_model/genie/modules/lam.py:48`) | UniVLA LAM |

---

## 6. 차이 목록 (D1–D27)

| ID | 분류 | 내용 | 근거 |
|---|---|---|---|
| D1 | 불일치 | VLA 정렬 손실: 논문은 제곱 L2, 코드 기본과 공개 명령은 L1 | 논문 §III-D; `finetune.py:157, 473, 479`; README 명령 `--ut_loss_type l1` |
| D2 | 불일치 | 디코딩 손실 가중치: 논문 γ 0.5, 코드 기본 0, 공개 명령 1.0 | 논문 §IV-A-d; `finetune.py:159`; `run_full_pipeline.sh:127` |
| D3 | 불일치 | MVAE 손실: 논문 가중치 1, 1, 1(β 없음). 코드 CE에 0.5, KL에 0.01과 워밍업. L1은 평균 | `train.py:38, 42–52`; `configs.py:34–36` |
| D4 | 누락 | 행동 전용 32차원 대조, 동역학 제거 스위치 없음 | §3 |
| D5 | 누락 | 고정 로그분산 초기값(−2, +1)은 초기 행동 가중치 약 0.953을 만듦. 논문에 없음 | `configs.py:39–40`; `model.py:36, 69`; §4 계산 |
| D6 | 일치 | 융합 후 KL, 융합 평균을 목표로 씀 | `train.py:48–50`; `model.py:238–239` |
| D7 | 누락 | 인코더와 디코더 구조(두 층 MLP 256, 활성 ReLU) | `model.py:19–130` |
| D8 | 일치 | 동역학 코드 4개 × 16 범주, CE | `configs.py:23–24`; `precompute_lam_indices.py:270` |
| D9 | 누락 | 주 실험의 LAM 종류. 공개 파이프라인은 UniVLA LAM | `run_full_pipeline.sh:25`; README 체크포인트 표. 논문은 Table III에서만 Villa-X와 UniVLA를 언급 |
| D10 | 불명확 | LAM 창 길이와 행동 청크 길이의 관계. 코드에서 MVAE 청크 8, 창 기본 8, 파이프라인 25, README 예시 25 | `configs.py:21`; `precompute_lam_indices.py:293`; `run_full_pipeline.sh:15`; README |
| D11 | 불명확 | MVAE는 데이터셋마다 따로 학습. 논문 Fig. 5는 네 suite를 한 MVAE로 본 것처럼 쓰임 | `configs.py:50`(기본 libero_spatial); `run_full_pipeline.sh:42–71`(데이터셋 인자) |
| D12 | 불일치 | MVAE 에폭: 코드 기본 200, 파이프라인 500 | `configs.py:30`; `run_full_pipeline.sh:77` |
| D13 | 재현 불가 | π0.5 + UVT 결과는 이 저장소에 코드가 없음 | 저장소 전체 검색(openpi, π0.5, pi0 키워드 0건) |
| D14 | 재현 불가 | 실물 실험 코드 없음(양팔 제어, 원격 조작, 과제 설정) | 저장소 파일 목록 |
| D15 | 누락 | 시행 수(코드 50, README 20), LIBERO 평가 시드 7, VLA 학습 시드 없음 | `run_libero_eval.py:120, 135`; `finetune.py`(set_seed 없음); README |
| D16 | 명명 | 평가 코드의 `use_ut_head`는 `VillaXLatentHead`를 불러옴(이름 잔재). 이 헤드는 잔차 블록 24개의 MLP | `run_libero_eval.py:143, 181–196`; `prismatic/models/action_heads.py:170–200` |
| D17 | 누락 | 해시 실패 ~6% 창은 0 벡터로 채우고 손실에서 마스크 | `datasets_lam_precomputed.py:195–205`; `finetune.py:463–470` |
| D18 | 누락 | 디코딩 후 LIBERO 그리퍼 변환 `(x+1)/2` | `finetune.py:506–508` |
| D19 | 결함 | 공개 명령과 스크립트의 인자 불일치. `train.py`는 `--data_npz`를 받는데 README와 파이프라인은 `--data_path`. `compute_u_for_subset.py`에는 `--dataset_path`, `--lam_indices_dir`, `--output_dir`이 없음. `extract_fusion_data_generic.py`는 `--data_root`, `--precomputed_dir`, `--output_npz`를 받음. `precompute_lam_indices.py`는 `--datasets`를 받음 | `train.py:241`; `compute_u_for_subset.py:67–76`; `extract_fusion_data_generic.py:62–68`; `precompute_lam_indices.py:289–299`; README와 `run_full_pipeline.sh` 명령 |
| D20 | 결함 | README 빠른 시작의 `vla.predict_u`는 저장소에 없음 | README:114 (전체 검색 0건) |
| D21 | 사소한 결함 | `train.py:108`의 except 절이 import 실패 시 정의되지 않은 `wandb`를 참조 | `train.py:101–110` |
| D22 | 이식성 | 개발 서버의 절대 경로가 하드코딩됨(평가 스크립트의 LIBERO 경로, `compute_u_for_subset.py` 문서 문자열). 공개 문서에는 쓰지 않음 | `run_libero_eval.py:13`; `compute_u_for_subset.py:12–16` |
| D23 | 라이선스 | LICENSE 파일 없음. `pyproject.toml`은 LICENSE 파일을 가리키고 MIT 분류자를 씀. GitHub 라이선스는 none. VLA-Adapter 계열(prismatic)과 UniVLA LAM 코드가 들어 있음 | `pyproject.toml:15–20`; GitHub API |
| D24 | 버전 | 저장소 최신 커밋(2026-03-11)이 arXiv 제출(2026-08-04)보다 5개월 앞섬. 논문 결과와 어느 커밋이 맞는지 확인할 수 없음 | `gh api` 커밋 목록 |
| D25 | 누락 | UVT 체크포인트 링크 없음(README 체크포인트 표에 기반 모델과 LAM만) | README 71–85 |
| D26 | 재현 불가 | LIBERO-Plus 평가 코드 없음 | 저장소 전체 검색 |
| D27 | 확인 불가 | 논문 Fig. 5(t-SNE)와 Fig. 6(부드러움)의 생성 코드 없음 | 저장소 파일 목록 |

---

## 7. 자세히 본 부분 (재현 관점의 결론)

- **MVAE 자체는 충실합니다.** 식과 코드가 같은 구조이고, 정밀도 가중이 맞습니다. 논문 식 대조에서 불일치는 손실 가중치와 β(D3)에 있습니다. 이 차이는 "동역학 항의 크기"를 바꾸므로 결과 해석에 직접 영향을 줍니다.
- **VLA 쪽은 공개 명령이 논문 설정과 다릅니다(D1, D2).** 논문 표가 어떤 설정에서 나왔는지 코드만으로는 알 수 없습니다.
- **평가 쪽은 시행 수와 시드를 알 수 없습니다(D15).** LIBERO 수치를 통계적으로 해석하려면 이것이 필요합니다. 본 분석은 suite당 500회를 가정했습니다(`03` §1).
- **공개 저장소는 "논문의 방법을 그대로 실행하는" 수준이 아닙니다.** 명령을 고쳐야 하고, 기반 모델 하나가 빠져 있고, 체크포인트가 없습니다.

## 8. 계산 기록 (정적)

- MVAE 파라미터 수(입력 56 행동, 64 원-핫 LAM, 은닉 256, 잠재 32, 디코더 56과 64 출력): 행동 인코더 96,832, LAM 인코더 98,880, 행동 디코더 88,632, LAM 디코더 24,896, 합계 약 309,242(α 두 개 포함). 계산은 층 크기에서 파이썬 산술로 구한 값이며, 저장소 코드를 실행한 것이 아닙니다.
- 초기 정밀도 비율: 행동 분산 e^{−2} ≈ 0.135, LAM 분산 e^{1} ≈ 2.718, α = 1일 때 T_a = 7.39, T_z = 0.368, 행동 가중치 w_a = T_a/(T_a + T_z) ≈ 0.953, T_a/T_z ≈ 20.1.
- 이 값은 학습 전 초기값입니다. 학습 후 분포는 확인할 수 없습니다.

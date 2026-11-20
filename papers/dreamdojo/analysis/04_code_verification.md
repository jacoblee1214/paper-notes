# 04. 코드 대조: 논문 ↔ 공식 저장소 (NVIDIA/DreamDojo)

공식 코드가 있으므로 `04_code_verification.md`로 씁니다. 저장소는 `repo/`에 얕은 clone(`--depth 1`)으로 받았고, commit은 `02f119b`(2026-03-21, "Add distillation")입니다. **정적으로만 읽었습니다.** 실행, import, 설치, 가중치와 데이터 다운로드는 하지 않았습니다. 저장소 코드와 설정은 신뢰할 수 없는 입력으로 보고 읽었습니다.

경로는 모두 `repo/` 기준입니다. 줄 번호는 위 commit 기준입니다.

## 1. 확인한 파일

| 주제 | 파일 |
|---|---|
| 잠재 행동 모델(LAM) 구조와 손실 | `external/lam_project/lam/model.py`, `external/lam_project/config/lam_0.yaml` |
| 상대 행동 계산 | `cosmos_predict2/action_conditioned.py` (`_get_actions`, 67행 이후), `groot_dreams/data/dataset.py` (`_get_all_steps`, 418행) |
| 행동 청크 주입 | `cosmos_predict2/_src/predict2/action/networks/action_conditioned_minimal_v1_lvg_dit.py` (198행), `configs/2b_480_640_pretrain.yaml` |
| 흐름 매칭과 시간 일관성 손실 | `cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py` (900–926행), 하위 클래스 `cosmos_predict2/_src/predict2/action/models/action_conditioned_video2world_rectified_flow_model.py` |
| 사전학습 혼합, 길이, 배치 | `configs/2b_480_640_pretrain.yaml` |
| 최적화 설정 | `cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_action_conditioned_rectify_flow_gr00t.py` (690, 821행 근처), `.../exp_2B_action_conditioned_rectify_flow.py` |
| 증류 | `docs/DISTILL.md`, `cosmos_predict2/_src/predict2/interactive/models/action_video2world_warmup.py` (36행 `denoising_step_list`) |
| 평가 | `docs/EVAL.md` |
| 문서 | `README.md`, `docs/SETUP.md`, `docs/LAM.md`, `docs/PRETRAIN.md` |

## 2. 항목별 대조

판정: **일치**, **부분 일치**(값은 맞으나 조건이 다름), **불일치**, **확인 불가**(저장소에 없음).

| # | 항목 | 논문 | 저장소 | 판정 |
|---|---|---|---|---|
| 1 | LAM 잠재 행동 차원, 입력 | 32차원, 연속 두 프레임 (§3.3.2, §4.1) | `lam_0.yaml`: 잠재 차원 32, 두 프레임 (`num_frames: 2`) | 일치 |
| 2 | LAM KL 가중치 β | 10⁻⁶ (§4.1) | `lam_0.yaml` 11행 `beta: 0.000001`. 다만 `lam/model.py` 31행 기본값은 0.01 | 일치 (설정이 기본값을 덮음) |
| 3 | LAM 옵티마이저 | AdamW, WD 0.01, lr 2.5×10⁻⁵ (§4.1) | `lam_0.yaml`: AdamW, lr 2.5e-5, WD 1e-2 | 일치 |
| 4 | LAM 학습 스텝과 배치 | 400k 스텝, 배치 256 (§4.1) | 체크포인트 이름 `LAM_400k.ckpt` (`text2world_model_rectified_flow.py` 187행). `lam_0.yaml`은 배치 20 × GPU 8 = 160 | 부분 일치 (배치는 GR-1 예시 설정이라 다를 수 있음) |
| 5 | LAM 크기 | 700M, 인코더 24블록, 디코더 24블록 (§4.1) | `lam_model_dim` 1024, 인코더·디코더 24블록, 헤드 16, 패치 16. 파라미터 수는 세지 않음 | 부분 일치 (블록 수 일치, 크기 미확인) |
| 6 | 상대 행동 (재기준) | 잠재 프레임 시작 자세(4 스텝마다)로 재기준 (§3.3.1) | `_get_actions`(67행)는 **연속한 두 스텝** 사이의 상대 이동과 회전을 이전 프레임 기준으로 계산하고 20을 곱함(118행). `groot_dreams`는 상대 변환 코드가 없고 저장된 행동을 정규화만 함. 학습 데이터 전처리는 저장소 밖 | **불일치 또는 확인 불가** (D2) |
| 7 | 행동 청크 주입 | 연속 행동 4개를 잠재 프레임 하나에 묶음 (§3.3.1) | `temporal_compression_ratio: 4`, `num_action_per_chunk: 12` (pretrain yaml 24–26행). 네트워크는 `_num_action_per_latent_frame = 4` (198행) | 일치 |
| 8 | 행동 임베더 마지막 층 0 초기화 | 0으로 초기화 (§3.3.2) | `zero_init_action_embedder: true` (pretrain yaml) | 일치 |
| 9 | 시간 일관성 손실 식 (4) | 프레임 간 차분의 제곱합, 예측 속도와 정답 속도 차이 | 기호는 같음. **차분 축이 시간이 아님**(D1) | **불일치** (D1) |
| 10 | 손실 가중치 λ | 0.1 (식 5) | 922–925행 `... * 0.1` | 일치 |
| 11 | 사전학습 혼합 비율 | In-lab : EgoDex : DreamDojo-HV = 1 : 2 : 10 (§4.1) | `dataset_mixing_weights: [1, 2, 10]` (pretrain yaml 37–40행) | 일치 |
| 12 | 시퀀스 길이 | 사전학습 13 프레임, 사후학습 13 프레임과 행동 12개 (§4.1) | `num_frames: 13`, `num_action_per_chunk: 12`, `state_t: 4 # 1 + 12 // 4` | 일치 |
| 13 | 사전학습 배치 | 유효 배치 1024, H100 256장 (§4.1) | `batch_size: 4` × GPU 256 = 1024 | 일치 (산술) |
| 14 | 사전학습 학습률과 WD | lr 1.6×10⁻⁴, WD 0.1 (§4.1) | `exp_2B_action_conditioned_rectify_flow_gr00t.py`의 두 설정에 `lr=16e-5`, `weight_decay=0.1`. 같은 파일의 다른 설정과 `..._rectify_flow.py`에는 2⁻¹⁴·⁵ ≈ 3.05e-5가 있음. 어느 설정이 DreamDojo 사전학습인지 확정하지 못함 | 부분 일치 (대응 설정 미확정, D4) |
| 15 | EMA | 학습 전체에 EMA, 결과 생성에 사용 (§4.1) | `register_ema()` (`action_conditioned/config.py` 88행) | 일치 |
| 16 | 해상도 | 세계모델 640×480, LAM 320×240 (§4.1) | 설정 이름 `2b_480_640_*`. LAM 해상도는 데이터 로더에서 확인 못함 | 부분 일치 |
| 17 | 기반 모델 | Cosmos-Predict2.5, WAN2.2 토크나이저 (§2) | `cosmos_predict2` 패키지, 토크나이저 `wan2pt2.py` | 일치 |
| 18 | 증류 3단계 | 워밍업 (교사 ODE 궤적), Self Forcing 증류, 분포 일치 손실 (§3.3.4) | `DISTILL.md`: 교사 생성 → 워밍업 → Self-Forcing, Self-Forcing에서 DMD 사용 | 일치 |
| 19 | 학생 디노이징 스텝 | 4 (§4.1) | 워밍업 `denoising_step_list = [0, 9, 18, 27]` (36행), 4개 | 일치 |
| 20 | 교사 디노이징 스텝 | 35 (§4.1) | 확인 안 함 | 확인 불가 |
| 21 | 학생 인과 어텐션 윈도 | 12 프레임 (§4.1) | 워밍업 `window_size=(self.config.cache_frame_size, -1, -1)` (129행). `cache_frame_size` 값은 확인 안 함 | 확인 불가 (부분) |
| 22 | 평가 설정 | 샘플 100개, 49 프레임 (§4.1) | `docs/EVAL.md`: `--num-frames 49 --num-samples 100` | 일치 |
| 23 | 평가 세트 | In-lab, EgoDex, DreamDojo-HV 평가셋, 반사실 평가셋, 배경을 바꾼 평가셋 2종, GR-1 Long 평가 (§4.1, §4.6) | `EVAL.md`에 In-lab, EgoDex, DreamDojo-HV 세 개만. 반사실, novel, GR-1 Long 설정은 없음 | 부분 일치 (D8) |
| 24 | 사람 선호도 평가 웹 UI, 12명 | 웹 UI, 12명 (§4.1, 부록 C) | 없음 | 확인 불가 (D10) |
| 25 | 정책 평가 (AgiBot, 20장면, Pearson, MMRV) | §4.7 | 저장소에서 `pearson`, `MMRV` 문자열 없음 | 확인 불가 (D10) |
| 26 | 가치 모델 (DINOv2, 4프레임) | 부록 D.6 | `value_model`, `dinov2` 문자열 없음 | 확인 불가 (D10) |
| 27 | 모델 기반 계획 | §4.7 | 계획 코드 없음 | 확인 불가 (D10) |
| 28 | 공개 체크포인트 | 2B, 14B, LAM (README) | README와 `docs/PRETRAIN.md`가 HF `nvidia/DreamDojo`를 가리킴 | 공개 주장 확인 (다운로드 안 함) |
| 29 | 공개 데이터 | DreamDojo-HV, In-lab, EgoDex (§3.2) | 설정 파일이 `/mnt/amlfs-…` 내부 경로를 가리킴. README는 GR-1 사후학습 데이터와 평가셋만 공개 | **불일치 (공개 범위)** (D7) |

## 3. 차이 목록

- **D1. 시간 일관성 손실의 차분 축 (가장 중요).** `text2world_model_rectified_flow.py` 922–925행의 코드는 다음과 같습니다.

  ```python
  per_instance_motion_consistency_loss = torch.mean(
      ((vt_pred_B_C_T_H_W[:, 1:] - vt_pred_B_C_T_H_W[:, :-1]) - (vt_B_C_T_H_W[:, 1:] - vt_B_C_T_H_W[:, :-1])) ** 2, ...
  ```

  변수 이름은 `B_C_T_H_W`입니다. 그러면 `[:, 1:]`은 시간 축(2번)이 아니라 **채널 축**(1번)을 자릅니다. 식 (4)의 프레임 간 차분을 구현하려면 `[:, :, 1:]`이어야 합니다. 아래 형태 검사로 확인했습니다. 텐서 크기는 (B, C, T, H, W) = (2, 16, 13, 4, 4)로 가정했습니다.

  ```
  code slice [:,1:]           -> (2, 15, 13, 4, 4)   (축 1 = 채널)
  paper Eq.4 diff over T      -> (2, 16, 12, 4, 4)   (축 2 = 시간)
  ```

  정적 판독과 형태 검사이며 학습을 돌려 확인하지는 않았습니다. 이 손실이 표 5의 "시간 일관성" 행에 들어간 실험 코드인지는 저장소에 없어 확인하지 못했습니다. 표 5의 이득(counterfactual PSNR +0.197 dB, GR-1 val +0.004 dB)은 이 구현에서 나온 값일 수 있으므로 저자에게 확인이 필요합니다.

- **D2. 상대 행동 계산.** 논문은 잠재 프레임 시작 자세로 재기준한다고 합니다. 저장소의 `_get_actions`는 연속한 두 스텝 사이의 상대 이동을 계산하고 20을 곱합니다. 그리고 학습 데이터 로더(`groot_dreams`)에는 상대 변환이 없습니다. 따라서 논문의 재기준은 저장소 밖 전처리에서 이루어졌을 수 있습니다. 어느 쪽인지 확정할 수 없습니다.
- **D3. 행동 표현의 차원.** `_get_actions`의 출력은 위치 3 + 회전 3 + 그리퍼 1(7차원)이고, 옵션으로 쿼터니언 8차원입니다. 논문은 "관절 자세"라고 쓰고 GR-1 설정은 팔과 손과 허리 관절 상태를 씁니다(`groot_configs.py`). 두 경로가 함께 있어서 논문 설명과 어느 경로가 쓰였는지가 명확하지 않습니다.
- **D4. 학습률 설정의 대응.** lr 1.6e-4와 WD 0.1은 `exp_2B_action_conditioned_rectify_flow_gr00t.py`에 있지만, 그 파일에는 lr 2⁻¹⁴·⁵와 3e-5 설정도 있습니다. 논문의 어느 단계(사전학습, 사후학습, 증류)에 대응하는지는 저장소 설명만으로 정해지지 않습니다.
- **D5. LAM 배치.** 저장소 설정은 GPU당 20 × 8장 = 160이고 논문은 256입니다. 저장소 설정은 GR-1 예시이므로 논문 설정과 같은지 알 수 없습니다.
- **D6. LAM β의 기본값.** `lam/model.py`의 기본 β는 0.01이고 설정 파일은 10⁻⁶입니다. 설정이 이기므로 결과에는 영향이 없을 것으로 보지만, 기본값만 쓰면 논문과 다른 모델이 됩니다.
- **D7. 데이터 공개 범위.** 사전학습 데이터(DreamDojo-HV 약 1.1M 에피소드, In-lab, EgoDex)는 저장소에 없고 설정은 내부 경로를 가리킵니다. 공개된 것은 GR-1 사후학습 데이터와 평가셋(In-lab_Eval, EgoDex_Eval, DreamDojo-HV_Eval)입니다. 사전학습 재현은 공개 데이터로 할 수 없습니다.
- **D8. 평가 설정 누락.** 논문의 반사실 평가셋, novel 평가셋(각 25개, 배경 편집), GR-1 Long 평가(600 프레임)의 실행 설정이 `EVAL.md`에 없습니다.
- **D9. 기준선 코드.** 논문의 기준선 Cosmos-Predict2.5(사전학습 없음)는 같은 저장소 설정에서 사전학습 데이터만 빼면 되는지 확인하지 못했습니다.
- **D10. 실험 코드 없음.** 사람 선호도 웹 UI, 정책 평가(Pearson, MMRV), 가치 모델, 모델 기반 계획, 실시간 원격조작 코드는 저장소에 없습니다(문자열 검색 결과). 표 4, 그림 5, 그림 6은 재현 코드 없이 보고 수치로만 확인할 수 있습니다.

## 4. 구현 가능성 (이 맥 기준, 추정)

기기: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음. 아래 판정은 저장소 설정과 파라미터 수에서 추정했고, 실제로 돌리지 않았습니다.

| 구성 요소 | 필요 자원 (논문·저장소 기준) | 이 맥에서 | 근거 |
|---|---|---|---|
| 사전학습 (2B, 14B) | H100 256장, 140k 스텝, 공개되지 않은 데이터 | **불가** | D7. 자원 규모 |
| 사후학습 (GR-1 등) | H100 128장, 50k 스텝 (논문) | **불가** | 자원 규모 |
| 잠재 행동 모델 학습 | 8 GPU, 400k 스텝 (`docs/LAM.md`, `lam_0.yaml`) | **불가** | 자원 규모 |
| 증류 (교사 생성, 워밍업, Self-Forcing) | H100 64장, 워밍업 10k 궤적 | **불가** | 자원 규모. 학생 체크포인트 공개 여부 확인 안 함 |
| 2B 추론 (49 프레임, 35 스텝) | 저장소는 H100과 CUDA 패키지(cu128, cu130)에서 시험됨 | **조건부** | 가중치 약 4 GB (bf16, 2B × 2바이트 추정). CUDA 전용 경로 포팅 필요. 메모리와 시간 미측정 |
| 14B 추론 | 가중치 약 28 GB (bf16 추정) | **불가 (추정)** | 24 GiB 메모리 초과 |
| LAM 추론 (잠재 행동 추출) | 700M (추정 약 1.4 GB bf16), 320×240 | **조건부** | 추론 속도와 MPS 호환 미확인 |
| 표 재계산과 토이 | numpy, 표준 라이브러리 | **가능** | `03` 참고 |

노력 추정: 2B 추론을 MPS에서 돌리는 최소 경로는 의존성 정리와 CUDA 전용 코드 교체까지 포함해 **3–5일**입니다(추정). 논문 수치 재현은 노력보다 자원이 막습니다.

## 5. 저자에게 물을 것

1. 시간 일관성 손실의 `[:, 1:]`이 채널 축인지, 실험에 쓴 코드와 저장소 코드가 같은지.
2. 상대 행동의 재기준 방식(잠재 프레임마다 4 스텝인지, 연속 스텝인지)과 그 전처리 코드.
3. 사후학습 학습률과 사전학습 학습률이 설정 파일 중 어느 것에 대응하는지.
4. 반사실, novel, GR-1 Long 평가의 실행 설정과 평가셋 공개 여부.
5. 사람 선호도 평가 UI와 정책 평가 스크립트 공개 여부.

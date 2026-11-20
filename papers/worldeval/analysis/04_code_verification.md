# WorldEval 공식 코드 대조 (04)

공식 저장소 `liyaxuanliyaxuan/Worldeval`(MIT, 커밋 `46084268031412973a2b6eed425a51439cfa6e3e`, 2025-06-11 15:15:59 +0800)을 `repo/`에 얕게 clone해서 **정적으로만** 읽었습니다. 실행, import, 설치는 하지 않았습니다. 저장소는 Wan2.1·DiffSynth Studio를 포크한 것입니다(Apache-2.0 원본, 수정 부분 MIT, README).

## 1. 공개 범위

| 논문 구성 요소 | 공식 코드 | 판정 |
|---|---|---|
| Wan 2.1 14B I2V + LoRA 학습 (§3.2) | `wanvideo/train_wan_t2v_act_embed.py`, `diffsynth/` | 있음 |
| 행동 조건 경로: 투영 → α 스케일 → 텍스트 문맥에 더하기 (§3.2) | `diffsynth/pipelines/wan_video_act_embed.py` L355–365 | 있음 (D3) |
| 행동 임베딩 추출기(정책의 액션 디코더 직전 잠재 벡터) | 없음 | **없음** (D1) |
| VQVAE / one-hot 행동 인코더 (표 2 비교군) | `wanvideo/encode_action/vqvae_*.py`, `onehot_encode.py` | 있음 (재학습 스크립트 포함) |
| 추론 (50 step, CFG 5.0) | `wanvideo/infer.py`, `pipelines/wan_video_act_embed.py` | 있음 |
| FID·LPIPS·DISTS 계산 | `wanvideo/utils/compute_metrics.py` (LPIPS·DISTS 출력은 주석) | 부분 (D7) |
| Gemini-2.0 성공 판정 (§3.3) | 없음 | **없음** |
| 정책 성공률·Pearson r·MMRV 계산 (표 1–2) | 없음 | **없음** |
| 실물 로봇 제어, 데이터 수집 | 없음 | 없음 |
| RoboTwin 실험과 MidJourney 시각 변환 (부록 C) | 없음 | 없음 (D8) |
| DexVLA, π0, DP 가중치 | README가 HuggingFace 링크 제공 (`kuromivv/…`) | 외부 |

## 2. 논문과 코드가 맞는 곳

- LoRA 설정: rank 16, alpha 16, 학습률 1e-4 (논문 §4 구현 상세 ↔ README `--lora_rank 16 --lora_alpha 16 --learning_rate 1e-4`). 일치합니다.
- 생성 프레임 81, 추론 50 step (논문 부록 C ↔ `infer.py` `num_inference_steps=50`, 프레임 기본값 81). 일치합니다.
- 투영층이 MLP(`action_proj.0`, `action_proj.2`)이고 α가 학습 가능한 스칼라(`action_alpha.requires_grad = True`). 논문 §3.2의 "학습 가능한 가중치 α"와 맞습니다.
- VQVAE 비교군은 `encode_action/vqvae_train.py`로 재학습이 가능합니다. 논문은 "Discrete Policy 프로토콜을 따른다"고만 씁니다.

## 3. 차이 (D1–D8)

**D1. 행동 임베딩 추출기가 공개되지 않았습니다 (핵심).** 논문 §3.2는 정책의 액션 디코더 직전 잠재 벡터를 쓴다고 씁니다. 저장소의 학습·추론 코드는 `encoded_action`을 담은 `.pt` 파일을 **읽기만** 합니다(`train_wan_t2v_act_embed.py` L121, `infer.py` L18). 파일을 만드는 코드는 VQVAE·one-hot 인코더뿐입니다. README는 DexVLA와 `lerobot`(π0)을 외부 추출 출발점으로 가리킬 뿐입니다. 따라서 **Policy2Vec 자체는 이 저장소만으로 재현할 수 없습니다.** 추출 지점(어느 층, 어느 토큰, 몇 차원)이 논문에 명시되지 않았습니다. `action_dim` 기본값 1280이 그 차원인지도 확인할 수 없습니다.

**D2. 해상도.** 논문 §4와 부록 C는 480×480이라고 씁니다. 데이터 클래스 기본값은 `height=480, width=832`이고(`train_wan_t2v_act_embed.py` L19), README의 `data_process.sh`도 `--height 480 --width 832`입니다. 가로세로비가 다릅니다.

**D3. 행동을 시간축으로 평균냅니다.** `wan_video_act_embed.py` L359–362:
```python
action = action.unsqueeze(0)
action = action.float().mean(dim=1)     # 시간축 평균
action_emb = dit.action_proj(action)
enhanced_ctx = original_ctx + action_emb * dit.action_alpha
```
청크 전체가 하나의 벡터로 줄어듭니다. 논문은 이 평균을 쓴다고 적지 않습니다. 같은 평균을 가진 두 궤적(빠른 증가와 빠른 감소)은 같은 조건을 받습니다. `analysis/toy_temporal_pool.py`는 이 경로에서 L2 거리 7.8인 두 궤적의 조건 차이가 1e-15 수준임을 보였습니다. 논문의 "정책 행동을 영상이 따른다" 주장(§4의 질문 3)을 생각하면 중요한 차이입니다. 다만 조건 벡터가 실제로 시간 정보를 잃는지는 학습된 `action_proj`의 가중치 없이는 단정할 수 없습니다. 이 toy는 무작위 투영으로 같은 연산 구조를 흉내 낸 것입니다.

**D4. 학습 에폭.** 논문 부록 C: "30 epochs". README 학습 명령: `--steps_per_epoch 500 --max_epochs 40`. 추론 출력 폴더명은 `lora_act_alpha_0.3_dex_ep30`입니다. 어느 체크포인트가 논문 결과에 쓰였는지 확인할 수 없습니다.

**D5. α 값.** 논문은 α 값을 보고하지 않습니다. 코드 기본값은 학습 0.1(`train_…py` L298), 추론 0.1(`infer.py` L150), README 예시는 0.3입니다. 어느 값이 표 1–3 결과에 쓰였는지 불명입니다.

**D6. 평가 파이프라인 부재.** Gemini-2.0 판정, 성공률 집계, Pearson r, MMRV, 체크포인트 비교, 안전 검출 실험 코드가 없습니다(§1 표). 표 1–3과 그림 4–7은 이 저장소로 다시 만들 수 없습니다.

**D7. `compute_metrics.py`가 계산하는 것.** FID를 `torchmetrics`로 계산하는데, 기준 프레임은 HDF5 에피소드의 `observations/images/cam_high`(상단 카메라)에서 옵니다. 생성 영상과 **같은 에피소드의 실제 프레임**을 짝지어 비교합니다. 논문 표 3의 FID가 정책 롤아웃의 실제 영상과 비교한 것인지, 시연 데이터와 비교한 것인지 코드만으로는 알 수 없습니다. LPIPS와 DISTS는 계산하지만 결과에 쓰지 않습니다(주석 처리). 또 FID는 첫 81프레임을 299×299로 바꿔 쓰므로, 한 과제당 샘플 수가 적으면 편향이 큽니다(`compute_metrics.py`의 `main`).

**D8. 실물 시뮬레이터 비교 (부록 C).** RoboTwin 관측을 MidJourney API로 사실적 영상으로 바꾼 뒤 정책에 넣는 단계가 있습니다. 코드와 프롬프트, 변환 설정이 없어 재현할 수 없습니다. 외부 유료 API에 의존합니다.

**D9. 조건 주입 위치.** 논문 §3.2는 "언어 임베딩에 더한다"고 씁니다. 코드는 `original_ctx`(텍스트 문맥)에 더합니다(L364). 일치합니다. 학습 스크립트가 같은 `model_fn_wan_video` 경로를 타는지는 전체 호출 체인을 따라가지 않아 확인하지 못했습니다. 학습과 추론이 같은 조건 코드를 쓴다는 가정은 미검증입니다.

## 4. 재현 관점의 결론

공식 코드로 할 수 있는 것은 "이미 있는 행동 임베딩 파일을 넣어 LoRA로 학습하고 영상을 생성하는 것"입니다. 논문의 핵심 주장(정책 임베딩이 정책 성능을 드러낸다)을 검증하는 부분, 즉 추출기, 판정기, 지표, 실물 실험은 공개되지 않았습니다. 이 저장소는 **세계모델 학습 코드**이고 **평가 파이프라인이 아닙니다.**

## 5. 확인하지 않은 것

- 학습 데이터(1,400궤적)와 체크포인트 가중치는 받지 않았습니다. 가중치는 README 링크로만 확인 가능하고, 이번 작업에서는 다운로드하지 않았습니다.
- `diffsynth/` 전체(모델 구현, VAE, 텍스트 인코더)는 읽지 않았습니다. 행동 경로와 관련된 부분만 읽었습니다.
- 라이선스는 MIT(저자 수정 부분)와 Apache-2.0(원본 부분)이 섞여 있습니다. `LICENSE`는 MIT입니다. 재배포 시 두 고지가 모두 필요합니다.

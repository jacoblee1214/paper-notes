# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: `ustcwhy/BitVLA`(공식, 논문 초록이 링크). 커밋 `a8d62dfe398bfcce01283278cd716d4d6eae3c5a`(2026-11-18, "Update README.md"), 전체 30개 커밋 중 최신. 얕은 clone(`repo/`)을 정적으로만 읽었습니다. 아래 스크립트에서 `repo/`는 `git clone https://github.com/ustcwhy/BitVLA repo && git -C repo checkout a8d62dfe398bfcce01283278cd716d4d6eae3c5a`로 복원한 폴더를 뜻합니다.
- **실행, import, 설치를 하지 않았습니다.** 가중치와 데이터는 받지 않았고, 허브의 파일 크기와 config는 API 메타데이터로 확인했습니다.
- 저장소 구성: BitVLA 자체 코드는 `openvla-oft/bitvla/`(모델, 토크나이저, 데이터 변환, 약 700줄)와 `openvla-oft/vla-scripts/finetune_bitnet.py`(829줄), `openvla-oft/experiments/robot/{bitnet_utils.py, libero/run_libero_eval_bitnet.py}`, `openvla-oft/ft_script/*.sh`, `openvla-oft/convert_ckpt.py`입니다. 나머지는 가져온 코드입니다. `transformers/`(HF 포크, 버전 4.51.0.dev0, 양자화 모델은 `models/llava/modeling_bitnet.py`와 수정한 `modeling_llava.py`, `siglip/modeling_siglip.py`), `lmms-eval/`(VQA 평가 도구 포크).
- 읽은 파일: `README.md`, `vl_eval_setup.sh`, 위의 BitVLA 파일 전부, `lmms-eval/eval-dense-hf.sh`, `transformers/.../modeling_bitnet.py`의 양자화·어텐션 부분, `modeling_siglip.py`의 BitLinear, 허브 `lxsy/bitvla-bf16`의 config와 README, I2S GGUF 저장소의 파일 목록.

## 1. 설치 요구사항과 라이선스

- 설치(README): torch 2.5.0(CUDA 12.4), `pip install -e openvla-oft/`, `pip install -e transformers`(포크), LIBERO 설치, `pip install -e bitvla/`. VQA 평가는 NVIDIA 24.07 컨테이너를 쓰라고 안내합니다. 전부 CUDA 전제입니다.
- 라이선스: 루트 `LICENSE` MIT(Copyright 2025 Hongyu Wang). 가져온 코드는 각자의 라이선스(OpenVLA-OFT MIT, transformers Apache-2.0, lmms-eval는 MIT 안내가 있는 별도 LICENSE). GitHub의 license 값은 MIT입니다.
- README의 "오픈소스 계획": 논문, VLM, 평가 코드 완료, 미세조정 코드와 모델 완료, 사전학습 VLA 완료, **사전학습 코드는 계획 중(🧭)**.

## 2. 논문 ↔ 코드 대조 (D 항목)

판정: **일치** = 논문과 코드가 같음, **차이** = 값이나 구조가 다름, **논문에 없음**, **미공개** = 저장소에서 찾지 못함.

| 번호 | 항목 | 논문 | 코드 | 판정 | 영향 |
|---|---|---|---|---|---|
| D1 | LLM 어텐션 마스크 | v2 III-B c: "인과 마스크를 유지. 양방향으로 바꾸면 실로봇 성능이 떨어짐" | 학습 순전파와 추론 모두 `use_bi_attn = True`로 호출(`bitvla/model/bitvla_for_action_prediction.py` L166, L280). 이 플래그는 `modeling_bitnet.py`의 eager 어텐션에서만 마스크를 양방향으로 바꾸고(L419–473), `flash_attention_2` 경로는 `causal = self.is_causal`(항상 참)이라 무시함(L628–631). 클래스 사전에 `sdpa`는 없음(L703). 로더는 `attn_implementation`을 지정하지 않음 | **차이** (어느 경로로 돌리느냐에 따라 다름) | v2의 "인과" 서술과 공개 기본 경로(양방향 마스크, v1 방식)가 다를 수 있음. 공개 체크포인트가 어느 마스크로 학습·평가된 값인지 코드로 확정 불가 |
| D2 | LIBERO-Long 미세조정 스텝 | 표 V, IV-B: Long 100K 스텝, 나머지 10K | 네 스크립트 모두 `--max_steps 10001`(`ft_script/ft_bitvla_libero_long.sh` L15) | **차이** | Long의 학습 길이가 논문의 1/10. 스크립트로 학습하면 표 I의 Long(92.8) 조건과 다름 |
| D3 | 학습률 | 표 V: Spatial·Object·Goal은 {5e-5, 1e-4, 3e-4} 중 최고. Long은 LLM 4e-4, ViT 8e-5 | Spatial 1e-4, Object 1e-4, **Goal 2e-4(집합에 없음)**, Long 4e-4 + `--lr_vit 8e-5`. Spatial·Object·Goal은 `lr_vit` 없음(`finetune_bitnet.py`에서 `lr_vit > 0`일 때만 ViT 그룹 분리, L649–670) | **차이** (Goal), 일치 (Long, Spatial, Object) | Goal 설정이 논문의 탐색 집합 밖. 세 스위트의 ViT 학습률은 LLM과 같음(논문은 이를 명시하지 않음) |
| D4 | 행동 손실 | 식 (8): Σ_{k=0..h} ‖â − a‖₁ | `torch.nn.L1Loss()`(평균)를 원소 전체에 적용(`finetune_bitnet.py` L339). 합 대신 평균, 청크 8 x 차원 7 = 56 원소의 평균 | **차이** (스케일) | 평균은 합의 1/56배. AdamW에서는 크게 영향이 없으나 기울기 클리핑이 없고 감쇠와 결합할 때 상수 배 차이 |
| D5 | 행동 토크나이저 구간 수 | III-B c: 256구간(사전학습) | 미세조정 토크나이저 `bins = 252`(L580), 체크포인트 config는 `n_action_bins: 256` | **차이** | L1 회귀 경로에서는 행동 토큰 임베딩을 0으로 비우므로(D6) 학습 결과에 영향 없음. 토큰 수와 어휘 크기 설정만 다름 |
| D6 | 행동 토큰 입력 처리 | 병렬 디코딩(OFT 방식) | 행동 구간 토큰의 임베딩을 0으로 만들어 입력(`inputs_embeds * ~all_actions_mask`, L148–156 근처, 추론 `_regression_or_discrete_prediction`). 라벨 토큰은 손실에 쓰지 않고 마스크로만 사용 | **일치** (OFT와 같음) | 학습 때 정답 행동이 입력으로 새지 않음(좋음) |
| D7 | 학습률 스케줄과 옵티마이저 | 코사인 감쇠, 워밍업 375, AdamW(0.9, 0.999, 1e-8), 감쇠 0.01, 클리핑 없음(표 V) | `LambdaLR` 워밍업 375 + `CosineAnnealingLR(T_max = max_steps − warmup)`, `AdamW`(PyTorch 기본 betas, eps 1e-8, weight_decay 0.01), 클리핑 없음(L659–680). 사용하지 않는 `num_steps_before_decay`와 `MultiStepLR` import가 남아 있음 | **일치** | 기본값이 논문 값과 같음(명시 설정이 아니라 PyTorch 기본값) |
| D8 | 효율 측정 경로 | III-A, 그림 3, IV-C: BitBLAS 커널, A100, 73 ms | 저장소에 BitBLAS 커널 사용 없음(`bitblas` 검색 결과는 transformers 포크의 상류 파일 `quantization_config.py`의 로그 문구 한 줄뿐). `BitLinear`는 `WeightQuant`·`ActQuant`로 가중치와 활성을 bf16 안에서 양자화-역양자화(`modeling_bitnet.py` L85–118, L157–195)하고 `F.linear`를 부름. README는 "마스터 가중치와 온라인 양자화"라 안내하고 오프라인 1.58비트 추론은 bitnet.cpp를 권장 | **미공개** (측정 경로) | 그림 6의 73 ms를 공개 PyTorch 코드로 얻을 수 없음. README의 PyTorch eager 377.8 ms(RTX 3090, 2이미지)와 robot.cpp 39 ms(같은 GPU) 수치가 별도 근거 |
| D9 | 메모리 1.4 GB | 표 I, II, 그림 1 | 공개 체크포인트는 BF16 마스터 가중치(사전학습 6.30 GB, 미세조정 5.64 GB, 허브 메타데이터). `quantize_weights()`(`modeling_bitnet.py` L166–180)는 가중치를 2비트(`quantize_to_int2`, 4개를 1바이트)로 묶는데 1.58비트가 아님. 1.44 GB는 robot.cpp용 I2S GGUF의 파일 합계(1,440,224,736 byte)와 같음 | **일치** (GGUF 기준) / **차이** (PyTorch 기본 경로) | `03` V12. "1.4 GB"는 오프라인 2비트 묶음 가중치 크기 |
| D10 | 양자화 함수 | 식 (1)–(2): 가중치 텐서 단위 absmean, 활성 토큰 단위 absmax, 삼진 {−1, 0, 1}, INT8 [−128, 127] | `WeightQuant`: `s = 1/mean|W|`(텐서 전체), `round(W·s).clamp(−1, 1)/s`. `ActQuant`: `s = 127/max|x|`(마지막 축 = 토큰 단위), `round(x·s).clamp(−128, 127)/s`. 역전파는 기울기 그대로(STE). 역양자화는 αβ/127 크기 | **일치** (식 (4)의 인쇄된 127α/β와는 다름, `03` K5) | 정의와 구현이 같음. 식 (4)는 표기만 다름 |
| D11 | 양자화 대상 층 | III-A: 비전 인코더와 LLM의 모든 선형층(임베딩 제외). 커넥터와 행동 헤드는 전정밀도 | LLM의 q, k, v, o, gate, up, down은 `BitLinear`. ViT의 q, k, v, out, fc1, fc2는 config `vit_weight_bits = 1`, `vit_act_bits = 8`일 때 `BitLinear`. 서브 노름(`attn_sub_norm`, `ffn_sub_norm`)은 BitNet 구조. 커넥터, 행동 헤드, 고유감각 투영은 일반 선형층. lm_head와 임베딩은 양자화 안 함 | **일치** | 구성이 서술과 같음 |
| D12 | LIBERO 평가 프로토콜 | IV-B: OFT와 같은 프로토콜, 청크 K = 8, 이미지 2장 + 상태 | `num_trials_per_task = 50`(스위트당 500회), `seed = 7`, 대기 10스텝, 스위트별 최대 스텝 220/280/300/520, `center_crop`, 열린 루프 8스텝(`run_libero_eval_bitnet.py` L70–76, L105–132). 한 번의 평가에 시드 하나 | **일치** (OFT와 같음). 시드 수는 논문에 없음 | `03`의 가정(500회, 한 번의 평가)이 코드 기본값과 맞음. 시드 평균을 내려면 스크립트를 여러 번 돌려야 함 |
| D13 | 공개 LIBERO 체크포인트 | 표 I의 BitVLA 행(96.0)은 사전학습 모델의 미세조정 | README가 안내하는 `hongyuw/ft-bitvla-bitsiglipL-224px-libero_*-bf16` 네 개는 생성 2025-06-24–28, 갱신 2025-07-03. 사전학습 체크포인트(`lxsy/bitvla-bf16`)는 2026-02-24 생성. 미세조정 스크립트는 `/path/to/BitVLA-pretrain-200k`에서 시작 | **미확인** (README가 어느 체크포인트가 96.0에 해당하는지 쓰지 않음) | 2025-07 체크포인트는 사전학습보다 앞선 시점이라 사전학습 없는 모델(94.8)의 미세조정일 가능성이 있음. 확정하려면 저자 확인이 필요 |
| D14 | 사전학습과 VLM 학습 코드 | III-B, 표 IV, V 하이퍼파라미터, 식 (5)–(7) | 단계 I–III(커넥터 정렬, 지시 튜닝, Quantize-then-Distill)와 OXE 사전학습 코드는 저장소에 없음. L_aux 구현 없음. README는 사전학습 코드를 "계획"으로 표시. 공개된 것은 VQA 평가용 모델 코드와 LIBERO 미세조정·평가 | **미공개** | 표 III과 표 I의 사전학습 행은 코드로 재현할 수 없음. VLM 평가 스크립트는 `attn_implementation=eager`로 호출 |
| D15 | 실로봇 설정 | IV-C: Franka, 청크 10, H800 학습, A800 원격 추론, 50개 초기 위치 | 실로봇 코드, 데이터, 과제 설정 없음. `prismatic/vla/constants.py`에는 LIBERO(청크 8, 행동 7, 상태 8), ALOHA(25, 14, 14), Bridge(5, 7, 7)만 있음. OFT에서 남은 ALOHA와 Bridge 평가 스크립트만 있고 실로봇 평가는 없음 | **미공개** | 그림 4, 5의 재현 불가. 청크 10은 상수에 없음 |
| D16 | 비전 해상도와 토큰 수 | III-A: 224 해상도, 이미지당 256 토큰 | 데이터 변환과 추론에서 이미지당 `(H/patch) x (W/patch)` 토큰, config 패치 14, 224 → 256. `resize_resolution=(224,224)`(L718 주석에 해상도를 확인하라는 TODO가 남아 있음) | **일치** | 없음 |

## 3. 일치한 부분과 코드 품질 메모

- 모델 구조(BitNet 30층, GQA 20/5, ReLU², SubLN)와 양자화 정의(D10, D11)는 논문 서술과 맞습니다. 허브 config의 파라미터 수 합(3,147,925,664)은 공개 config에서 센 값과 약 0.01% 안에서 맞습니다(`03` V12).
- 미세조정의 배치는 GPU 4개 x 배치 2 x 누적 8 = 64로 표 V와 같고, 워밍업 375, 코사인, 감쇠 0.01은 기본값으로 맞습니다(D7).
- 학습 결과와 관계없는 부분: `finetune_bitnet.py`의 docstring이 "Fine-tunes OpenVLA via LoRA"로 남아 있고, 사용하지 않는 import와 옵션(`MultiStepLR`, `num_steps_before_decay`)이 남아 있습니다. 평가 스크립트는 OpenVLA-OFT의 것을 그대로 씁니다. 기능에는 영향이 없습니다.
- 평가의 `TASK_MAX_STEPS`와 초기 상태 파일 사용은 OFT와 같습니다.

## 4. 구현 가능성 판정 (이 맥, Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**논문의 수치 재현은 이 맥에서 불가능합니다. 방법의 핵심(W1.58A8 선형층, 층별 증류)은 소형 PyTorch로 구현해 검증할 수 있고, 공개 체크포인트의 추론은 MPS 이식 뒤에 판정할 수 있는 조건부입니다.**

| 작업 | 판정 | 근거와 노력 추정 |
|---|---|---|
| 표 재계산, 정적 코드 대조, 선형층 토이 | 완료 | `03`, 이 문서, `analysis/toy_quant.py` |
| W1.58A8 BitLinear + 층별 증류의 소형 구현(작은 ViT, 합성 또는 소형 데이터) | 가능 | torch 2.15가 이미 설치됨. BitLinear와 STE는 약 30줄(D10). 노력 약 1–2일(추정). 새 설치 없음 |
| 공개 LIBERO 체크포인트 한 청크 추론 | 조건부 | 체크포인트 5.64 GB(bf16)가 메모리에 들어가고 한 청크의 연산은 약 3.4 TFLOP(추정). 다만 CUDA 전제(`cuda:0`, 포크한 transformers 설치, `flash-attn` 선택)라 MPS·CPU 이식이 필요. 노력 3–5일(추정, 측정 전). 체크포인트 다운로드와 포크 설치 승인 필요 |
| LIBERO 평가 재현(스위트당 500회 x 4) | 불가 (현실적으로) | 한 청크 1–2초(추정)로 에피소드당 약 30–130초, 2,000회 전체 약 1.5–2일(추정)에 시뮬레이터(robosuite, 오프스크린 렌더링) 구성 비용. 소표본(과제 1개 x 몇 회)만 조건부 |
| LIBERO 미세조정 | 불가 | 2.8B 전체 파라미터 학습은 AdamW 상태를 포함해 약 34 GB 이상(추정)으로 메모리 초과. 스크립트는 CUDA·`torchrun` 전제 |
| VQA 평가(표 III) | 불가 | 모델 6.3 GB와 lmms-eval, 데이터셋 다운로드, 컨테이너 전제. 5개 벤치마크의 MPS 처리 시간 미측정 |
| 단계 I–III VLM 학습, 로봇 사전학습 | 불가 | 1,344 GPU시간(H800)과 5,376 GPU시간, 데이터(MammoTH-VL, OXE) 수 TB 규모. 코드도 미공개(D14) |
| 실로봇(그림 4, 5) | 불가 | Franka, 데이터, 과제 설정 미공개(D15) |

- 새 패키지 설치, 공식 코드 실행, 가중치 다운로드는 하지 않았고 위 조건부 항목은 승인 후에 판정합니다.
- 메모리가 아니라 **학습 계산과 장비**가 막는 항목이 대부분입니다. 가장 값싼 검증은 소형 QAT·증류 재구현입니다.

## 5. 저자에게 물어볼 것

1. 공개 미세조정 체크포인트와 표 I의 평가는 어느 어텐션 마스크(`eager` 양방향, `flash_attention_2` 인과)로 학습·평가했는가? (D1)
2. LIBERO-Long 미세조정은 100K 스텝(표 V)인가 스크립트의 10,001 스텝인가? Goal의 학습률 2e-4는 어떤 탐색에서 나왔는가? (D2, D3)
3. 표 I의 BitVLA 96.0행에 대응하는 미세조정 체크포인트가 공개되어 있는가? (D13)
4. 그림 4의 평균 막대(π0 약 53, 사전학습 없음 약 1.3)와 과제별 라벨 평균(63.3, 7.5)의 관계는? 시행 수와 π0·OFT의 학습 설정은? (`03` K4, §3)
5. 그림 6의 73 ms를 얻은 커널 경로는 공개 예정인가? robot.cpp의 수치와 어떻게 대응하는가? (D8)
6. "약 100만 샘플"은 궤적 수(957,161)인가? (`03` V11)

# LAPA: 공식 코드 대조 (논문 ↔ 저장소)

분석일 2026-11-21. 대상은 로컬 복제(얕은 복제, commit `46aca51d7faebcec02d7d323bbb3820c2df07bc6`, 2025-01-22)입니다. **정적 읽기만 했습니다.** 코드를 실행, import, 설치하지 않았고 가중치와 데이터를 받지 않았습니다. 대상 논문은 arXiv v2입니다.

차이 라벨은 D1–Dn입니다. 재계산 검사(V)와 구분합니다.

## 0. 대조 범위

| 논문 부분 | 저장소 위치 |
|---|---|
| LAQ 구조, NSVQ, 교차 어텐션, sg[p1] (§3.1, 부록 A, 식 1–4) | `laq/laq_model/latent_action_quantization.py`, `nsvq.py`, `laq/train_sthv2.py` |
| 잠재 사전학습과 손실 (§3.2) | `latent_pretraining/train.py`(modality `vision,text,delta`), `delta_llama.py` |
| 행동 미세조정과 이산화 (§3.3) | `scripts/finetune_real.sh`, `data/finetune_preprocess.py` |
| 학습 설정 (§4.6, 부록 B) | `scripts/latent_pretrain_openx.sh`, `scripts/finetune_*.sh`, `README.md` |
| SIMPLER 평가 (§4.1, 부록 B) | `SimplerEnv/simpler_env/main_inference_lapa.py`, `scripts/lapa_bridge.sh`, `data/simpler.jsonl` |

## 1. 일치하는 부분

| 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|
| 잠재 행동 차분 d = e2 − e1 (식 1) | §3.1, 부록 A | `input_data = input_data_last - input_data_first` (nsvq.py) | 일치 |
| 코드북 크기 8, 잠재 토큰 4 (2×2) | §3.1, §5.1 | `codebook_size=8`, `code_seq_len=4` (train_sthv2.py) | 일치 |
| NSVQ 양자화 (식 2) | §3.1 | `nsvq.py`: 정규화된 잡음 벡터로 VQ 오차 대체 | 일치 |
| 교차 어텐션으로 잠재 행동 주입 | §3.1 "cross attention" | `has_cross_attn=True`, 디코더 `context=actions` | 일치 |
| 현재 프레임 패치 임베딩에 stop-gradient (식 3) | §3.1, 부록 A | 학습 경로 `first_frame_tokens.detach()` (L268). `to_patch_emb_first_frame`의 출력 | 일치 |
| 재구성 손실 L2 (식 4) | 부록 A | `F.mse_loss` | 일치 |
| 코드북 교체, 초기 단계 | §3.1 | `replace_unused_codebooks`, 0–1000 스텝 중 특정 간격 | 일치 |
| 백본 LWM-Chat-1M 7B | §4.2 | `--load_llama_config='7b'`, README의 LWM 경로 | 일치 |
| 잠재 행동 어휘 8, 토큰 4 | §3.2 | `delta_vocab_size=8`, `n_tokens_per_delta=4` | 일치 |
| 행동 이산화 등분위(quantile) 256 bin | §3.3 | `pd.qcut(..., 256)` (`finetune_preprocess.py`) | 일치 (단 D9) |
| 미세조정 배치 128, 이미지 증강 | §4.4, 부록 C | `batch_size=128`, `img_aug=True` | 일치 |
| 백본 시각 토큰화 | §3.2 | 데이터 로더가 `vqgan` 체크포인트로 프레임을 토큰화 (`latent_pretraining/data.py` `encode_images`) | 일치 (VQGAN 학습 설정은 논문에 없음) |

## 2. 차이 (D)

### D1. 사전학습 배치와 스텝 (논문과 코드와 README가 다름)

- 논문 §4.6: **8 × H100, 34시간, 배치 128**, 총 272 H100 시간.
- `scripts/latent_pretrain_openx.sh`: `batch_size=256`, `total_steps=70000`.
- README: "70K steps with a batch size of 256 ... enough to get decent performance". 두 값은 같은 저장소 안에서도 다릅니다.
- 영향: 논문의 272 H100 시간 계산이 어떤 설정을 따르는지 알 수 없습니다. 효율 비교(`03` V7)의 기준이 불확실합니다.

### D2. 사전학습 손실에 텍스트 보조 항 (논문에 없음)

- `latent_pretraining/train.py` L313: `loss = 0.99 * delta_loss + 0.01 * text_loss`. `vision_loss`는 계산만 하고 손실에는 넣지 않습니다(메트릭으로만 기록).
- 논문 §3.2는 잠재 행동 예측 목적만 씁니다. 텍스트(명령어) 손실 가중치 0.01은 본문과 부록 어디에도 없습니다.
- 영향: 작은 가중치라 효과는 작을 것으로 추정합니다. 그러나 "순수한 잠재 행동 행동 복제"라는 설명과 다릅니다. 측정은 하지 않았습니다.

### D3. 잠재 행동 헤드 구조 (표현 차이)

- 논문 §3.2, §3.3: "a separate latent action head (a single **MLP** layer)".
- 코드 `delta_llama.py` L401: `nn.Dense(delta_vocab_size, use_bias=False)`. **단일 선형층**입니다. 은닉층이 없습니다.
- 영향: 용어 차이입니다. 파라미터 수와 표현력의 차이는 작지만 "MLP"는 틀린 표현입니다.

### D4. 시각 인코더 동결 (구현 확인 불가)

- 논문 §3.2: "freeze the vision encoder and unfreeze all of the LM parameters".
- `latent_pretraining/llama.py` L351: `get_frozen_param_exclusions(freeze_base)`가 `freeze_base`일 때 `("vte", "vision_head")`를 반환합니다. 저장소 전체에서 **이 함수를 호출하는 곳을 찾지 못했습니다** (`grep -rn "get_frozen_param_exclusions" --include='*.py'`는 정의 한 곳만 나옵니다).
- 즉 학습 경로에서 동결이 실제로 적용되는지 정적으로는 확인할 수 없습니다. 논문 설명과 맞는지는 학습 로그 또는 옵티마이저 설정이 있어야 판정합니다.

### D5. LAQ 학습 코드 범위 (미공개)

- 저장소의 LAQ 학습은 `laq/train_sthv2.py` 한 개뿐입니다. README도 "data loader is based on something-something v2"라고 씁니다.
- 논문 §4는 LAQ를 **OpenX와 Bridge** 등 로봇 영상으로도 학습했다고 씁니다(§3, "same pretraining dataset"). 그 학습 코드와 H = 3(Bridge), H ≈ 2.4 s(사람 영상)의 프레임 변환 코드는 **없습니다**.
- 공개 가중치 `laq_openx.pt`(README 2024-11-22 공지)는 Hugging Face에 있습니다. 추론에만 쓸 수 있습니다.

### D6. LAQ 학습 설정 (논문에 수치가 없음)

- `train_sthv2.py`: 차원 1024, quant_dim 32, 공간 8층, 시간 8층, 패치 32, 이미지 256, 배치 64, lr 1e-4, 100,005 스텝, offset 30 프레임.
- 논문 부록 A는 구조를 설명하지만 학습 하이퍼파라미터와 offset 프레임 수를 주지 않습니다(부록 F는 0.6초, 2.4초만 씀). 따라서 대조할 값이 없습니다.
- 사람 영상 offset 30 프레임이 SSv2의 몇 fps에서 2.4초인지 저장소에서 확인할 수 없습니다.

### D7. 창 크기 H (로봇 영상)

- 논문 부록 F: Bridge 5 Hz에서 H = 3 (0.6 초). 저장소의 로봇 LAQ 설정이 없어서 대조 불가 (D5와 같은 원인).

### D8. CUDA 하드코딩과 설치 제약 (구현 장벽)

- `nsvq.py`: `NSVQ(..., device='cuda')`와 `train_sthv2.py`의 `.cuda()`.
- `requirements.txt`: `jax[cuda12]==0.4.23`, `flax==0.7.0`, `transformers==4.29.2`, `git+https://github.com/lhao499/tux.git`(외부 의존).
- 영향: 이 맥에서 실행하려면 `device` 인자 교체와 JAX CPU 또는 Metal 빌드가 필요합니다. 핀 고정 버전이 맥 Python 3.10 환경에서 맞는지 설치해 보아야 합니다. 설치는 하지 않았습니다.

### D9. 이산화: 그리퍼는 분위수 빈이 아님 (논문에 없음)

- 논문 §3.3: 행동 **각 차원**을 동일 개수 빈으로 이산화합니다.
- `data/finetune_preprocess.py`: 여섯 차원은 `pd.qcut(…, 256)`으로 빈을 만들지만, 일곱 번째(그리퍼)는 `int(action[6])`로 **원래 값을 정수로 자릅니다**. 이후 `total_bin[-1] = [-0.5, 0.5, 1.5]`로 두 빈을 쓴 CSV를 저장합니다.
- 영향: 그리퍼는 2값(0/1 계열)으로 가정합니다. 논문은 이 예외를 밝히지 않습니다. 그리퍼 값이 연속이면 결과가 다릅니다.

### D10. 미세조정 단계의 헤드 재초기화 (확인 불가)

- 논문 §3.3, 각주 2: 잠재 헤드를 버리고 **새 행동 헤드**를 붙입니다(재초기화).
- 코드 미세조정(`finetune_real.sh`)은 `params::<체크포인트>`로 파라미터를 읽습니다. 행동 헤드가 체크포인트에 있는지 없는지는 가중치 키를 봐야 알 수 있습니다. 가중치는 받지 않았습니다.
- 따라서 "재초기화"가 실제로 일어나는지는 **확인 불가**입니다.

### D11. 사전학습 데이터와 평가 자산의 공개 범위

- 사전학습 jsonl(`latent_action_pretraining_openx.jsonl`)과 `laq_openx.pt`, `params`, `vqgan`, `tokenizer.model`은 Hugging Face에 있습니다(모델 카드, 용량은 받지 않아 확인 못함).
- 공개되지 않은 것: Language Table 평가 스택(외부 저장소 참조), 실로봇 제어(`polymetis` 보관됨), 실로봇 데이터, Bridge LAQ 학습, OpenVLA 기준선 코드(외부 참조), 기준선 UniPi와 VPT 코드.
- `data/simpler.jsonl`(2,482행)은 SIMPLER 미세조정 자료로 포함되어 있습니다. 논문은 "100 궤적"이라고만 씁니다. 행 수와 궤적 수는 같은지 확인하지 않았습니다.

### D12. 저작권과 라이선스

- 공식 저장소 `LICENSE`: MIT, "Copyright (c) 2024 Seonghyeon Ye and Joel Jang"(GitHub API `mit`와 일치).
- 백본 LWM은 Apache-2.0입니다. 저장소가 LWM 코드를 기반으로 한다고 README가 밝힙니다. 파생 코드를 쓸 때는 Apache-2.0 고지가 필요합니다.
- `SimplerEnv/`는 MIT(`simpler-env`), `laq_model/nsvq.py`는 원저 MIT(`MHVali`) 출처를 파일 머리 주석에 밝혔습니다.
- Hugging Face 모델은 MIT입니다(`latent-action-pretraining/LAPA-7B-openx`, 모델 카드 메타데이터).

## 3. 대조 결과 요약

| 구분 | 개수 | 항목 |
|---|---|---|
| 일치 | 12 | §1 표 |
| 차이 (코드와 논문 또는 README가 다름) | 4 | D1, D2, D3, D9 |
| 확인 불가 (정적 읽기로는 판정 못 함, 로그나 가중치 필요) | 2 | D4, D10 |
| 저장소 범위 밖 또는 미공개 | 3 | D5, D7, D11 |
| 논문에 수치가 없어 대조 불가 | 1 | D6 |
| 구현 장벽 | 1 | D8 |
| 라이선스 정보 | 1 | D12 |

- 논문 핵심 방법(LAQ의 교차 어텐션, sg[p1], NSVQ, 2단계 학습)은 **코드와 일치**합니다. 차이는 학습 설정(배치, 손실 보조항, 헤드, 이산화 예외)과 공개 범위에 몰려 있습니다.
- 학습 설정 차이(D1, D2)는 논문 주장(효율, 성능) 해석에 영향을 줍니다. 특히 효율 비교의 기준이 흔들립니다.

## 4. 구현 가능성 판정

이 맥(Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음) 기준입니다. 설치와 실행은 하지 않았습니다. 판정은 코드 구조와 요구 자원에서 나온 것입니다.

| 작업 | 필요 자원 | 이 맥 판정 | 근거 |
|---|---|---|---|
| 논문 수치 재현 (LAPA 사전학습 + 미세조정 + 실로봇) | 8 × H100 34시간 (272 H100-h), 실로봇 Franka, Language Table/SIMPLER 시뮬 | **불가** | 자원 규모 (논문 §4.6). 실로봇 장비 없음 |
| LAQ 학습 (SSv2) | 100,005 스텝, 배치 64, `.cuda()` 하드코딩 | **불가** | GPU 필요. D8 |
| LAPA-7B 추론 (`params` 체크포인트) | 7B 가중치 (bf16 약 14 GB 이상), JAX CUDA 핀 | **불가 (실용)** | 메모리와 JAX Metal 지원 불확실. 체크포인트 크기는 받아서 확인 필요 |
| `laq_openx.pt`로 잠재 행동 추출 (추론만) | 300M급 LAQ 가중치, `device='cuda'` 패치 | **조건부** | D8 패치 후 MPS 실행 가능성. 설치, 가중치 다운로드, 공식 코드 실행 승인 필요 |
| SIMPLER 평가 (Bridge) | 시뮬 환경(SimplerEnv 설치 필요), 렌더링, 학습된 정책 | **불가 (이 맥, 설치 전 판단)** | 설치 가이드를 읽지 않아 확정은 못 함. 렌더링 요건이 맥에서 맞는지 확인 필요. Linux 쪽에서 재검토 |
| 사람 영상 LAQ 설계 검증 (토이) | CPU, numpy 또는 torch | **가능** | 논문 주장의 축소판. 2분 이내 목표 |

**노력 추정 (측정 전):**
- CPU 토이 (잠재 이산 행동 + NSVQ의 코드북 사용률 검사): 1–2일.
- 공식 LAQ 추론 (Linux GPU 기준): 2–3일 (설치 포함). 맥은 D8 패치와 JAX/MPS 확인 후 판단.
- 논문 전체 재현: 자원상 불가. 추정도 하지 않습니다.

## 5. 재현 전 확인 목록

1. D1의 배치와 스텝을 메인테이너 설정으로 확정할 방법(이슈 등)을 찾습니다. 저장소에는 없습니다.
2. D4 동결 여부는 학습 로그나 체크포인트 키로만 확인할 수 있습니다.
3. D10 헤드 재초기화는 `params` 키 목록으로 확인합니다. 가중치 다운로드 승인이 필요합니다.

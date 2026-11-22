# 04. 공식 코드 대조와 구현 가능성: Being-H0.5 (arXiv:2601.12993)

분석일 2026-11-22. 공식 저장소 `BeingBeyond/Being-H`(Apache-2.0)를 로컬에 얕게 복제(커밋 `27ec67b84ac6c8e55f95408205e1ad7ba8bec253`, 2026-09-27)해 **정적으로만** 읽었습니다. 설치, import, 실행은 하지 않았습니다. 복제한 내용은 신뢰하지 않는 자료로 다뤘습니다. 가중치와 데이터는 받지 않았고, Hugging Face의 모델 카드 메타데이터와 `config.json` 두 개(KB 단위)만 읽었습니다. H0.5 코드는 `Being-H05/`(파이썬 약 2.3만 줄, 교육용 복사본 `tutorials/Being-H-EDU/` 포함)입니다. 논문은 v1(2026-01-19)이고, 코드는 이후에 바뀌었습니다(D15).

## 1. 공개 범위 한눈에

| 논문의 구성 요소 | 공개 코드 | 상태 |
|---|---|---|
| 통합 상태·행동 공간 (§5.1.1) | `configs/data_config.py`, `docs/unified_action_space.md`, `beingvla.py` (200차원) | 있음 |
| MoT(이해 + 행동 전문가), 정류 흐름 (§5.1) | `BeingH/model/beingvla.py`, `llm/qwen3_navit.py` | 있음 (밀집 단일 행동 전문가) |
| **Mixture of Flow 라우팅 (§5.1.2)** | 없음 | **없음 (D1)** |
| **사람 모션 데이터, 이산 모션 토큰, L_MASK, 하이브리드 목적 (§5.2)** | `hmg_dataset.py`가 라이선스 두 줄만 | **없음 (D2)** |
| **ESA 슬롯별 어댑터 (§5.3.1)** | 없음 | **없음 (D3)** |
| MPG (§5.3.2, §6.1) | `layers.py`, `beingvla.py` | 있음 |
| UAC 학습 시뮬 지연 (§5.3.3) | `beingvla.py` | 부분 있음 (전역 지연, 꺼짐) (D4) |
| **UAC 배포: 두 스레드, 링 버퍼, 하드 스티칭 (§6.2, §6.3)** | 없음 (서버가 전체 청크 반환) | **없음 (D4)** |
| 사전학습 스크립트와 혼합 구성 | 없음. README TODO "Complete pretraining scripts and documentation" 미완료 | **없음 (D10)** |
| LIBERO·RoboCasa 후학습·평가 | `scripts/train/`, `scripts/eval/`, `configs/posttrain/` | 있음 |
| 실로봇 5종 설정·데이터 변환·배포 | 없음 (SO101 교육 예제만 `tutorials/`) | 없음 |

## 2. 논문 ↔ 코드 차이 (D)

심각도: **높음** = 논문의 핵심 주장 구성 요소가 코드에 없음, **중간** = 설정이 논문과 다르거나 서술이 맞지 않음, **낮음** = 사소, **일치** = 논문과 맞는 것을 확인.

| ID | 심각도 | 논문 | 코드 | 위치 |
|---|---|---|---|---|
| D1 | 높음 | §5.1.2: 행동 전문가는 공유 Foundation 층 + 라우터가 상위 K를 고르는 Specialized 전문가. 활성 파라미터만 추론에 사용 | **라우터, 상위 K, 전문가 집합 코드가 없음.** 행동 전문가는 층마다 하나의 밀집 MLP(`mlp_mot_gen`)와 어텐션 투영을 가진 MoT 한 갈래(hidden 1024, 28층, Qwen3-0.6B 모양). 문서는 "Mixture of Action Expert"라고 부름. 체크포인트 config도 `use_moe: None`, 전문가 hidden 1024, 28층. 그림 11의 "pt + MoE" 곡선은 공개 코드로 재현할 수 없음 | `llm/qwen3_navit.py:296-315`, `docs/training.md` "Action Expert Initialization", HF config |
| D2 | 높음 | §5.2.2–5.2.3: 모션 생성·설명·연속 과제, 이산 모션 토큰 + 마스크 토큰 예측 L_MASK(식 8), 어텐션 게이트 G(식 9), 위치 정렬(식 10), L_act = λ1 L_FM + λ2 L_MASK, L = λ_text L_text + λ_act L_act | 사람 모션 데이터셋 모듈 `hmg_dataset.py`는 라이선스 헤더 두 줄뿐. **L_MASK, G, λ 가중치 코드가 없고**, 학습 손실은 `loss = action_loss + und_loss`(가중치 1:1). 모션 토큰 관련 흔적은 `set_optimize_motion`과 사전학습 체크포인트 config의 `action_context_mode: discrete`, `max_motion_chunk_length: 256`뿐 | `dataset/datasets/hmg_dataset.py`, `train/train.py:1111`, `beingvla.py:298-304` |
| D3 | 높음 | §5.3.1, 식 11–12: 슬롯 K개의 어댑터 뱅크 W_ESA ∈ R^{K×d_out×d_in}, 활성 슬롯 I_e만 갱신 | **슬롯별 어댑터가 없음.** 상태·행동 인코더와 디코더는 전 임베디먼트가 공유하는 `SimpleMLP`/`ActionEncoder`이고, 비활성 슬롯은 손실에서 마스킹될 뿐(`mask_invalid_action_dims`). 배치에 실리는 `embodiment_ids`는 모델 안에서 쓰이지 않음. GR00T 계열의 `CategorySpecificLinear/MLP/ActionEncoder` 클래스는 정의만 있고 사용되지 않음 | `beingvla.py:192-215,333,633`, `layers.py:210-285`, `dataset/base_dataset.py:361-435` |
| D4 | 높음 | §5.3.3, 식 17–19: 임베디먼트별 지연 분포 π(e)와 최대 지연 d_max(e), 제어 주기와 지연 예산으로 설정. §6.2–6.3: 두 스레드 링 버퍼, 하드 프리픽스 고정, 하드 스티칭 | 학습 RTC는 있으나 **지연이 하나의 전역값** `simulated_delay`이고 분포는 가중치 exp(max_delay−1−d)(작은 지연 우대) 또는 균등 중 하나. 임베디먼트별 분포는 없음. **모든 공개 후학습 레시피(LIBERO, 교차 임베디먼트, SO101)와 LIBERO 체크포인트 config가 `use_training_time_rtc False`, `simulated_delay 0`**이고 평가 스크립트는 `--no-enable-rtc`. 두 스레드 배포 코드(`threading`, 링 버퍼, 뮤텍스)가 저장소에 없음. 식 18의 타임스텝(prefix 1, 나머지 t_base)과 postfix 전용 손실은 논문과 일치 | `beingvla.py:141-146,367-420,594-606,735-770`, `scripts/train/*.sh`, `scripts/eval/*.sh`, `inference/beingh_policy.py:386-396` |
| D5 | 중간 | §5.3.2 식 13–15: H̃ = H + λ g W E_obs(H) + λ b, g = exp(−D/τ), D는 슬라이스드 바서슈타인 거리, 앵커 Z̄는 노이즈 없는 행동 임베딩의 평균(학습: 정답, 추론: 이전 반복), g에 stop-gradient | 구조는 식 13과 일치(게이트를 투영 앞에 곱함, 정답 앵커, stop-gradient). 코드 주석은 "논문의 gate × residual이 아니라 gate × 입력"이라고 쓰는데 이는 식 13의 설명과 같은 방식이라 주석이 논문(DiG-Flow의 출력 게이팅 대비)과 오해를 부름. 세부: (a) E_obs가 학습 가능한 Linear + 편향이라 그 편향은 게이트됨, 최종 투영 편향만 비게이트. (b) 앵커를 관측 길이로 방송하므로 SWD가 **평균 제곱 거리**와 같음(`03` V13). (c) 게이트는 **패킹된 시퀀스 전체에 하나의 스칼라**이고 앵커는 배치 평균이라 샘플·임베디먼트별 신뢰도가 아님. (d) 투영 방향은 seed가 없으면 호출마다 무작위. (e) 정제 N_ref 기본 1(논문 1–3). (f) τ가 세 곳에서 다름: 클래스 기본 1.0, config 기본 2.0, 학습 스크립트 1.0. 공개 LIBERO 체크포인트는 2.0, λ 0.1, M 32 | `layers.py:402-493,495-690`, `beingvla.py:129-140,449-505,776-900` |
| D6 | 중간 | §5.1.1: 사람 손의 손목은 EEF 슬롯, **손가락은 "fine-manipulation" 예약 슬롯**에 사상해 로봇과 의미 정렬. 정규화 안 함, EEF 상대 변위, 축-각, 관절 절대 라디안 | 차원 200(논문에 없음). EEF 위치·회전(상대, 축-각, 슬롯 0–6), 그리퍼 1, 로봇 손 관절 6(슬롯 20–26)은 일치. LIBERO·RoboCasa EEF는 정규화 없음, 그리퍼만 `binary`. **사람 MANO는 로봇 손 슬롯(20–44)이 아니라 별도 슬롯 90–200에 있음**(β 10×2, θ 45×2). 로봇 손가락과 사람 손가락이 같은 슬롯을 쓰지 않으므로, 손가락 수준의 슬롯 공유 전이는 구조적으로 없고 네트워크가 사상을 학습해야 함. 문서의 "통계가 임베디먼트 전체에서 차원별로 정규화됨"과 논문의 "통계 정규화 안 함"은 서로 다른 서술. 슬롯 표에 그리퍼 범위가 겹침(18–19, 19–20) | `docs/unified_action_space.md`, `configs/data_config.py:129-285`, `beingvla.py:192-193` |
| D7 | 일치 | §5.1.1: 임베디먼트별 입출력 헤드를 두지 않고 하나의 공간을 공유 | 상태 인코더, 행동 인코더, 행동 디코더가 모두 단일 `SimpleMLP`/`ActionEncoder`(200차원 입출력)로 공유됨 | `beingvla.py:196-215` |
| D8 | 중간 | §7.2.1: 행동 청크 크기 8, 유효 배치 128, GPU당 7,680 토큰 | 학습 스크립트 `ACTION_CHUNK_LENGTH=16`, 사전학습 체크포인트 최대 16, **공개 LIBERO 체크포인트 config는 15**, 평가가 앞 8개만 실행(`EVAL_CHUNK_SIZE=8`). 토큰 패킹 `expected 8192`/`max 8704`, 유효 배치를 직접 지정하는 인자가 없음(토큰 단위). 논문의 "청크 8"은 실행 길이로 읽어야 함 | `scripts/train/train_libero_example.sh`, `scripts/eval/eval-libero.sh`, HF config |
| D9 | 중간 | §7.2.1: 전문가 45k 스텝, 4×A800, 범용은 약 2배 스텝(90k) | 예제 스크립트는 **60k 스텝**(LIBERO), **120k**(교차), 4 GPU, LR 1e-4, 가중치 감쇠 1e-5, 워밍업 5%, 코사인. 저장 규칙이 `step > 25,000 && step % 10,000 == 0`이라 **45k는 저장 격자에 없음**(30k, 40k, 50k, 60k). 논문의 체크포인트 선택 규칙은 본문에 없음 (`03` V12) | `scripts/train/*.sh`, `train/train.py:1192` |
| D10 | 높음 | §1: "모델 가중치, 학습 파이프라인, 시뮬 스크립트에 더해 실세계 배포 인프라와 1,000 GPU-hour 사전학습 레시피를 공개" | README TODO에서 "Complete pretraining scripts and documentation", "Out-of-the-box real robot pretrained checkpoints" 미완료. UniHand 혼합 비율, 사람 모션 로더, 사전학습 하이퍼파라미터가 없음. 사전학습 체크포인트 하나(2B, "preview")와 후학습 체크포인트 3개만 있음. 데이터 로더는 로봇(`vla_dataset.py`)과 비전-언어(`vlm_dataset.py`)만 있음 | `README.md` TODO, `dataset/datasets/` |
| D11 | 중간 | §7.2.1–7.2.2: LIBERO 스위트당 500회, RoboCasa 과제당 50회, 5개 held-out 장면, unseen 물체·주방 스타일 | 스위트당 `NUM_TRIALS=50` × 10과제 = 500은 일치. RoboCasa도 `NUM_TRIALS=50`(데이터클래스 기본은 10). 평가 시드 단일(`EVAL_SEED=41`, 서버 42). **held-out 지정이 코드에 없음**: `create_env`에 `obj_instance_split=None`, `layout_ids=None`, `layout_and_style_ids=None`을 넘겨 RoboCasa 기본값에 맡김. 24개 과제를 PnP 8/문·서랍 6/기타 10으로 나누는 구성은 논문 가중 평균과 일치(`03` V4). LIBERO 학습은 4개 스위트를 한 체크포인트로 학습(`libero_all`)하며 "전문가"의 정의(벤치마크 전용)와 일치. 체크포인트·시드 선택 규칙은 없음 | `scripts/eval/*.sh`, `benchmark/robocasa/run_robocasa_eval_fast.py:29-80,131`, `configs/posttrain/` |
| D12 | 중간 | §7.2: "2B 백본", 추론 K < 10 | 공개 체크포인트는 **28.1억 파라미터**(safetensors 합계 2,807,618,760, 파일 11.2 GB fp32): 이해 쪽 InternVL3.5-2B(Qwen3, hidden 2048, 28층) + 행동 전문가(hidden 1024, 28층). 노이즈 제거 단계 기본 4. 사전학습 체크포인트의 `mpg_lambda`는 **0.0**(사전학습에서 MPG 비활성, 논문의 "후학습 기법" 서술과 일치). LIBERO 체크포인트: `use_mpg true`, λ 0.1, τ 2.0, 반복 1, RTC 꺼짐 | HF `config.json` 2개 |
| D13 | 중간 | §6.1: 비전·언어 프리픽스는 KV 캐시로 재사용해 효율을 높임, "K < 10" | 노이즈 제거 반복마다 `forward_train`을 전체 시퀀스(이해 토큰 포함)에 다시 돌림. `past_key_values`, `use_cache`, KV 캐시 사용이 `beingvla.py`에 없음. MPG 정제(기준 1 + 정제 1)면 단계 4 × 2 = 전체 순전파 8번 | `beingvla.py:790-930` |
| D14 | 낮음 | §5.2.2 식 7: x_t = (1−t)x_0 + t a, 속도 목표 a − x_0, t ~ p(t) | 보간과 속도 목표는 식 7과 일치(정류 흐름, 속도 예측 MSE). 시간 샘플링은 GR00T식 Beta(1.5, 1)로 t = (0.999 − s)/0.999, 1000 버킷 이산화(논문에 p(t) 미기재). 손실은 활성 슬롯 마스크 합으로 정규화 | `beingvla.py:307-309,433-447,574-606` |
| D15 | 중간 | (논문은 v1 하나, 2026-01) | 논문 이후 두 개의 학습-추론 일관성 수정이 들어감: 2026-08-04 "Fix invalid unified action dimension sampling"(비활성 슬롯의 노이즈 경로 처리, `mask_invalid_action_dims`), 2026-08-25 "Align inference image token packing with training"(뷰마다 시작·끝 토큰을 따로 둠). 1월에 공개된 체크포인트와 논문 표를 만든 추론 코드가 이 수정 이전 코드였다면 보고 수치와 현재 코드의 동작이 다를 수 있음 | 커밋 `e94af37`, `e0d3f76`(GitHub API) |
| D16 | 낮음 | 해당 없음 | 코드 기본값이 논문 경로가 아님: `use_flow_matching=False`, `loss_func="l1"`(이전 H0 방식)가 클래스 기본이며 학습 스크립트가 `--use_flow_matching True`로 켬. `mask_invalid_action_dims`의 기본값이 두 곳에서 다름(config True, 모델 getattr False) | `beingvla.py:75,83,128,177` |

일치로 확인한 항목: 공유 입출력 헤드(D7), 정류 흐름 식(D14), 식 18의 지연 타임스텝과 postfix 손실, MPG의 입력 게이팅 구조와 stop-gradient, 청크 평가 실행 길이 8, LIBERO·RoboCasa 시도 수, 24개 RoboCasa 과제 분류.

## 3. 해석

- **구현은 논문의 "후학습 쪽 절반"입니다.** 공개 코드가 재현하는 것은 InternVL3.5-2B + 0.6B급 흐름 행동 전문가를 LIBERO와 RoboCasa에 후학습하는 경로이고, 논문의 이름 붙은 기여 가운데 **MoF, ESA, UAC의 임베디먼트별 부분, 이산 모션 목적(L_MASK)은 공개본에 없습니다**(D1–D4). 따라서 LIBERO 98.9%와 RoboCasa 53.9%는 "MoF 없는 밀집 전문가 + MPG(λ 0.1) + RTC 끔"으로 얻은 값일 가능성이 큽니다(체크포인트 config 근거). 논문이 이 수치를 MoF, UAC의 효과로 읽게 하는지는 표 4·5가 구성요소별로 분해되지 않아 판단할 수 없습니다.
- **MPG는 구현되어 논문의 식 13과 맞지만**, 게이트가 평균 제곱 거리의 단조 함수이고 LayerNorm이 크기를 지웁니다(`03` V13). "분포 이동 방어"는 이 구현에서 크기 변화에는 작동하지 않고 방향 정렬만 봅니다.
- **실로봇 결과(그림 10, 13)의 재현 경로는 없습니다.** 로봇별 설정, 데이터 변환, 배포 코드, 실로봇 후학습 데이터 대부분이 공개되지 않았습니다(PND Adam-U 후학습 데이터 9개는 HF에 있음, `02` §2).
- **정직한 한계:** 이 대조는 정적 읽기입니다. 체크포인트 가중치를 받지 않았으므로 config 값 외에 가중치 구조(예: 라우터 파라미터 유무)는 확인하지 않았습니다. 다만 파라미터 합계 28.1억은 D1의 밀집 구조와 크기가 모순되지 않습니다(층별 대조는 하지 않음).

## 4. 구현할 수 있는지 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 논문 전체의 재현은 불가능합니다. 공개 후학습 경로의 소규모 부분과 수치 검증은 이 맥에서 조건부로 가능합니다. 이 분석은 재현을 하지 않았습니다.**

| 작업 | 이 맥에서 | 필요 자원 | 근거 |
|---|---|---|---|
| 사전학습 (UniHand-2.0, 35,000 h) | **불가** | 데이터 수십 TB, 다수 GPU. 사전학습 스크립트와 사람 모션 코드가 공개되지 않음 | D2, D10. 논문은 사전학습 GPU 규모를 밝히지 않고 "1,000 GPU-hour 레시피"를 약속(§1) |
| LIBERO·RoboCasa 후학습 재현(45k–60k 스텝, 4 GPU) | **불가** | CUDA GPU 4장 규모 | 공식 예제 `NUM_GPUS=4`, 60k 스텝. 코드가 flash-attn, `flex_attention`, `torch.cuda` 호출(19곳), DeepSpeed, decord 등 CUDA 전제 의존성을 가짐(`requirements.txt`, `beingvla.py:11,17`) |
| 시뮬 평가(LIBERO, RoboCasa) | **조건부, 사실상 어려움** | MuJoCo 기반 평가 + 모델 추론. RoboCasa 자산 약 5 GB, 학습 데이터 약 4.9 GB(CC-BY-4.0) | 평가 환경은 EGL 헤드리스 렌더링을 전제(`docs/evaluation.md`). 로봇 시뮬 렌더링 비용이 큼. 설치·다운로드는 승인 필요 |
| 2B 체크포인트 CPU/MPS 추론(1청크) | **조건부** | 가중치 11.2 GB(fp32, bf16이면 약 5.6 GB), 24 GiB에 들어감. flash-attn/flex_attention 경로를 MPS·CPU용으로 포팅해야 함 | 정적 읽기만 했고 포팅 가능성은 검증하지 않음. 청크당 순전파 8번(D13). 한 청크 수 초~수십 초 규모로 **추정**(측정 아님) |
| MPG 게이트 수치 검증(장난감) | **완료(CPU)** | numpy, 수 초 | `analysis/recompute.py` V13 |
| 표 재계산, 통계 검정 | **완료(CPU)** | numpy, scipy | `03` |
| 실로봇 5종 | **불가** | Adam-U, G1+O6, FR3+Inspire, D1, SO-101 실물과 제어 인프라 | 장비 없음. 로봇 설정·배포 코드 미공개 |
| SO-101 교육 예제 (`Being-H-EDU`) | **불가에 가까움** | SO-101 팔 실물, GPU 후학습 | 교육 예제도 같은 torchrun 기반 학습 코드를 쓰므로 CUDA 전제로 보임(확인하지 않음). 6 DoF 저가 팔이라 장비 비용은 낮음 |
| MPS 위 소형 비교 구현 (자체 작성) | **가능** | torch(MPS), 합성 데이터 | 통합 행동 공간 슬롯 패딩, 정류 흐름 소형 모델, MPG 입력 게이팅을 자체 작성하면 CPU/MPS에서 장난감 수준 검증 가능. 공식 코드 사용 아님 |

**공수 추정(측정 전)**
- 후학습 코드 CPU 추론 포팅: 2–4주(flex_attention·flash-attn 대체, 체크포인트 로딩 검증, 속도 측정).
- 통합 행동 공간과 MPG의 자체 작성 소형 재구현: 1–2주(MPS, 합성 또는 소규모 공개 데이터).
- 논문 사전학습 재현: 해당 없음(코드, 데이터, 규모 모두 부족).

노력 추정은 어림값이며 이 분석에서 측정하지 않았습니다.

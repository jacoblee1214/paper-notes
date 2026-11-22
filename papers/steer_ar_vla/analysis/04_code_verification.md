# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 대상: `jasontchan/openpi-fast-steering` (저자 개인 계정의 openpi 포크). 논문과 프로젝트 페이지는 이 저장소를 링크하지 않지만, README 제목 "Autoregressive FAST Token Editing for Inference-Time Steering", 첫 쪽 변수명(`arrows`, `fast_arrows`), 저자의 분석 노트북이 논문 실험과 일치해 **저자 코드로 봅니다**(공식 여부는 저자 확인 필요).
- 얕은 복제(`--depth 50`) 커밋 `8fb0a577ba3ec815338696d7229a028bfffba2bc`(2026-05-08, 메시지 "analyses"). 업스트림 기준 커밋은 `fdc03f5`(2026-03-29)이고 `git diff fdc03f5 HEAD`로 포크의 변경만 읽었습니다. 변경 32개 파일, +7,135/−355줄입니다. 그중 대부분은 같은 파일의 스크래치 복사본(`*_original.py`, `*_steer_v1.py`, `policy copy.py` 등)입니다.
- **정적 읽기만 했습니다.** 실행, 설치, import, 체크포인트·데이터 다운로드를 하지 않았습니다. 복제 내용은 신뢰하지 않는 데이터로 다뤘습니다(`repo/`는 `.gitignore`).
- 핵심 파일: `src/openpi/models/pi0_fast.py`(디코딩 루프), `src/openpi/transforms.py`(`TokenizeFASTInputs`), `src/openpi/policies/droid_policy.py`(입력 키), `src/openpi/models/model.py`(`Observation`), `scripts/arrow_log.txt`, `paper-analyses/expt1-figures.ipynb`.

## 1. 설치 요구사항과 라이선스

- 라이선스: Apache-2.0(`LICENSE`, openpi 상속), `LICENSE_GEMMA.txt` 동봉(PaliGemma 가중치 조건). 포크 자체의 추가 라이선스 표시는 없습니다.
- 요구사항: openpi 그대로입니다(JAX, uv, 학습된 `pi0_fast_droid` 체크포인트, `physical-intelligence/fast` 토크나이저). README는 제목 한 줄뿐이라 설치·실행 안내가 없습니다.
- 평가·로봇 쪽 코드(키보드 입력, 말단 속도 → 관절 속도, 제어 루프, 사용자 연구 기록)는 저장소에 없습니다(D6).

## 2. 논문 ↔ 코드 대조 (D 항목)

| ID | 논문 | 코드 | 판정 |
|---|---|---|---|
| D1 | §3.2: z_i = z̃_i (i ∈ [b, b+w)), 그 밖은 정책이 생성 | `pi0_fast.py` `step()`에서 `token = where(use_action_prefix, prefix_token, where(use_arrow, arrow_token, sampled_token))`. 주입 토큰은 KV 캐시에 일반 토큰처럼 들어가 다음 스텝 조건이 됨. 창은 `(step ≥ prefix_len+bias_start) & (step < prefix_len+inject_window+bias_start)` | **일치** (디랙 질량 = 토큰 강제) |
| D2 | §4.1–4.2: w ∈ {1,2,4,6}, b ∈ [0,3] 스윕 | `inject_window = 4`, `bias_start = 0`이 **함수 안 상수**. 주석 "UNCOMMENT BELOW FOR MANUAL INJECT WINDOW SWEEP". 스윕은 코드를 고쳐 가며 한 것. 사용자 연구 값(w = 4)이 현재 기본값. `pi0_fast_freq_expt.py`에는 `freq_offset = 5`인데 주석은 "skip first two FAST tokens", `inject_window = 2`의 주석은 "inject tokens 3 and 4 only" | **불일치/불명**: 설정 파일·CLI 없음. 상수와 주석이 서로 안 맞음. 논문의 b ∈ [0,3]은 최신 `pi0_fast.py`의 `bias_start`로 가능하나 어느 값으로 어느 그림을 만들었는지 코드로 확인 불가 |
| D3 | 언급 없음 | `"Action: "` 접두 토큰을 모델이 아니라 코드가 먼저 강제(주석 "idk why this was never here..?"). 업스트림은 모델이 생성 | 논문 서술과 차이 없는 부가 변경. 조향 토큰 z̃_0은 "Action: " 바로 뒤 위치 |
| D4 | §3.2: q̇를 "padded to the action horizon"해 ã 생성 | `transforms.py`: `arrows_action_chunk = zeros((10, 8)); chunk[:, :] = arrows[None, :]` → 같은 벡터를 **10행 복제**. `_fast_tokenizer(chunk)[0]`으로 인코딩, `MAX_ARROW_TOKENS = 32`에 채움 | **일치하되 "padded"는 부정확**: 영 패딩이 아니라 시간 복제. 이 선택이 직류 계수만 남게 해 §4.2 해석에 영향(`03` T1) |
| D5 | §3.2: u는 방향 키 | 조향이 켜지는 조건: `not allclose(arrows[:-1], 0) and len(fast_arrows_pg) > 1`. 그리퍼 성분(마지막)은 판정에서 제외, 조향 청크의 그리퍼 값은 0 | 키를 안 누르면 마스크 0으로 조향 없음 ✓. 조향 토큰이 1개 이하면 조향이 꺼짐(w = 1 실험은 청크 토큰이 2개 이상일 때만 가능) |
| D6 | §3.2, §4.5: 키보드 → v_cart = m·u → q̇ = J†v | **코드 없음.** `arrows`는 클라이언트가 `observation/arrows`(8차원 관절 속도)로 서버에 보냄(`droid_policy.py`). 키 입력, IK, 크기 m, 키 누름 "burst" 처리는 저장소 밖(저자 계정에서도 찾지 못함, `02` §2) | **누락**: 방법의 절반이 재현 불가 |
| D7 | 단위·스케일 미기재 | `arrows`는 `norm_stats` 키(`state`, `actions`)가 아니어서 `Normalize`를 거치지 않음(`transforms.py` `Normalize`는 stats에 있는 키만). 즉 조향 토큰은 **원시 관절 속도 값**으로 토큰화됨. 로그 입력은 모두 \|0.65\| 이하(`arrow_log.txt`)인데 말단 1 m/s의 J†v는 토이 기준 수 rad/s(`03` T2: 앞 방향 3.4 rad/s)이므로 클라이언트가 스케일을 맞췄을 것 | **불명**: 모델의 행동 정규화 단위(DROID 정규화 관절 속도)와 조향 입력 단위가 같은지 논문·코드 어디에도 없음. 정적 읽기 판단이며 실행 확인은 안 함 |
| D8 | §3.2: "autoregressively sampled from p_θ" | `sample_actions(..., temperature: float = 0.0)`, `temperature > 0`일 때만 `categorical`, 아니면 `argmax`. 저장소에서 온도를 지정하는 곳 없음(`policy_config.py`의 `sample_kwargs`는 기본 None) | **기본은 탐욕 디코딩.** §4.3 "다양성"과 엔트로피는 확률적 샘플링이 아니라 장면 의존성과 분포의 평탄도를 본 것. 실험에서 온도를 올렸는지는 불명 |
| D9 | 해당 없음 | `Pi0FASTConfig.inputs_spec`: `arrows`는 `[B, 4]`, `fast_arrows`는 `[B, action_dim=8]`, 반면 실제 값은 8차원과 32 토큰. `Pi0Config`에도 `arrows` 4차원 | 사양과 실제 모양 불일치(주로 `fake_obs`·컴파일 모양 용). 포크가 파이프라인 전체에 걸쳐 정리되지 않음 |
| D10 | §3.2 | 조기 종료(EOS)는 업스트림 그대로: 강제 토큰이 EOS면 종료, 모든 배치가 EOS면 루프 종료 | 일치 |
| D11 | §4.1: H = 10, D = 8 | 최신 코드는 (10, 8) 복제. 그러나 `scripts/arrow_log.txt`(커밋 `8fb0a57`에 포함)의 "decoded arrow action shape: (32, 32)"는 **H = 32 시절** 출력이다(커밋 `4c3019f` 2026-05-02 "changing repeated 32 time horizon to 10"). 이 로그는 비영 관절 속도 7개 입력이 9개 토큰으로 인코딩됨을 보임 | 로그가 코드보다 오래됨. 로그의 디코딩 값(예: −0.3005)은 horizon 32의 DCT 크기(√32)와 맞아 오래된 설정임 |
| D12 | §5.1 "while real-time", 지연 수치 없음 | 디코딩 루프 안 `jax.debug.print` 여러 개(매 스텝 출력), `return_debug=True`가 기본이라 `sample_actions`가 튜플 반환, 정책 서버가 디버그 딕셔너리를 직렬화해 전송 | 지연 측정을 방해하는 코드. 실시간성 주장은 코드로도 논문으로도 확인 불가 |
| D13 | 부록 B: SIR, MPE, 엔트로피 | **평가 코드 없음.** 엔트로피 계산은 저장소 어디에도 없음(logit을 반환하지 않음). `paper-analyses/expt1-figures.ipynb`는 궤적 CSV를 그리는 노트북 하나이며 창 크기당 파일 8개(총 32개), 데이터는 저장소에 없고 저자 로컬 경로를 읽음. SIR/MPE 계산 셀 없음 | 그림 2–5의 수치를 코드로 재생성할 수 없음. 노트북의 시행 수 8은 표 1의 n ≥ 16과 다름(`03` V1, K3) |
| D14 | 해당 없음 | `TokenizeFASTInputs`가 `arrows = data.get("arrows")`를 무조건 `np.asarray(...)`하고, `Observation.from_dict`도 `arrows`, `tokenized_arrows`를 **필수 키**로 요구 | 업스트림의 일반 π0-FAST 경로(학습, 조향 없는 추론)가 `arrows` 없이는 KeyError 등으로 깨질 가능성이 큼(정적 판단, 실행 안 함). 이 포크는 조향 전용 |
| D15 | 해당 없음 | README는 제목 한 줄(64 B). 스크래치 파일 다수: `pi0_fast_steer_v1.py`, `pi0_fast_steer_v2_works.py`, `pi0_fast_freq_expt.py`, `pi0_fast_with_debug_logging.py`, `pi0_fast_original.py`, `policy copy.py`, `transforms_*` 등 | 어느 파일이 어느 그림을 만들었는지 불명. 실제 연결된 것은 `pi0_fast.py`와 `transforms.py`뿐(`config.py`는 `pi0_fast`만 import) |

### 2.1 논문 주장의 코드 쪽 확인

- "추가 학습 없음"(C1): 가중치·학습 코드 변경 없음, 디코딩 토큰만 바꿉니다. **확인됨**(D1).
- "TS는 어떤 저차원 입력에도 쓸 수 있다"(§5): 코드는 8차원 관절 속도 벡터를 받을 뿐이라 입력 형식 요구는 "관절 속도 8차원"입니다. 방향 키 이외의 입력은 변환이 클라이언트 몫(D6).
- "저주파 토큰이 전역 의미를 가진다"(C3): 코드가 조향 토큰에 넣는 것은 시간 복제 청크의 FAST 인코딩이라 위치 0–4는 직류 계수(`03` T1). 코드가 이 해석을 뒷받침하지는 않습니다.

## 3. 논문 실험과 코드의 연결

| 논문 실험 | 코드로 재생성 가능한가 |
|---|---|
| §4.1 창 크기 (표 1, 그림 2) | 부분. 창은 상수 수정(D2), 궤적 CSV 그림 노트북 있음, SIR·MPE 계산·원자료 없음(D13) |
| §4.2 시작 인덱스 (그림 3) | 부분. `bias_start` 상수, 별도 `freq_expt` 파일(D2) |
| §4.3 엔트로피 (그림 4, 7) | 불가. logit·엔트로피 코드 없음(D13), 기본 탐욕 디코딩(D8) |
| §4.4 언어 교정 (그림 5) | 불가. 클라이언트와 평가 코드 없음 |
| §4.5 사용자 연구 (표 2) | 불가. 키보드·IK 클라이언트와 기록·통계 코드 없음(D6) |

## 4. 구현 가능성 (이 맥)

이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 가능한 것과 아닌 것입니다.

| 작업 | 판정 | 근거 |
|---|---|---|
| 논문 표·그림 수치 재계산, 정합성 검사 | **완료** | `analysis/recompute.py` |
| 조향 토큰 prefix 정보 토이 | **완료** | `analysis/toy_steering_prefix.py` |
| 공식 코드 정적 대조 | **완료** | 위 D1–D15 |
| 논문 실험 재현 (DROID 실로봇, 7명 사용자 연구) | **불가** | Franka Panda + Robotiq + ZED 카메라, 사용자 모집·IRB 필요 |
| π0-FAST-DROID 추론 (JAX) | **조건부** | JAX·openpi 설치, 체크포인트 다운로드 승인 필요. 3B급 모델을 CPU/MPS에서 돌려 보는 것은 가능할 수 있으나 속도 미측정(논문 추론은 RTX 5090) |
| 이 포크 그대로 실행 | **조건부** | 위 + `arrows` 클라이언트를 새로 작성해야 함(D6) |
| 방법 구조 재구현(작은 자기회귀 모델, 토큰 prefix 조향) | **가능** | 합성 궤적, DCT + 양자화 + 소형 트랜스포머, PyTorch(MPS/CPU). 학습 수 분–수 시간 |
| FAST+ 토크나이저로 조향 청크 토큰 열 직접 확인 | **조건부** | HF 토크나이저(작은 파일) + transformers·scipy 설치 승인 필요. 몇 분 |

노력 추정(추정): 소형 모델 구조 재구현 3–5일, 클라이언트 스택(키보드 → IK → 제어, 실로봇) 1–2주, 사용자 연구 재현은 사용자 20명 이상 기준 수 주.

**판정: 논문 실험은 이 맥에서 구현할 수 없습니다(실로봇과 GPU, 사용자 연구가 필요).** 방법의 핵심(토큰 prefix 강제)은 20줄 남짓의 디코딩 루프 수정이라 **구조 재구현은 쉽고**, 작은 모델 토이로 이 맥에서 검증할 수 있습니다. 단 저주파/고주파 해석을 검증하려면 조향 청크를 직류가 아닌 신호로 구성해야 합니다(`03` T2).

## 5. 재실행

이 문서의 대조는 읽기만 했으므로 재실행 명령이 없습니다. 같은 포크를 받으려면:

```bash
git clone --depth 50 https://github.com/jasontchan/openpi-fast-steering
git -C openpi-fast-steering checkout 8fb0a577ba3ec815338696d7229a028bfffba2bc
git -C openpi-fast-steering diff fdc03f5 HEAD --stat
```

# DreamerV3 공식 코드 대조 (04)

분석일 2026-11-21. 대상은 공식 저장소 `danijar/dreamerv3`의 커밋 `e01491f`(2026-10-04)입니다. 얕은 복제본은 `repo/`에 있고, **정적으로만 읽었습니다.** 실행, import, 설치, 체크포인트 다운로드를 하지 않았습니다. 저장소 내용은 신뢰할 수 없는 자료로 취급했습니다.

읽은 파일: `dreamerv3/configs.yaml`(220줄), `dreamerv3/agent.py`(핵심 손실·최적화기), `dreamerv3/rssm.py`(RSSM, free bits, unimix), `embodied/jax/heads.py`(symexp twohot), `dreamerv3/main.py`(로거의 스텝 환산), `embodied/envs/{procgen,minecraft,dmc}.py`(환경 래퍼 기본값), `baselines.yaml`(정규화 기준값), `requirements.txt`, `Dockerfile`, `README.md`, `LICENSE`.

## 1. 저장소 구조

| 경로 | 내용 |
|---|---|
| `dreamerv3/main.py`, `agent.py`, `rssm.py`, `configs.yaml` | 학습 진입점, 에이전트(손실, 최적화), RSSM, 설정 |
| `embodied/` | 공용 인프라: 환경 래퍼(Atari, DMC, DMLab, Minecraft, ProcGen 등), JAX 유틸, 실행 루프 |
| `baselines.yaml` | Atari·DMLab·ProcGen 정규화 기준값 |
| `scores/` | 실행별 곡선 파일 27개와 `view.py`(읽기 전용 요약 스크립트, 실행하지 않음) |
| `Dockerfile`, `requirements.txt` | 환경 구성. Docker는 NVIDIA 드라이버 베이스 이미지 |

- 코드 규모: `dreamerv3/*.py` 1,125줄, 저장소 전체 Python 10,360줄입니다.
- 라이선스: MIT, 저작권 표기 "Copyright (c) 2024 Danijar Hafner".
- README는 이 구현을 "DreamerV2 코드 기반의 재구현(reimplementation)"이라고 부르며, "Google이나 DeepMind와 무관하다"고 씁니다. 그리고 "모든 결과를 재현하도록 테스트했다"고 씁니다. 저자가 직접 관리하는 공식 구현입니다.

## 2. 논문과 일치하는 항목

| 항목 | 논문 | 코드 (위치) | 판정 |
|---|---|---|---|
| 손실 가중치 β | pred 1, dyn 1, rep 0.1 (식 2) | `loss_scales: rec 1, dyn 1, rep 0.1` (configs `agent`) | 일치 |
| 비평자 반복 손실 | βrepval = 0.3 | `repval: 0.3` | 일치 |
| free bits | max(1, KL), 1 nat (식 3) | `dyn = max(dyn, free_nats)`, `rep = max(rep, free_nats)`, `free_nats: 1.0` (rssm.py:127–129, 32) | 일치 |
| 범주 1% 혼합 | 인코더·동역학·행위자 1% 균등 (§Distributions) | `unimix: 0.01` (rssm.py:25, policy unimix 0.01) | 일치 |
| 할인율·지평 | γ = 0.997, 지평 333 | `horizon: 333` | 일치 |
| 상상 길이 | H = 15 (표 4) | `imag_length: 15` | 일치 |
| 람다 반환 | λ = 0.95 | `imag_loss.lam 0.95`, `repl_loss.lam 0.95` | 일치 |
| 엔트로피 | η = 3×10⁻⁴ | `imag_loss.actent: 3e-4` | 일치 |
| 반환 정규화 | 5–95 백분위, EMA 0.99, L = 1 (식 7) | `retnorm: impl perc, rate 0.01, limit 1.0, perclo 5, perchi 95` | 일치 |
| 비평자 EMA | 감쇠 0.98 (표 4) | `slowvalue.rate: 0.02` | 일치 (rate 0.02 = 감쇠 0.98) |
| 학습률 | 4×10⁻⁵ (표 4) | `opt.lr: 4e-5` | 일치 |
| AGC | 0.3 (§Optimizer) | `opt.agc: 0.3` | 일치 |
| LaProp ε, β₁ | 10⁻²⁰, 0.9 (본문) | `eps: 1e-20`, `beta1: 0.9` (agent.py:346–347) | 일치 |
| 재생 | 균일, 온라인 큐, 용량 5×10⁶ (표 4) | `replay: size 5e6, online True, fracs uniform 1.0` | 일치 |
| 배치 | 16×64 (표 4) | `batch_size: 16`, `batch_length: 64` | 일치 |
| 인코더 | stride-2, symlog 입력 (§Networks) | `enc.simple: symlog True, mults [2,3,4,4]`, 깊이 64 | 일치 |
| 보상·가치 출력 | symexp twohot, 출력층 0 초기화 (§Robust, §Critic) | `rewhead/value: output symexp_twohot, outscale 0.0`, `bins 255` | 일치 |
| 순환 모델 | GRU, 블록 8 (§Networks) | `rssm.blocks = 8` (rssm.py:31), `BlockLinear` | 일치 |
| 층 수 | 인코더·디코더·행위자·비평자 3층, 보상·계속 1층 (§Networks) | `layers: 3` (enc, dec, policy, value), `rewhead/conhead layers: 1` | 일치 |
| Atari | 스티키 행동, action repeat 4 (§Benchmarks) | `atari: sticky True, repeat 4` | 일치 |
| 환경 병렬 | 16 (§Environment instances) | `run.envs: 16` | 일치 (Minecraft 제외, D8) |
| ProcGen | hard 난이도, 무제한 레벨 (§Benchmarks) | 설정에 지정 없음. 패키지 기본값에 의존 | 확인 필요 (패키지 소스는 읽지 않음) |
| Minecraft 블록 깨기 배율, 길이 | 100, 36,000 스텝 (§Minecraft) | `minecraft: break_speed 100.0, length 36000` | 일치 |

## 3. 차이 (D1–D8)

판정 기호: **불일치** (논문 본문 또는 표와 코드가 다름), **미기재** (논문이 말하지 않은 것을 코드가 정함), **확인 필요** (정의나 단위가 달라 보이나 확정할 수 없음).

| 번호 | 항목 | 논문 | 코드 | 판정과 영향 |
|---|---|---|---|---|
| D1 | LaProp β₂ | 본문 "β₂ = 0.99" (§Optimizer) | `beta2: 0.999` (configs `opt`, agent.py:348) | **불일치.** 최적화기의 기억 길이가 10배 다릅니다. 표 4에는 β가 없어서 본문과 코드 중 어느 쪽이 실험에 쓰였는지 판단할 수 없습니다. 저자에게 확인이 필요합니다. |
| D2 | 제어 스위트 모델 크기 | 표 2: DMC 고유감각·시각 모두 **12M** 모델 (§Computational choices도 같음) | 고유감각: `*size1m`(깊이 4, hidden 64, deter 512, 약 1M급). 시각: 크기 지정 없음 → 기본 **200M**(deter 8192). `size12m` 블록은 정의만 되어 있고 어떤 과제에도 쓰지 않음 (configs 120–130, 179) | **불일치.** 두 DMC 설정 모두 논문의 12M과 다릅니다. 시각 스위트는 기본 200M 모델로 학습하는 것으로 보입니다. 논문 본문의 "12M 모델이 같은 성능"이라는 문장과 맞지 않습니다. |
| D3 | 12M 모델의 순환 유닛 | 표 3: 12M, 순환 유닛 **1,024** (8d는 2,048) | `size12m`: deter **2,048**, hidden 256, classes 16, depth 16, units 256 | **표 3의 오기로 보입니다.** 식 8d와 코드가 맞고, 표 3의 1,024만 다릅니다. 다른 크기(25M–400M)는 모두 8d와 일치합니다. |
| D4 | 재생 비율 (replay ratio) | 표 2: Atari 100k 128, DMC 고유감각 512, DMC 시각 512, ProcGen 64, Minecraft 32, DMLab 32, Atari 32, BSuite 1024 | 설정: Atari 100k **256**, DMC 고유감각 **1024**, DMC 시각 **256**, ProcGen 64, Minecraft 32(기본), DMLab 32, Atari 32, BSuite 1024 | **불일치 (세 과제).** 논문은 action repeat를 뺀 시간 스텝 기준이라고 정의합니다(§Implementation). 코드의 `train_ratio`가 같은 정의인지는 확인하지 못했습니다(**확인 필요**). 정의가 같다면 Atari 100k, DMC 두 스위트는 설정이 바뀌었습니다. |
| D5 | 예산과 action repeat | 표 2: DMC 고유감각 500K, 시각 1M, action repeat **2**. ProcGen 50M, DMLab 100M, Atari 200M 프레임 | DMC 설정은 action repeat 기본값 **1** (`env.dmc.repeat: 1`, dmc.py 기본 1). 스텝 설정: DMC 1.1M(×1), ProcGen 1.1e8(×1), DMLab 2.6e7(×4), Atari 5.1e7(×4), Atari 100k 1.1e5(×4) | **불일치 (DMC, ProcGen).** 로거는 `repeat`를 곱해 env 스텝을 기록합니다(main.py `make_logger`). 그래서 DMLab·Atari는 100M·200M과 맞습니다(2.6e7×4 = 104M, 5.1e7×4 = 204M). DMC는 action repeat 1이 기본이라 논문 예산과 다릅니다. ProcGen 1.1e8은 논문 50M의 두 배가 넘습니다. 설정이 최대 한도인지 실험 값인지 **확인 필요**. |
| D6 | 보상·가치 bin 개수와 범위 | 식 10: B = symexp(−20 … +20), 개수는 본문에 없음 | 255개, `linspace(−20, 0)`을 대칭으로 펼친 값 (heads.py:70, 134–144) | **미기재.** 범위(±20)는 맞고, 개수 255는 본문에 없습니다. 재구현하는 사람이 알아야 할 정보입니다. |
| D7 | 최종 점수 집계 | 본문에 없음 | 저장소에 집계 규칙 없음 (`scores/view.py`는 곡선만 요약) | **미기재.** 재계산 결과(`03` §V5)에서 벤치마크마다 다른 평가 점 수가 맞았습니다. 논문과 저장소 모두 이 규칙을 적지 않았습니다. |
| D8 | Minecraft 환경 병렬 수 | 표 2: 64 인스턴스 (원격 CPU 워커) | Minecraft 설정 블록에 `envs` 없음 → 기본 16. `remote_envs: False` | **미기재.** 논문 설정에 맞추려면 원격 워커 설정이 필요한데 저장소에는 없습니다. |

- 확인 방식: D1, D2, D3, D4는 `configs.yaml`과 `agent.py`, `rssm.py`를 직접 읽고 대조했습니다. D5는 `main.py`의 로거 환산 코드를 확인한 뒤 계산했습니다. D6은 `heads.py`를 읽었습니다.
- 공식 저장소의 실행 점수 파일(`scores/`)과 설정의 관계: Atari 100k 파일은 5 seed이고, 설정의 `train_ratio 256`과 표 2의 128 중 어느 쪽으로 실행했는지 파일만으로는 알 수 없습니다.

## 4. 구현 가능성 판정 (이 맥)

**판정: 이 맥에서 논문 전체 재현은 불가능합니다. 공식 코드의 부분 실행과 핵심 손실의 소형 재구현은 가능하지만, 아직 하지 않았습니다.**

이 맥은 Apple M4 Pro, 약 24 GiB 메모리, MPS 가속기이며 CUDA는 없습니다.

| 구분 | 이 맥에서 | 근거 |
|---|---|---|
| 표 재계산과 검정 | **완료**. CPU, 수 초 | `analysis/recompute.py` |
| 공식 코드를 CPU로 짧게 돌리기 (설정 `debug`, `--jax.platform cpu`) | **조건부**. 설치 필요 (JAX 등). 이 분석에서는 설치하지 않았음 | README는 CPU 실행을 지원한다고 적음. `requirements.txt`는 `jax[cuda12]==0.4.33`을 고정함. 그대로 설치할 수 없음. JAX의 Apple GPU 경로는 이 버전에서 쓸 수 없는 것으로 보이나 확인하지 않음 |
| 디버그 설정 한 번 학습 (수 분 규모) | 조건부 | `debug` 블록은 네트워크를 작게 줄이고 학습을 "좋은 모델이 아니다"라고 명시 |
| Atari 100k 1게임 (200M 모델, 표 2) | 불가 (시간, 추정) | replay ratio 256 (설정), 100K 에이전트 스텝, 200M 파라미터 모델. CPU 학습 시간은 측정하지 않음 |
| DMC 1과제, 500K–1M 스텝 | 불가 (시간, 추정) | replay ratio 512–1024, 그라디언트 스텝 약 10⁵ 규모로 추정. 측정 전 |
| DMLab 100M, ProcGen 50M, Atari 200M | 불가 | 예산 자체가 GPU 수십 일 규모(표 2). DMLab은 Bazel 빌드와 Linux 환경을 가정 |
| Minecraft 100M | 불가 | MineRL 0.4.4 패치 wheel (GitHub 릴리스), Java 8, 리눅스 Docker 가정. A100 9일 (표 2) |
| 표 값 재현 (논문 전체) | 불가 | 위 항목들 |
| PyTorch로 RSSM·손실·반환 정규화를 소형 재구현해 토이 과제에서 확인 | 가능 (추정 1–2주) | 손실과 정규화는 식 1–12로 충분히 구현 가능. 커뮤니티 PyTorch 구현도 있음(`02` §4). 이 분석에서는 하지 않음 |

- **노력 추정 (측정 전):** 공식 코드의 CPU 설치와 `debug` 실행 0.5–1일. 소형 PyTorch 재구현과 토이 검증 1–2주. 공식 설정으로 DMC 한 과제를 CPU에서 끝까지 학습하는 것은 수일에서 수주 규모로 추정하며 측정하지 않았습니다.
- **외부 설치 스크립트 주의:** `Dockerfile`은 DMLab 설치 스크립트를 `gist.githubusercontent.com`에서 받아 바로 `sh`로 실행합니다. 재현할 때는 내용을 먼저 확인해야 합니다.

## 5. 결론

- 논문의 핵심 식(손실 구조, free bits, unimix, 반환 정규화, symexp twohot, 최적화기)은 코드와 대체로 일치합니다. 주요 하이퍼파라미터도 대부분 일치합니다.
- 차이는 주로 **제어 벤치마크의 설정**(D2, D4, D5)과 **LaProp β₂**(D1), **표 3의 오기**(D3)에 몰려 있습니다. 논문 본문만 읽으면 이 차이를 알 수 없습니다.
- 집계 규칙(D7)과 bin 개수(D6)처럼 재구현에 필요한데 빠진 정보가 있습니다.

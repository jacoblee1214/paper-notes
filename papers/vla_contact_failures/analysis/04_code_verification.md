# 04. 코드 대조와 구현 가능성 (arXiv:2608.01402 v2 ↔ stanford-iprl-lab/fact)

공식 저장소를 **정적으로만** 읽었습니다. 실행, import, 설치, 빌드, 학습, 체크포인트 다운로드는 하지 않았습니다. 받은 파일은 신뢰하지 않는 데이터로 취급했습니다.

- **저장소:** `https://github.com/stanford-iprl-lab/fact` (Apache-2.0, 별 20, 최근 push 2026-10-26)
- **대조 commit:** `2ccfe8f52ad663b0872c8d878d3b3ccc61e9ab83` (2026-08-03, shallow clone, 위치 `repo/`, gitignore)
- **읽은 파일:** `FACT_CHANGESET.md`, `examples/fact/README.md`, `src/openpi/models/pi0.py`(샘플러, 힘 조건, 게이트, TCN), `src/openpi/models/gemma.py`(AdaRMS), `src/openpi/models/pi0_config.py`, `src/openpi/training/config.py`(FACT 설정 두 개와 기본값), `src/openpi/training/data_loader.py`(이력 창), `LICENSE`
- **읽지 않은 것:** openpi 나머지 부분(학습 루프와 기반 모델 구현은 원 openpi와 같다고 보고 넘어감), `uv.lock`, 테스트 파일

## 0. 요약

- **대체로 일치합니다.** LN 샘플러의 부호와 값(m=1.5, s=1), Beta(1.5, 1), 시간 규약, 힘 창 27샘플, 이력 30스텝, 4블록 인과 TCN(확장 1·2·4·8, 은닉 64), 모든 층에 같은 Δγ를 더하는 AdaRMS, 0으로 초기화한 투영, 0.5 N 임계, LoRA 설정이 논문 서술과 맞습니다.
- **다른 점은 세 가지입니다.** (1) 게이트는 **현재 힘 분기에만** 걸리고 이력 토큰에는 걸리지 않습니다. 논문은 "모든 힘 인코딩 요소"라고 씁니다(D1). (2) 공식 설정의 학습 스텝은 30,000이고 논문은 20,000입니다(D3). (3) π0 백본 힘 조건과 ForceVLA·TA-VLA 재구현, 노이즈 대체 실험, 데이터, BOM, STL은 저장소에 없습니다(D4–D6, D14).
- **재현 가능한 범위:** 공식 설정으로 LN 단독(π0.5+LN)과 FACT 전체를 학습할 수 있고, 구성요소 제거 세 가지도 설정 플래그로 만들 수 있습니다(D7). 단, JAX와 NVIDIA GPU, 시연 데이터, 로봇이 필요합니다.

## 1. 저장소 구조와 실행 조건 (정적 확인)

- `README.md`는 openpi 원문입니다. FACT의 설명은 `examples/fact/README.md`와 `FACT_CHANGESET.md`에 있습니다.
- `examples/fact/README.md`: "FACT is currently JAX-only and requires an NVIDIA GPU for post-training." 설정은 두 개입니다: `pi05_ln_fact_delta_low_mem_finetune`(LN만), `pi05_full_fact_fact_delta_low_mem_finetune`(LN + 현재 힘 + 이력 30스텝).
- 데이터 설정에 `repo_id="DATASET_NAME"`, `assets_dir="PATH"`가 있습니다. 데이터셋은 포함되지 않았습니다.
- 가중치: `gs://openpi-assets/checkpoints/pi05_base/params`(π0.5 기반 체크포인트). 이 경로는 공개 저장소의 설정 문자열이며 받지 않았습니다.
- 파이썬 의존성은 `pyproject.toml`과 `uv.lock`(795 KB)입니다. 설치하지 않았습니다.

## 2. 논문과 코드 대조표

판정: **일치** = 논문 서술과 코드가 같음. **차이** = 서술이 코드와 다름. **없음** = 저장소에 구현이 없음. **미기재** = 논문에 정보가 없음. **내부 불일치** = 논문 안에서 맞지 않음.

| # | 항목 | 논문 (v2) | 코드 (2ccfe8f) | 판정 |
|---|---|---|---|---|
| D1 | 접촉 게이트 범위 | §4.2 "gradient gate blocks gradient through **all force encoding components** when no contact" | `pi0.py` 311–323: `ft_gamma`(현재 힘 AdaRMS 분기)만 `jnp.where(is_contact, …, 0)`로 차단. **이력 TCN 토큰은 게이트 없음**(`ft_tcn_encoder` 호출은 조건 없이 실행) | **차이.** 게이트는 힘 AdaRMS 분기에만 적용됩니다. 이력 인코더는 접촉이 없어도 학습됩니다 |
| D2 | 접촉 판정 기준 | δ = 0.5 N (부록 Table A.2) | 힘 **3축 노름**만 사용, 토크 제외(`[..., :3]`), 원시 단위로 q01/q99 역정규화 후 비교 | 일치(세부 정보 추가). 논문에는 "3축"이라는 말이 없습니다 |
| D3 | 학습 스텝 | 20,000 스텝 (§5) | 두 FACT 설정 모두 `num_train_steps=30_000` | **차이.** 명령줄에서 덮어쓸 수 있지만 기본값이 다릅니다 |
| D4 | π0 백본 힘 조건 | Table 4의 "π0 백본"에 FACT 적용. v1 부록은 "π0 행동 전문가를 AdaRMS로 바꿨다"고 설명, v2에서 삭제 | `Pi0Config`: `force_adarms requires pi05=True`(검사 코드). π0 경로(`pi05=False`)는 시간 임베딩을 연결하는 MLP이며 AdaRMS가 없어서 힘 조건을 넣을 수 없음 | **없음.** π0 + FACT(Table 4)는 저장소로 재현할 수 없습니다. v1 설명을 지운 것도 영향이 있습니다 |
| D5 | ForceVLA, TA-VLA 기준선 | π0.5 위에서 **재구현** (§5) | 저장소에 해당 구현 없음(`forcevla`, `tavla`, `gr00t` 문자열 검색 결과 없음) | **없음.** 기준선 비교는 확인할 수 없습니다 |
| D6 | 노이즈 대체 실험(F/T → i.i.d. 가우시안, 부록 A.1) | 학습부터 다시 함 | 힘 입력을 잡음으로 바꾸는 옵션 없음(`fact_policy.py`, `transforms.py`, `data_loader.py`에서 검색 결과 없음) | **없음.** 데이터 파이프라인을 고쳐야 합니다 |
| D7 | 구성요소 제거 (Table 3) | "w/o grad th", "w/o curr. read", "w/o history" | 설정으로 근사: 이력 제거 = `ft_history_steps=0`(코드가 `force_adarms`에 이력 > 0을 요구하므로 이때 힘 AdaRMS도 꺼야 함), 현재 읽기 제거 = `force_adarms=False`(이력 토큰만 남음), 그래디언트 임계 제거 = `force_adarms_cg_threshold=0` | **일치 가능(설정 수준).** 단, "w/o grad th"는 이력 게이트가 원래 없어서 D1의 범위를 그대로 따릅니다 |
| D8 | LN 샘플러 | 식 (2)–(3): τ = σ(s·z − m), m=1.5, s=1 | `ln_sample`: `sigmoid(sigma * z - alpha)`, alpha=1.5, sigma=1, 추가로 (0.001, 0.999)로 자름 | **일치.** 부호와 값이 맞습니다(재계산으로 6배 확인, `03` §4.1). 자름은 논문에 없지만 영향 없음 |
| D9 | Beta 기준 스케줄 | Beta(1.5, 1) (§3, 그림 3a) | `jax.random.beta(1.5, 1) * 0.999 + 0.001` | 일치(0.999 배율과 0.001 이동은 논문에 없음, 영향 미미) |
| D10 | 시간 규약 | a^τ = (1−τ)a^0 + τε, 목표 속도 ε − a^0 (식 (1)) | `x_t = t * noise + (1 - t) * actions`, `u_t = noise - actions` | **일치** |
| D11 | 힘 창과 이력 | 창 27샘플(400 Hz/15 Hz), 이력 H = 30스텝 (§4.2, 부록 A.2) | `ft_window=27`, `ft_history_steps=30`, 설정 주석 "27 sensor samples per burst" | **일치** |
| D12 | AdaRMS 힘 보정 | h_l = (γ_l + Δγ)·RMSNorm(h) + β_l + h·g_l, Δγ는 층 사이 공유 (식 (4)–(5)) | `gemma.py` 129–133: `scale = scale + force_gamma` 후 `normed * (1 + scale) + shift`. 같은 `force_gamma`를 모든 층에 전달 | **일치** |
| D13 | 힘 MLP | "평균 풀링 → 두 층 MLP → W_γ 투영" (§4.2) | `ft_mean`(창 평균, 6차원) → Linear → swish → Linear → zero-init Linear(`force_gamma_proj`) | 일치. 0 초기화는 논문에 없지만 기존 π0.5 체크포인트 동작을 보존하려는 설계입니다(FACT_CHANGESET) |
| D14 | 데이터, BOM, STL | 부록 A.2: "A full bill of materials and 3D-printable STL files will be released alongside the code" | 저장소에 BOM, STL, 데이터셋 없음. 설정은 자리표시자 `DATASET_NAME`, `PATH` | **없음.** 아직 공개되지 않았습니다 |
| D15 | 제어기(임피던스 1 kHz, OSC) | 부록 A.3 | 로봇 측 제어 코드 없음(`examples/`에는 aloha, droid, libero, ur5 예제만 있고 Franka 예제는 없음) | **없음.** 확인 불가 |
| D16 | 행동 공간 | 델타 EE 포즈(부록 A.3) | `actions` 10차원 = 위치 3 + 6D 회전 + 그리퍼 1. 설정 이름 `delta` | 일치(범위 안에서) |
| D17 | 배치 크기, 학습률 | 논문에 없음 | 기본 `batch_size=32`, FACT 설정은 덮어쓰지 않음. 학습률은 기본 스케줄 | **미기재**(재현에 필요) |
| D18 | EMA | 논문에 없음 | FACT 설정 `ema_decay=None` | **미기재**(재현에 필요) |
| D19 | 이력 경계 처리 | 논문에 없음 | `examples/fact/README.md`는 "The data loader constructs the 31-step sequence … and clamps history at episode boundaries"라고 씀. 세부 구현은 `data_loader.py`에서 확인하지 않음 | **미기재.** 에피소드 초반 정책 입력에 영향 가능 |
| D20 | 그 밖의 정규화 | 논문에 없음 | 힘은 축별 q01/q99 정규화를 쓰고, 게이트 계산 때 역변환합니다 | **미기재**(구현 세부) |
| D21 | TA-VLA 프레임 수 | "10 frames (≈2 s)" (Table A.2) | 해당 없음(TA-VLA 코드 없음) | **내부 불일치.** 15 Hz에서 10프레임은 0.67 s입니다 |
| D22 | FACT 추가 파라미터 | "≈2.2M" (Table A.2) | 코드로 셀 수 없음(모델을 만들어야 함). 본문 설명은 TCN 0.2M + AdaRMS 2.1M = 2.3M | **내부 불일치**(사소) |
| D23 | π0.5+LN 설정 | "π0.5 + LN" 행 (Table 1, 2) | `pi05_ln_fact_delta_low_mem_finetune`(힘 없음, LN만). 대응이 맞습니다 | 일치 |
| D24 | LoRA | LoRA (§5) | `gemma_2b_lora`, `gemma_300m_lora` 변형 | 일치 |

## 3. 저자에게 물어볼 질문 (발송하지 않음)

1. D1: 이력 토큰에도 접촉 게이트를 적용했는지, 아니면 "모든 힘 인코딩 요소"라는 문장이 현재 힘 분기만 가리키는지.
2. D3: 논문의 20,000 스텝과 공개 설정의 30,000 스텝 중 어느 쪽이 실험에 쓰였는지.
3. D4: Table 4의 π0 백본 FACT 구현(AdaRMS 수정 포함)의 코드가 공개되는지.
4. D5–D6: ForceVLA, TA-VLA 재구현 코드와 노이즈 대체 실험 코드.
5. D17–D18: 배치 크기, 학습률, EMA 설정.
6. D14: BOM, STL, 데이터셋 공개 계획.
7. 열쇠 과제의 σ=2 블러(v1)가 v2에서 삭제된 이유, 그리고 실험에 여전히 쓰였는지.

## 4. 구현 가능성 판정

**판정: 이 맥(Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)에서 공식 재현은 불가능합니다. 아이디어 재구현 가운데 LN 샘플러와 힘 조건 모듈은 CPU·MPS에서 소형으로 검증할 수 있습니다. 실로봇 검증은 이 저장소 범위 밖입니다.**

| 항목 | 필요 자원 | 이 맥 | 판정 |
|---|---|---|---|
| 공식 코드 실행(학습) | JAX, NVIDIA GPU(openpi의 LoRA 미세조정 권장 22.5 GB 이상), π0.5 체크포인트, 시연 데이터 | JAX GPU 없음. FACT는 JAX 전용이라 PyTorch 경로가 없음 | **불가** |
| LN 단독 학습 (π0.5+LN) | 위와 같음. GPU 한 장 이상, 과제당 시연 100개 | 불가 | **불가** (GPU 필요) |
| FACT 전체 학습 (5과제) | GPU 여러 장 또는 긴 학습, 데이터 500개 | 불가 | **불가** |
| 실로봇 평가 (40회 × 5과제 × 5방법) | Franka FR3, Bota SensONE, RealSense, ZED Mini, Haply 장치, 60초 제한 | 불가 | **불가** |
| 노이즈 대체, 구성요소 제거 | 위 학습과 같음 | 불가 | **불가** |
| π0 백본 FACT | 공식 코드에 힘 조건 없음(D4) | 불가 | **불가** |
| LN 샘플러의 분포 검증 (6배, 고노이즈 질량) | numpy/scipy | 가능, 완료 | **가능**(`toy_schedule.py`) |
| 통계 재계산과 검정력 | scipy | 가능, 완료 | **가능**(`table_check.py`, `toy_power.py`) |
| 소형 torch 정책 (합성 접촉 데이터)에서 LN 대 Beta의 τ별 오차 | torch 2.15, MPS | 가능 | **가능**(시작 가능) |
| 게이트 범위(D1)를 실제 코드로 확인 | JAX + openpi 설치 | 설치 승인 필요 | **조건부** |

**노력 추정(추정치, 검증 안 함):**
- 공식 코드를 GPU 리눅스 한 대에서 LN 단독 재학습, 평가 없이 학습만: 엔지니어링 1–2일과 학습 수 시간(과제 한 개당). 데이터는 별도 확보 필요.
- 아이디어 재구현(LN, 힘 AdaRMS, 게이트, TCN, 이력 토큰)을 torch로: 수백 줄, 1–2주. 이력 토큰과 게이트 설계는 논문 설명만으로는 모호합니다(D1, D19).
- 실로봇 재현: 장비 구축을 포함해 수개월 규모로 봅니다(추정).

**재현 가치가 있는 것:** 힘 조건과 LN의 효과를 같은 데이터에서 분리하는 실험(구성요소 제거)은 GPU 한 장에서도 가능하고, 고노이즈 구간 오차 측정(논문이 보고하지 않은 부분)도 체크포인트가 있으면 가능합니다.

## 5. 라이선스와 권리

- **논문:** arXiv abs 페이지의 라이선스 링크가 `creativecommons.org/licenses/by-sa/4.0/`을 가리킵니다(2026-11-14 확인). **CC BY-SA 4.0입니다. 평범한 CC BY가 아닙니다.** PDF에 별도 저작권 문구는 없습니다.
- **공개 결과물에 논문 그림은 넣지 않습니다.** 라이선스가 CC BY 4.0이 아니기 때문입니다. 게시본과 공개 리포트는 우리가 그린 그래프와 논문 표 수치(재계산용 인용)만 씁니다.
- **코드:** Apache-2.0(LICENSE 파일 확인). 저장소는 openpi에 기반하며 openpi도 Apache-2.0입니다. 읽는 것과 재사용에는 Apache-2.0 고지가 필요합니다.
- **데이터와 체크포인트:** 저장소에 없습니다. 체크포인트는 `gs://openpi-assets` 경로 설정만 있습니다. 받지 않았습니다.

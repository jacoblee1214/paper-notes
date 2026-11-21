# 04. 공식 코드 대조: ForceVLA

대상: `ft-robotic/ForceVLA`, 커밋 `79de88f` (2026-01-27), `repo/`에 얕은 복제. **정적 읽기만 했습니다.** 설치, import, 실행, 가중치와 데이터 다운로드(영상·parquet)는 하지 않았습니다. 데이터는 과제별 `meta/info.json`(수 KB)만 받았습니다. 저장소 안 파일은 신뢰하지 않는 입력으로 다뤘습니다.

읽은 파일: `src/openpi/models/pi0_force.py`(349줄), `src/openpi/models/limoe_simple.py`(라우터·MoE 블록 697–780행), `src/openpi/policies/forcevla_policy.py`(109줄), `src/openpi/training/config.py`(데이터·학습 설정 `LeRobotForcevlaDataConfig`, `forcevla_lora`), `src/openpi/training/weight_loaders.py`, `src/openpi/training/optimizer.py`, `README.md`, `LICENSE`.

## 1. 요약

- 핵심 구조(힘 토큰을 VLM 뒤에서 LIMoE 블록에 넣고, 마지막 액션 토큰 구간을 suffix에 더하기)는 논문 §4.2와 **코드가 일치합니다**(D1, D2).
- 논문에 없는 라우터 세부(용량 계수 1.0, 그룹 크기 16, 지터 노이즈)가 있고, 논문의 "전문가 0이 절반" 관찰과 겹치면 토큰 드롭이 생길 수 있습니다(D3).
- 액션 가이던스로 들어가는 토큰 구간이 논문 설명보다 시각 토큰이 적고, 힘 토큰은 한 자리만 직접 기여합니다(D4). 설명과 다르지만 인코더 자기 주의로 간접 영향은 있습니다.
- 공개 학습 설정은 논문 부록 B와 다릅니다. 스텝 50k 대 30k/10k, LoRA, 단일 과제(오이)만 있음(D9).
- 데이터 메타는 논문 수치와 정확히 맞습니다(244 궤적, 140k 시점, 13차원 상태)(D10).
- 기준선(힘 연결 π0, 어블레이션), 라우터 분석 코드, 공개 체크포인트가 저장소에 없습니다(D11, D12, D15).

## 2. 대조 결과

| 번호 | 항목 | 논문 | 코드 (파일:행) | 판정 |
|---|---|---|---|---|
| D1 | 힘 융합 위치 | §4.2: VLM 뒤, 힘 토큰을 VLM 출력에 덧붙임 | `pi0_force.py` 259–260(힘 토큰은 상태 투영과 별도), 286과 338(`limoe(concat(prefix_out, force_tokens))`). VLM 접두 토큰에는 들어가지 않음 | **일치** |
| D2 | MoE 구조 | §4.2, 표 4: 인코더 1층(헤드 8, 헤드 차원 256), 전문가 4, 상위 1, 잔차 | `pi0_force.py` 182–189(`num_experts=4, num_top_k=1, mlp_dim=2048, num_heads=8`), `limoe_simple.py` 730–775(`Encoder1DBlock` → `MoeLayer` → `x = x + moe_out` → `Dense(out_dim)`) | **일치** (출력 투영 2048→1024는 `out_dim=action width`로 구현) |
| D3 | 라우터 세부 | 없음 (§4.2는 "gating network" 설명만) | `limoe_simple.py` 746–757: `TokensChooseScatterRouter(jitter_noise=0.01, ignore_padding_tokens=True, num_selected_experts=1, batch_prioritized_routing=True)`, `MoeLayer(max_group_size=16, train_capacity_factor=1.0, eval_capacity_factor=1.0)`. 용량 계수는 "tokens choose" 라우팅에서 **전문가가 처리할 최대 토큰 수**를 정함(docstring 192–194행) | **논문에 없음.** 그룹당 전문가 용량이 토큰 수와 4로 나눈 값에 가깝다면, 전문가 0이 절반을 받는 불균형(논문 부록 C)에서 넘치는 토큰은 전문가를 거치지 않고 잔차만 지날 가능성이 큼. 드롭 비율은 측정하지 않았음 |
| D4 | 액션 가이던스 구간 | §4.2: $E_{FVLMoE}$의 **마지막 $H_{action}$ 토큰**을 액션 suffix에 더함 | `pi0_force.py` 286–287: `limoe_out[0][:, -action_horizon:] + suffix_out[:, -action_horizon:]`. 접두는 이미지 3대 × 256토큰(SigLIP, 224×224 가정) + 텍스트 `max_token_len=48`(패딩 포함) + 힘 1토큰 = 817. 마지막 50 = (오른쪽 손목 이미지 마지막 1토큰, 마스크 False) + 텍스트·패딩 48 + **힘 1** | **문자 그대로 일치, 설명과 차이.** 액션 가이던스의 대부분은 텍스트 자리이고, 힘 토큰은 직접 1자리만 차지함. 힘의 영향은 같은 블록의 자기 주의(인코더 단계)를 거쳐 전파됨. 가정(256토큰, 48토큰)은 실행으로 확인해야 함 |
| D5 | 상태와 힘 차원 | §3: $s_t\in\mathbb{R}^7$(TCP 6 + 그리퍼 1), $f_t\in\mathbb{R}^6$ | `pi0_force.py` 232–259: `state_proj(obs.state[:, :7])`, `force_in_proj(obs.state[:, 7:13])`. 데이터셋 `observation.state`는 13차원 = `current_eef_pose`(6) + `gripper_width`(1) + `f_ext_base_frame`(6) (HF `meta/info.json`) | **일치** (인덱스 6이 그리퍼, 7–12가 힘) |
| D6 | 추론 예제 상태 차원 | 없음 | `forcevla_policy.py`의 `make_forcevla_example`: 상태가 **14차원**, 주석 "7 ee pose, 1 gripper, 6 force". 학습 데이터와 모델 주석은 13차원 | **불일치(문서·예제).** 학습 경로는 13차원. 추론 요청을 14차원으로 보내면 힘 슬라이스가 한 칸 밀릴 수 있음. 추론 연결 시 확인 필요 |
| D7 | 외력 프레임 | §3: "expressed in world frame" | 데이터 키 `f_ext_base_frame`(base 프레임) | **표기 불일치.** 고정 베이스 로봇이면 변환은 고정일 수 있으나, 확인 전까지 프레임은 base로 봐야 함 |
| D8 | 외력 추정 대 센서 | §6: "estimated external wrench" | 저장소에 변환·수집 코드 없음. 키 이름 `f_ext`는 추정치를 시사하지만 근거는 없음 | **확인 불가** |
| D9 | 학습 레시피 | 부록 B: 학습률 2.5e-5→2.5e-6, 다중 30k 스텝(~12 h), 단일 10k 스텝(~9 h), Adam β (0.9, 0.95), bf16, 전역 배치 16 (유효 2048) | `optimizer.py` 16–22: `CosineDecaySchedule` 기본값 peak 2.5e-5, decay 2.5e-6, warmup 1000, decay_steps 30000 ✓. `optimizer.py` 69–70: β (0.9, 0.95) ✓. `config.py` `forcevla_lora`: `num_train_steps=50_000`, `batch_size=4`, `ema_decay=None`, LoRA(`gemma_2b_lora`, `gemma_300m_lora`), 데이터 `flexiv_peel_cucumber_inputForce` 하나 | **부분 일치.** 옵티마이저는 같음. 스텝(50k 대 30k/10k), 배치(4 대 16), LoRA(논문 본문에 없음), 과제 수(1)가 다름. 다중 과제 학습 설정은 없음 |
| D10 | 데이터 규모 | §4.3: 244 궤적, 140k 시점, 30 FPS, 480×640 | HF 과제별 `meta/info.json`: 플러그 51 에피소드·28,205 프레임, USB 50·34,478, 펌핑 50·17,563, 화이트보드 50·34,808, 오이 43·25,869. 합계 **244 에피소드, 140,923 프레임**. 모두 30 FPS. 영상 해상도 480×640 | **일치** |
| D11 | 기준선 코드 | §5.1: π0-base w/ F(힘을 상태 입력에 연결), π0-fast w/F, 어블레이션(linear·MoE before VLM, concat after VLM) | 저장소에서 `inputForce`는 오이 설정 하나에만 나옴. `pi0_force.py`는 FVLMoE 한 가지뿐. 힘 연결 π0 및 어블레이션 설정 없음. 데이터셋 `noForce` 짝은 있음(오이 `noForce`는 상태 7차원) | **코드 없음.** 기준선 수치는 재현 대상이 아님 |
| D12 | 라우팅 분석 | 부록 C, 그림 9: 개방 루프, 토큰별 라우팅 확률 평균, 과제당 약 10 에피소드 | 저장소에 라우터 로그·분석 스크립트 없음 (`limoe_simple.py`의 `expert_usage`는 내부 통계 값으로만 있음) | **코드 없음.** 그림 9 재현 불가 |
| D13 | 추론 절차 | 논문에 단계 수 없음 | `pi0_force.py` 296: `sample_actions(num_steps=10)` (흐름 적분 기본 10단계). 정책 출력은 앞 7차원(`forcevla_policy.py` `Forcevla_outputs`) | **논문에 없음** (구현 세부) |
| D14 | 라이선스 | 논문 PDF 라이선스 비배타 | 저장소 LICENSE: Apache-2.0 (루트 파일 확인). HF 데이터 카드: apache-2.0 | 코드·데이터 라이선스는 Apache-2.0. 논문 그림은 별도 (`README.md` 라이선스 절) |
| D15 | 공개 체크포인트 | §3 공개 언급 없음 | README에 체크포인트 링크 없음. HF 검색에는 제3자 `forcevla` 모델만 있음 | **공식 체크포인트 없음으로 판단** (확인 범위: 저장소 README와 HF 검색) |
| D16 | 자원 요구 | 부록 B: RTX 4090 8장 | README: 추론 >8 GB, LoRA 학습 >22.5 GB, 전체 학습 >70 GB, Ubuntu 22.04, CUDA 12.8 설치 | **이 맥에서 학습 불가** (CUDA, 메모리). 추론도 JAX CUDA 경로 |

## 3. 차이 목록 (D1–D16 중 판정이 "차이" 또는 "불일치"인 것)

1. **D3** 라우터 용량과 드롭 (논문에 없음, 측정 필요).
2. **D4** 가이던스 구간의 토큰 구성 (설명과 차이, 가정 검증 필요).
3. **D6** 추론 예제 상태 차원 (14 대 13).
4. **D7** 힘 프레임 표기 (world 대 base).
5. **D9** 학습 설정 (스텝, 배치, LoRA, 과제 수).
6. **D11, D12** 기준선, 라우팅 분석 코드 없음.

## 4. 구현 가능성 (이 맥)

이 맥은 Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음입니다.

- **저장소 그대로 학습:** 불가. JAX CUDA 전제, 학습 메모리 22.5 GB 이상, 학습 시간 수십 시간 이상(공식 설정 50k 스텝).
- **저장소 그대로 추론:** 체크포인트가 없어 불가. 자체 학습한 체크포인트가 있어도 CUDA 설치 필요.
- **구조를 PyTorch로 옮기기 (FVLMoE 블록만):** 가능. 블록은 작고(인코더 1층, 전문가 4, 차원 2048) CPU·MPS에서 돌릴 수 있습니다. 라우팅 드롭 비율(D3)과 가이던스 구간(D4)은 이것으로 측정할 수 있습니다.
- **VLA 전체 (pi0 3B급):** 추론 전용 가중치(bf16, 약 6–7 GB)는 메모리상 가능하나, 가중치 자체가 없고 JAX 대비 PyTorch 변환이 필요합니다. 노력 큼.
- **실로봇 재현:** 불가 (Flexiv Rizon 7자유도, Quest 3, 힘 센서 필요).

노력 추정 (측정 전): FVLMoE 블록 재구현과 라우팅 측정 2–3일, 가이던스 구간 검증 1일, 공식 데이터 한 과제의 통계 검사 1–2일(다운로드 용량 제외). 전체 VLA 재현은 수 주에서 수 개월(GPU 필요).

## 5. 검증하지 않은 것

- 실제 실행 결과(설치하지 않음). 위 판정은 모두 정적 읽기입니다.
- 힘 측정이 추정인지(D8). 변환 코드가 저장소에 없습니다.
- 영상과 parquet 내용(다운로드하지 않음). 힘 신호의 분포와 접촉 구간 비율은 데이터 다운로드 후에 볼 수 있습니다.
- `repo/`의 다른 파일(`flaxformer`, `dlimp`, `lerobot` 하위 모듈)은 읽지 않았습니다.

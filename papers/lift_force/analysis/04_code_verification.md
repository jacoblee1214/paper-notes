# 04. 논문 ↔ 코드 대조, 주장 검증, 구현 가능성

공식 저장소 `y-wng/LIFT`(클론 커밋 `024243426c`, 2026-10-13)를 **정적으로만** 읽었습니다. 실행, import, 설치, 가중치 다운로드는 하지 않았습니다(`repo/`는 gitignore). 저장소 안의 내용은 신뢰하지 않는 데이터로 보고 대조했습니다. 파일 경로는 `repo/` 기준입니다.

## 1. 코드 대조 (D1–D12)

| ID | 논문 | 코드 (`repo/`) | 판정 |
|---|---|---|---|
| D1 | 식 (1)의 $\varepsilon$, $\varepsilon^r$은 독립 (Section 4.3). "output-equivalent at initialization"(Section 4) | 학습: `src/openpi/models/pi0.py` L637–650에서 `noise_rng`와 `reac_noise_rng`를 따로 뽑음. 추론: `src/openpi/policies/policy.py` L243–244, `pi0.py` L742–746도 별도 노이즈 | 학습과 추론 모두 일치. **초기화 동일성은 같은 노이즈에서만 샘플 수준으로 성립**(토이 C). 논문 표현은 분포 수준으로 읽어야 함 |
| D2 | 단일 프레임 힘 ablation ("single-frame force input", Section 5.1). Q2 해석: "순간 노이즈에 잘 속는다" | `src/openpi/models/pi0.py` `make_cross_attn_mask`: `use_force_history=False`이면 **k==0만 허용**. 설정 스크립트 `scripts/train_online_dagger_lerobot_reactive_no_force_history.sh`의 주석: "permits only the first force-memory token (k == 0)... does not select each action's latest force" | **불일치 가능성.** 코드의 단일 프레임 기준선은 청크 시작 시점의 힘 1개를 쓰며, 현재 힘이 아님. 기준선이 "현재 힘 한 프레임"보다 약하게 설정되었을 수 있음 |
| D3 | 힘 메모리는 "recent force history" 또는 "short memory window" (Section 5.3, Algorithm 1에서 청크마다 `F ← []`) | 관측 `wrench`의 길이는 `action_horizon`(=10)로 고정. `_get_wrench_memory`는 GRU를 그 창에 한 번 돌림. 마스크 행렬 크기도 H×H | 확인. 메모리는 **최대 10 프레임(1초), 청크 단위로 리셋** |
| D4 | Table 2: AdamW(β 0.9, 0.95, ε 1e-8, weight decay 1e-10), 학습률 5e-5, 10k 워밍업, 30,000 스텝, 배치 32, 클리핑 1.0 | `src/openpi/training/config.py` L505–560 (book, hanoi, towel reactive 설정): batch 32, 30k 스텝, peak_lr 5e-5, warmup 10k, decay 1e6, clip 1.0. `src/openpi/training/optimizer.py`: b1 0.9, b2 0.95, weight decay 1e-10 | 확인. ε는 기본값 확인 안 함 |
| D5 | Table 2에 EMA 언급 없음. 평가 방식 불명 | 모든 reactive 설정에 `ema_decay=0.999` | **논문 표에 없는 설정.** 평가 체크포인트가 EMA인지 불명 (확인 필요) |
| D6 | 초기화: "Starting from pretrained π0.5"(Section 4.4), Table 1 "Copied from base action expert" | Hanoi 설정(`pi05_iPhoneSingle_hanoi_v1_100_reactive`)은 `gs://openpi-assets/checkpoints/pi05_base/params`에서 로드. Book 설정은 `$OPENPI_CHECKPOINT_ROOT/30000/params`(자체 1단계 체크포인트)에서 로드 | **과제별 초기 체크포인트가 다름.** 비교군(π0.5 w/ Online DAgger)의 초기화는 논문에 없음 (확인 필요) |
| D7 | 주요 실험은 고정 1:1 혼합, 0:1·1:2는 부록 K.1 | 반응 학습 스크립트 `train_online_dagger_lerobot_reactive.sh`: `min_online_ratio=max_online_ratio=0.5`(환경변수 기본값). `online_data_fetcher.py`의 기본 샘플러는 0.2–0.8 적응형 | 확인 (LIFT 스크립트 기준). 기본 샘플러는 LIFT가 아닌 경로 |
| D8 | 온라인 데이터는 사람 개입 프레임만 (Appendix C, E) | `--online-intervention-only` (반응 스크립트 L164), `control_flag == -1` 필터 (`data_loader.py`) | 확인 |
| D9 | 추론 시 두 스트림을 계산하고 반응 행동만 보냄 (O3.1) | `pi0.py` `sample_actions`: `return x_0 if self.original_head else reac_x_0` | 확인 |
| D10 | 지연 정렬 L=3 (Table 1, Appendix A) | `cross_attention_latency=3` (config.py의 모든 reactive 설정), 마스크 `k <= i - s3` | 확인 |
| D11 | 제어 10 Hz, 힘 1000 Hz 이상을 10 Hz로 내림 (Appendix D) | 전처리 YAML `expected_fps` 10, `fps=10`(config.py L444). 힘은 `left_wrench` 열 | 확인 |
| D12 | 데이터 공개: v1은 "code and data", v3는 "code" | 저장소에 데이터셋 없음. 변환 YAML만 있음. 변환 스크립트는 외부 SDK `nmx_nedf_api`가 필요(README). 로봇 드라이버와 TDK는 저장소 밖(README) | **데이터와 수집 도구 미공개.** 재현의 가장 큰 장벽 |
| D13 | 힘 OOD 처리 (Figure 1, 서론) | 분포 이동을 측정하는 코드 없음. 온라인 DAgger 루프만 있음 | 힘 OOD는 정량 측정이 없음 (논문 주장으로만 읽음) |

추가로 코드에만 있는 기능 (논문 미기재):
- `sample_actions_rtc`, `reactive_replan`, `pinv_corrected_velocity_reactive`: 실시간 청크 보정(RTC) 경로가 반응 스트림에도 있습니다. 실험에 쓰였는지는 논문에 없습니다(확인 필요).
- `pi05_iPhoneSingle_*_residual`, `residual_policy.py`: 잔차 기준선 구현이 저장소에 있습니다. 논문 Appendix H의 28M 파라미터 규모는 확인하지 않았습니다.
- `scripts/revalidate_lift*.py`, `train_online_dagger_lerobot_reactive_ratio_*.sh`: 재검증과 혼합비 실험 스크립트가 있습니다. 결과 파일은 없습니다.

## 2. 주장 검증 (C1–C16)

| ID | 주장 (출처) | 근거 | 판정 |
|---|---|---|---|
| C1 | 초기화 시 출력 동일 (초록, Section 4.2, O2.1–O2.2) | 토이 C 같은 노이즈 8.9e-16; 독립 노이즈 최대 차 5.3 (`03` §3). 코드 D1 | **조건부 확인.** 같은 입력에서만 성립. 독립 노이즈에서는 분포 수준 |
| C2 | shifted causal mask로 $r_i$가 $a_i$와 같은 문맥을 봄 (Figure 3) | 토이 C 마스크 검사 통과, 코드 `make_reactive_attn_mask`와 인덱스 일치 | 확인 |
| C3 | 교차어텐션 출력 0 초기화 → 0 스텝 잔차 0 (O2.2) | `CrossAttention`의 `cross_attn_vec_einsum`에 `init_fn=nn.initializers.zeros` (`gemma.py` L302) | 확인 |
| C4 | 힘 지연 정렬로 "그때 쓸 수 있는 힘"만 사용 (O1.2) | `make_cross_attn_mask`의 `k <= i-s3-1` 또는 `i-s3` (`pi0.py` L117–126) | 확인. 최신 힘은 3 스텝(300 ms) 과거 |
| C5 | 힘 입력은 "high-frequency"이며 빠른 반응을 가능하게 함 (서론) | 센서 1000 Hz이지만 10 Hz로 내려 씀, 반응 루프 10 Hz (D11, `03` §4) | **동기와 불일치.** 실제 반응 주기는 100 ms, 지연 하한 300 ms |
| C6 | 수건: LIFT 피크 0.825 > 온라인 DAgger 0.725 (Section 5.2) | 수치는 본문 그대로. SD 0.2–0.5에서 p=0.05–0.44 (`03` §2.3) | **수치 확인, 차이는 비유의**(SD 0.3 이상) |
| C7 | 책: LIFT 피크 0.6 > 0.4 (5.2) | p=0.0001 (SD 0.2) – 0.12 (SD 0.5) | **조건부** (SD 가정, 피크 선택, 시드 없음) |
| C8 | 하노이: 피크 0.6 > 0.3, 최종 0.6 > 0.2 (5.2) | Fisher p=0.037 (피크), p=0.003 (최종). 롤아웃 기준 | **확인(롤아웃 기준).** 학습 시드 없음 |
| C9 | 힘 메모리가 단일 프레임보다 낫다 (Q2) | 하노이 피크 0.6 vs 0.5, p=0.60 (**비유의**); 최종 0.6 vs 0.1, p=0.0001. 책 피크 0.6 vs 0.4 (확인 보류). 수건에서는 단일 프레임이 0.95로 LIFT 0.825보다 높음 | **부분 확인.** 최종 점수만 확인. 기준선이 청크 시작 프레임(D2) |
| C10 | 온라인 교정 데이터가 필요하다 (Q4) | Offline DAgger: 책 0, 하노이 약 0.23, 수건 약 0.63 (Figure 5 판독) | **책·하노이에서 확인**(그림 판독). 수건에서는 차이가 작음 |
| C11 | 일반화에서 큰 저하 없음 (서론, 5.4) | Figure 6 라벨: 책 "Another desktop" 0.5→0.1, 하노이 "Another desktop" 0.6→0.3 (`03` §2.4) | **반박됨**(두 조건). 라벨이 부록 L과 맞지 않음 |
| C12 | 샘플 효율: "reaches 0.65 with fewer samples", "learns faster" (5.2, 서론) | 같은 점수에서 비교하지 않고 다른 샘플 수의 점을 비교. 점 간격이 0.5K 이상 | **부분 확인.** 같은 조건의 곡선 비교 없음 |
| C13 | 잔차 기준선이 크게 낮다 (Q5, Figure 7) | 기준선 최종 점수 약 0.1 이하, LIFT 곡선 없음, x축 5k–20k 스텝 vs LIFT 30k | **부분 확인.** 그림만 있고 같은 축에서 비교하지 않음 |
| C14 | 95% CI, n=30 (5.1) | 롤아웃 표본만. 학습 시드 부재 (`03` §1.9) | **한계.** CI는 실험 간 변동을 빠뜨림 |
| C15 | 초록: "higher peak and final performance than vision-only DAgger" (3 과제 모두) | 하노이 최종 확인, 책 피크 조건부, 수건 최종 값은 그림에서 판독 불가 | **부분 확인** |
| C16 | Figure 1 "average peak score"는 Figure 5 피크와 같은 값 (암묵) | 수건 +0.008, 책 -0.033, 하노이 -0.033 차이 (`03` §1.1) | **불일치.** 정의가 없음 |
| C17 | 코드 공개 (초록) | 공식 저장소(Apache-2.0) 있음. 학습·추론 코드 포함. 데이터와 TDK 없음 | **확인(코드만).** 데이터는 공개되지 않음 |
| C18 | 하이퍼파라미터 (Table 1–2) | D4, D10 일치. 단 EMA 0.999 추가 (D5) | **확인**(EMA 제외) |
| C19 | 학습 한 번 8×A800, 온라인 실험 2–3시간 (Appendix G, J) | 보고만 있음. 저장소에 시간 기록 없음 | **검증 불가**(보고) |

## 3. 저자에게 물을 질문

1. 단일 프레임 기준선은 청크 시작의 힘 한 프레임(k=0)입니까, 아니면 현재 힘 한 프레임입니까? 코드는 전자입니다(D2).
2. Figure 1의 "average peak score"는 무엇을 평균한 값입니까? 본문 피크와 다른 이유는 무엇입니까(C16)?
3. 평가 횟수 10에서 30으로 바뀌었는데 본문 수치는 그대로입니다. 같은 롤아웃을 재집계한 것입니까, 새로 측정한 것입니까?
4. 학습 시드는 몇 개입니까? 방법마다 한 번씩 학습했다면 CI는 롤아웃 변동만 반영합니다.
5. 일반화 표(Figure 6)의 "Another desktop", "Another book"은 부록 L의 어떤 조건에 해당합니까? 책과 하노이에서 "Another desktop"의 큰 저하는 어떻게 설명합니까?
6. 과제마다 초기 체크포인트가 다릅니다(D6). 비교군 π0.5 w/ Online DAgger는 같은 체크포인트에서 시작했습니까?
7. 평가 체크포인트는 EMA(0.999) 가중치입니까(D5)?
8. 힘 데이터를 10 Hz로 내리는 이유와, 1000 Hz 원신호를 쓰는 대안을 시험했는지 알고 싶습니다(C5).
9. 데이터는 공개할 계획이 있습니까(v1 문장의 변경 이유)?
10. Residual Policy 기준선의 파라미터 수(28M)와 학습 스텝 범위는 LIFT와 같은 조건입니까(C13)?

## 4. 구현 가능성 (이 맥 기준)

**판정: 구조와 마스크, 초기화 규칙은 구현할 수 있습니다(중간 노력, CPU 소형 검증까지 가능). 논문 수치 재현과 π0.5 규모 학습은 이 맥에서 불가능합니다. 실로봇 재현은 저장소 밖 장비와 데이터가 있어야 합니다.**

기준 환경: Apple M4 Pro, 통합 메모리 약 24 GiB, MPS 사용 가능, CUDA 없음. 로컬 가상환경에 torch와 mujoco가 설치되어 있습니다.

| 작업 | 이 맥 | 필요 자원 | 판정 |
|---|---|---|---|
| 코드 정적 읽기와 대조 (D1–D13) | 가능 | 없음 | **완료** |
| 마스크 규칙과 초기화 동일성 토이 (C1–C4) | 가능 | numpy | **완료** (`03` §3) |
| 공식 코드 실행 (openpi, 학습 또는 추론) | 불가 | 리눅스, CUDA GPU. README가 명시. JAX 설치와 실행은 승인 대상 | **불가 (이 맥)** |
| π0.5(약 2.5–3B, 추정)에 반응 expert를 붙여 추론 | 조건부 | 체크포인트 다운로드(금지, 가중치는 받지 않음), 메모리 | 불가(가중치 다운로드 승인 전) |
| 학습(방법당 8×A800, 30k 스텝) | 불가 | GPU 8장 이상 | **불가** |
| 실로봇 재현(Flexiv Rizon 4S, 6D 힘 센서, TDK, iPhone) | 불가 | 장비 전체 | **불가** |
| 소형 PyTorch 재구현(반응 expert, 힘 GRU, 0 초기화 교차어텐션, 마스크) | 가능 | MPS 또는 CPU | **가능** (소형 BC 정책 기준) |
| MuJoCo 접촉 과제에서 힘 메모리와 지연 L 비교 | 가능 | mujoco, torch (로컬 가상환경) | **가능** (소형 과제 기준) |

**구현 노력 (추정):**
- 소형 재구현(BC 정책 + 힘 GRU + 반응 스트림 + 마스크 단위 테스트): 약 300–500줄, 3–5일.
- 소형 실험(MuJoCo, 3 시드, 지연 비교): 1주 안팎.
- 논문 규모 재현(π0.5, 실데이터, 실로봇): 데이터 수집과 GPU 자원이 필요해 수 주에서 수 개월(추정). 데이터가 공개되지 않아 자체 수집이 필수입니다.

**가장 큰 장벽:** (1) 데이터와 TDK 미공개(D12), (2) GPU 학습, (3) 6D 힘 센서와 개입 운영자가 필요한 교정 루프.

## 5. 라이선스와 권리

- **논문 (arXiv:2607.14236):** abs 페이지의 license 링크는 `creativecommons.org/licenses/by-sa/4.0/`입니다(2026-11-14 확인). **CC BY-SA 4.0**이며 CC BY가 아닙니다. 논문 그림은 **개인용 리포트와 PPT에만** 넣었습니다. 공개본(`report_public.pdf`, `publish/`)에는 넣지 않았습니다.
- **PDF의 권리 문구:** 1쪽에 arXiv 스탬프와 CoRL 2026 표기만 있습니다. 재인쇄 금지 문구는 없습니다.
- **공식 코드 (y-wng/LIFT):** Apache-2.0 (`LICENSE`). `LICENSE_GEMMA.txt`(Gemma Terms of Use, 2024-02-21 최종 수정)가 함께 있습니다. 모델 가중치는 Gemma 약관을 따를 가능성이 큽니다(추정, 확인 필요). 가중치는 받지 않았습니다.
- **openpi (Apache-2.0)와 Flexiv TDK (Apache-2.0):** 기반 및 도구 코드입니다. 이 분석에서는 쓰지 않았습니다.
- **공개 범위:** 공개본에 논문 그림, 원문 PDF, 공식 코드 사본(`repo/`)은 넣지 않습니다.

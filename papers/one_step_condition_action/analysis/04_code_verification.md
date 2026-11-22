# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- 저장소: `NTUMARS/UCA-Policy` (공식 구현). 프로젝트 페이지의 Code 링크는 옛 저장소 주소를 가리키며, GitHub에서 위 저장소로 이름이 바뀌어 넘어갑니다(API 응답의 `full_name`). README의 설치 명령도 옛 주소를 씁니다.
- 커밋 `40084a903d03417a4d32b09506c0a48a11b2dba4` (2026-10-23, "Correct reported success and latency aggregates and motivation figure"). 얕은 복제를 `repo/`에 두었습니다(137 MB, gitignore).
- **정적 읽기만 했습니다.** 코드를 실행, import, 설치하지 않았습니다. 체크포인트, 데이터, 가중치는 저장소에 없습니다(`.pt`, `.ckpt`, `.zarr`, `.npz` 파일 검색 결과 0개).
- 읽은 파일: `README.md`, `install.md`, `requirements.txt`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, `UCA/uca/config/{ucaflow,uca,uca_real}.yaml`, `UCA/uca/config/task/{adroit_door,adroit_hammer,metaworld_*,real_*}.yaml`(형상과 에피소드 항목), `UCA/uca/policy/{ict1policy,dit1policy,ditx1policy}.py`, `UCA/uca/model/mean/ict.py`, `UCA/uca/model/mean/ditx.py` (교차 어텐션 블록), `UCA/uca/model/mean/dit.py`(AdaLN), `UCA/train.py`(체크포인트 선택과 롤아웃 주기), `UCA/eval.py`, `scripts/{train_policy,eval_policy,train_ucaflow_new4,gen_demonstration_metaworld,gen_demonstration_adroit,train_uca_real_h16}.sh`.
- 판정 기호: **일치**는 논문과 코드가 같음, **차이**는 다름, **미기재**는 논문에 없어 코드 값만 있음, **확인 불가**는 자료가 없어 판정하지 못함입니다.

## 1. 저장소 상태와 라이선스

- 라이선스: 루트 `LICENSE`는 MIT이고 저작권 줄은 "Copyright (c) 2026 UCA-Flow Authors"입니다. 저장소 설명과 API의 `license`도 MIT입니다. 재사용한 코드는 원래 라이선스를 유지합니다(`THIRD_PARTY_NOTICES.md`). MP1에서 온 코드는 `LICENSES/MP1-MIT.txt`에 MIT 문구가 있습니다. `third_party/`에는 `rrl-dependencies`와 `mjrl` 등 외부 코드가 들어 있습니다.
- 이력: 커밋 두 개입니다. 2026-10-12 "Initial UCA-Flow release with paper figures and license notices", 2026-10-23 집계 정정. API의 `created_at`은 2026-08-19이며, 이력은 새로 만든 것으로 보입니다(`THIRD_PARTY_NOTICES`가 "새 Git 이력"을 언급합니다).
- 별 3개, 포크 0개, 마지막 push 2026-10-23, 아카이브 아님.
- 체크포인트와 데이터는 공개되지 않았습니다. 데이터 생성 스크립트만 있습니다(`scripts/gen_demonstration_*.sh`).

## 2. 논문 ↔ 코드 대조 (D 항목)

| 번호 | 항목 | 논문 | 코드 (커밋 40084a9) | 판정 | 영향 |
|---|---|---|---|---|---|
| D1 | 학습 에폭 | §4.1: Adroit 3,000, Meta-World 1,000 에폭 | 공식 설정 `ucaflow.yaml`의 `num_epochs: 301`. 시뮬레이션용 에폭 오버라이드는 `train_policy.sh`에 없음 | **차이** | 공개 설정으로 시뮬을 돌리면 논문보다 짧게 학습함(약 3–10배). 논문 값을 쓴 설정 파일은 저장소에 없음 |
| D2 | Adroit 점군 수 | §4.1: FPS 1024점 (Adroit), 512점 (Meta-World) | Adroit 과제(`adroit_door.yaml` 등)의 `point_cloud` 형상 `[512, 3]`. Meta-World도 512. 실로봇만 1024 | **차이** | Adroit 입력이 논문보다 절반. 공식 설정이 논문 본문과 다름 |
| D3 | 체크포인트 선택과 보고 | §4.1: 평가 매 200 에폭, **상위 5개 체크포인트의 평균** (그림 5 캡션: 시드별 top-5 평균) | `ucaflow.yaml`의 `checkpoint.topk.k: 1`, `monitor_key: test_mean_score`(환경 롤아웃 점수). 롤아웃은 `rollout_every: 200`에만 수행됨 (`train.py` L272) | **차이 + 선택 편향** | 설정은 체크포인트 하나만 남깁니다. 선택 지표가 평가 롤아웃과 같으므로, 같은 점수로 고른 체크포인트를 같은 점수로 보고하면 낙관적입니다. 평가 전용 분할이 없습니다 |
| D4 | 이중 손실 가중 | 식 11: $L = ρ(V−sg(v)) + ρ(v_θ−sg(v))$ | `loss = loss_u + loss_v` (`ict1policy.py`, `dit1policy.py`, `ditx1policy.py` 모두) | **일치** (가중 1:1) | — |
| D5 | 적응 가중 ρ | 식 12: $ρ(Δ)=sg((‖Δ‖²_2+c)^{−(1−γ)})‖Δ‖²$, ‖·‖는 호라이즌과 행동 차원 전체 | `adaptive_l2_loss(error, gamma=0.5, c=1e-3)`. 제곱 오차를 **평균**(`torch.mean`)으로 계산(합이 아님). 논문에 $c$, $γ$ 값 없음 | **미기재 (값)**, **차이 (노름)** | 합 대 평균은 $c$의 실효 크기를 호라이즌×차원 배만큼 바꿈(Adroit 112, 실로봇 16×차원) |
| D6 | 구간 샘플링 | 본문에 없음. "$t \ge r$" 만 적음 | `time_dist = ['lognorm', −0.4, 1.0]`, `t = max(s1, s2)`, `r = min(s1, s2)`. `flow_ratio = 0`이라 $r=t$(순간 속도 경계)는 학습에서 일부러 뽑지 않음. 단 두 번째 통과의 $r=t$는 별도 순전파로 있음 | **미기재** | 샘플 분포가 결과에 큼. 재현하려면 명시해야 함 |
| D7 | JVP 방향 | 식 9: $(sg(v_θ), 1, 1)$ 방향, $D_t u = ∇_z u·sg(v_θ) + ∂_t u + ∂_h u$ | 코드는 $(z, t, r)$에서 $(sg(v_θ), 1, 0)$으로 `torch.autograd.functional.jvp`를 호출. $h = t − r$이므로 $r$을 고정하면 $dh/dt = 1$으로 식 9와 같은 방향 | **일치 (동치)** | 식 9의 $\partial_h$ 표기는 $(t,h)$ 좌표계 기준. 구현은 $(t,r)$ 좌표계 |
| D8 | 토큰 구조와 어텐션 | 식 3–8: 조건, 타임스텝, 구간, 행동 토큰을 한 열에서 셀프 어텐션, 출력은 행동 토큰 | `ict.py` `DiTModel.forward`: 접두 토큰(타임스텝 $N_t=1$, 구간 $N_h=1$, 관측 2) + 행동 토큰(호라이즌 4)을 이어 붙이고 `attn_mask=None`으로 모든 블록 통과, 출력은 `seq[:, prefix_len:]` | **일치** | 논문은 $N_t$, $N_h$ 값을 적지 않음 (**미기재**: 공식 설정은 둘 다 1) |
| D9 | 모델 크기 | 파라미터 수, 층 수, 폭 없음 ("compact model") | `ucaflow.yaml`: `n_layer 8`, `n_head 8`, `n_emb 384`, `encoder_output_dim 64`. 블록 파라미터 약 8 × 12 × 384² ≈ 1,420만 (추정, 인코더 제외) | **미기재** | "compact"를 판단할 근거가 없음. DP3 등과 파라미터 비교 자료도 없음 |
| D10 | 옵티마이저와 스케줄 | AdamW, 배치 128, 학습률 1e-4 | AdamW 베타 (0.95, 0.999), 가중치 감쇠 1e-3, 코사인 스케줄, 웜업 500 스텝, 기울기 클리핑 1.0, EMA(감쇠 0.9999, 평가는 EMA 모델) | **일치 (배치, 학습률)**, **미기재 (나머지)** | EMA 평가 여부는 논문에 없음 |
| D11 | 추론 행동 구간 | 호라이즌 4, 한 번 추론 | `n_action_steps: 3` (시뮬), 추론은 `action_pred[:, To−1 : To−1+3]`, 즉 호라이즌 4 중 1–3번 행동 3개를 실행 | **미기재** | 논문은 실행 행동 수를 적지 않음. 실로봇은 호라이즌 16 중 15개 |
| D12 | 한 스텝 추론식 | §3.1: $x̂ = z_1 − u_θ(z_1, 1, 1, o)$ | `z = z − model(sample=z, timestep=1, r=0)` (`ict1policy.predict_action`) | **일치** | $h = 1$은 $r = 0$과 같음 |
| D13 | 절제 변형의 대응 | §4.2: −Unified/c(교차 어텐션), −Unified/a(AdaLN), −DualPass | 교차 어텐션: `ditx1policy.py` + `ditx.py`(`PureTransformerBlock`에 교차 어텐션 블록). AdaLN: `dit1policy.py` + `dit.py`(`modulate_rmsnorm`, `GlobalCondEncoder`). −DualPass: **전용 설정 없음**. 세 정책 모두 `loss_u + loss_v` | **일치 (두 변형), 확인 불가 (−DualPass)** | −DualPass를 재현하려면 $v$ 항을 끄는 설정을 직접 만들어야 함. 세 변형의 학습 예산과 파라미터 수가 맞는지 확인 불가 |
| D14 | 실로봇 설정 | §5.1: 호라이즌 16, 관측 간격 2, 300 에폭, 시연 40개, 점 1024 | `uca_real.yaml`: 호라이즌 16, 관측 2, `n_action_steps: 15`, 301 에폭, 점 1024 (`real_*.yaml`) | **일치 (대부분)**, **미기재 (실행 행동 15)** | 논문은 실행 행동 수를 적지 않음 |
| D15 | 실로봇 과제 목록 | §5: 2개 과제 (Pick & Place, Drawer Close) | `task/`에 `real_pick`, `real_close`, `real_stack`, `real_stack_qube`, `*_rebuttal`, `*_old` 등 9개 이상. 논문 표 4–5의 과제와 대응하는 설정은 추정만 가능 | **확인 불가** | 대응 설정이 `real_pick`과 `real_close`로 보이나 증명할 자료 없음. "rebuttal" 설정이 논문 뒤에 추가된 것으로 보임 |
| D16 | 시연 수 | §4.1: 과제당 시연 10개 (시뮬) | Adroit 생성 스크립트 `--num_episodes 10`. Meta-World 생성 스크립트 `--num_episodes 20`. 학습 설정의 `max_train_episodes: 90`은 상한이고 실제 개수는 데이터(미공개)에 따름 | **Adroit 일치**, **Meta-World 확인 불가** | Meta-World에서 20개를 생성해 10개를 쓰는지 확인 필요 |
| D17 | 평가 에피소드 | 본문 미기재 (표 4–5는 실로봇 20회) | `eval_episodes: 20` (시뮬 각 과제), 실로봇 20회 | **미기재 (시뮬)**, **일치 (실로봇)** | 시뮬 성공률의 표준오차를 계산하려면 과제별 시도 수가 필요함 |
| D18 | 기준선 수치 출처 | 표 2 캡션: "Baseline summaries are retained as reported" | 저장소에 `dp3.py`, `simple_dp3.py`, `flowpolicy.py`, `meanpolicy.py`(MP1)와 설정이 있어 같은 코드로 재실행 가능. 그러나 논문은 "as reported"라고 함. DP3 원논문(arXiv v7)은 과제 묶음이 Easy 28, Medium 11, Hard 6, Very Hard 5로 다릅니다. 본 논문의 묶음(21/4/4/5, 34과제)은 MP1 계열 묶음으로 보임 | **확인 불가** | 기준선 값을 어느 논문과 어느 프로토콜에서 가져왔는지 본문에 없음. 원출처를 확인하기 전에는 기준선 차이를 해석하기 어려움 |
| D19 | 지연 측정 | §4.1: "안정된 GPU 부하에서 시드 세 개를 동시에 실행하며 측정" | 측정 스크립트는 찾지 못함. `env_runner.detail_timing: true`와 `scripts/compute_flops.py`(FLOPs)가 있음. 측정 절차는 미공개 | **확인 불가** | 동시 실행은 지연에 영향을 줌. 절대값 비교가 어려움 |
| D20 | 라이선스 | — | MIT (루트 LICENSE). 재사용 코드는 각자 라이선스 유지 | **일치** | 코드를 내 저장소에 가져갈 때는 `THIRD_PARTY_NOTICES`를 함께 봐야 함 |
| D21 | 설치 조건 | 논문: "RTX A5000 한 장" | `install.md`: Python 3.8, torch 2.2.1 (cu118), mujoco 2.1.0 바이너리와 mujoco-py 2.1.2.14. `requirements.txt`: CUDA 12 휠(`nvidia-*-cu12`)과 저자 포크의 metaworld, mujoco-py를 특정 커밋으로 고정. 두 파일의 CUDA 버전도 다름 | **차이 (설치 문서 간)** | CUDA 전제와 구버전 mujoco-py 때문에 이 맥에서 설치하기 어려움(`§4`) |

요약: 구조(D7, D8, D12, D4)와 학습 방정식은 논문과 맞습니다. 차이는 학습량(D1), 점군 수(D2), 체크포인트 선택(D3)에 있고, 셋 다 논문 본문 수치를 바꿀 수 있습니다. 논문이 적지 않은 값(D5, D6, D9–D11)은 재현하려면 직접 정해야 합니다. 기준선 출처(D18)와 지연 측정(D19)은 확인하지 못했습니다.

### 2.1 논문 주장과 코드의 관계

- **주장 "조건 토큰이 행동과 함께 갱신된다" (§3.3).** 코드의 `DiTModel`이 이를 구현합니다(D8). 셀프 어텐션은 모든 토큰이 모든 토큰을 보도록 열려 있어(`attn_mask=None`), 조건 토큰이 층마다 행동 토큰의 정보를 받습니다.
- **주장 "조건을 외부 주입하는 기준선보다 낫다" (§4.2).** 대조 변형(`ditx1`, `dit1`)이 저장소에 있어 같은 학습 루프로 비교할 수 있습니다(D13). 파라미터 수와 학습 예산이 같은지는 설정 파일로 확인할 수 있으나 이 분석에서는 세부 값을 비교하지 않았습니다(다음 작업 후보).
- **주장 "한 번의 추론으로 충분하다" (§3.1).** 추론은 순전파 한 번입니다(D12).

## 3. 기준선 구현 (저장소 내)

`uca/policy/`에는 DP3(`dp3.py`), Simple DP3(`simple_dp3.py`), FlowPolicy(`flowpolicy.py`), MP1(`meanpolicy.py`), ACT, pi0, DiT 계열이 들어 있습니다. 따라서 논문 표 2–3의 기준선은 같은 저장소의 코드로 재실행할 수 있는 구조입니다. 다만 논문 본문은 기준선 수치를 "as reported"라고 적었습니다(D18). 재실행 결과와 보고값이 같은지는 확인하지 않았습니다.

## 4. 구현 가능성 (이 맥)

**맥 환경:** Apple M4 Pro, 통합 메모리 약 24 GiB, MPS 있음, CUDA 없음.

**판정: 핵심 모듈은 이 맥에서 구현할 수 있습니다. 공식 코드의 설치와 전체 수치 재현은 이 맥에서 어렵습니다. 실로봇 재현은 불가입니다.**

| 수준 | 가능성 | 근거 |
|---|---|---|
| 핵심 모듈(통합 토큰 열, 이중 손실, 한 스텝 추론)의 PyTorch 재구현 | **가능** (노력 중간, 약 1주, 추정) | `ict.py` 258줄, `ict1policy.py`의 손실 블록. MPS에서 돌아감. 측정 전 |
| 공식 코드 설치 (맥) | **어려움** | Python 3.8, torch 2.2.1 cu118, mujoco-py 2.1.2.14(MuJoCo 2.1 바이너리)는 맥 arm64에서 빌드 실패 가능성이 큼(추정, 설치 미시도). `requirements.txt`는 CUDA 12 휠과 포크 커밋을 고정 |
| 시뮬 한 과제의 짧은 학습 확인 (MPS) | **조건부** | 시연 10개, 배치 128, 301 에폭이면 수천 스텝 규모. 시간은 측정 전 추정. 승인 후 가능 |
| 37과제 × 3시드 × 3,000 에폭 재현 | **불가** (이 맥에서) | 111회 학습. 1회에 수 시간(추정)이므로 수 주 이상 |
| 실로봇 (UR5e, L515) | **불가** | 장비, 제어 브리지, 데이터 필요. 저장소 밖 |
| 기준선 재실행 (DP3, MP1, FlowPolicy) | **조건부** | 같은 설치 문제. 보고값 출처(D18)를 확인하기 전에는 비교의 의미가 제한적 |

**노력 추정 (측정 전):**
- 핵심 모듈 재구현과 MetaWorld 한 과제 학습 확인: 1–2주.
- 공식 코드를 맥에서 설치하고 한 과제를 돌림: 3–5일(설치 실패 가능성 포함).
- 전체 재현: 이 맥에서는 현실적이지 않음.

## 5. 재실행

이 문서의 검사는 파일 읽기로 끝납니다. 재실행할 스크립트는 없습니다. 공식 저장소를 다시 얻으려면 다음 명령을 쓰되 실행하지는 마십시오.

```bash
git clone --depth 1 https://github.com/NTUMARS/UCA-Policy.git repo
```

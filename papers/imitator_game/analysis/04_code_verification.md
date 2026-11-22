# 04. 공식 코드 대조와 구현 가능성

대상: `imitator-game/The-Imitator-Game` 커밋 `3193957e3b0187d086e44d79e0e4af9845fbd4fd` (2026-11-09, MIT). 얕은 복제(`repo/`, gitignore)를 **정적으로만** 읽었습니다. 실행, import, 설치를 하지 않았고 데이터·가중치도 받지 않았습니다. 논문 v1(2026-08-23)과 코드 커밋(2026-11-09)은 시점이 달라서 코드가 논문 이후에 바뀌었을 수 있다는 점을 전제로 읽어야 합니다. 경로는 모두 `repo/` 기준입니다.

## 1. 구성

| 경로 | 내용 |
|---|---|
| `mani_skill/envs/tasks/tabletop/dual_tasks/` | 시뮬 과제 50개 + 템플릿(`_001`–`_051`). L3 변형은 `dual_tasks_l3/` 50개 |
| `mani_skill/envs/tasks/tabletop/utils/L0_L3_utils.py` | 레벨 스위치(L1 오프셋, L2 모델 교체·좌우 거울, L3) |
| `mani_skill/envs/tasks/tabletop/utils/reward_utils.py` | 하위 목표 피크 보상 추적(`RewardTracker`) |
| `examples/baselines/{act,diffusion_policy,vqbet,uniskill,xskill,gr00t,rdt,pi,openvla_oft}/` | 베이스라인 학습·평가 (논문의 9개 모델) |
| `examples/baselines/{zero_wam,astra}/` | 논문에 없는 추가 기준선 (Zero-WAM, Astra) |
| `examples/baselines/lerobot_dataset/` | 데이터 로더, 과제 매핑, 분할 설정, 캡션 JSON |
| `examples/baselines/exp_scripts/` | "논문의 모든 수치를 만드는" 런처 (README 주장) |
| `scripts/` | 데이터 수집·변환 |

Imitator Arena(웹 평가 플랫폼), 리더보드 집계 스크립트(`all_master_records.csv`를 만드는 파이프라인), 실제 로봇 데이터 수집 코드는 저장소에 없습니다.

## 2. 논문과 코드의 차이 (D1–D18)

판정 열: **차이**(논문 서술과 코드가 다름), **일치**, **확인 불가**(저장소에 없음).

| 번호 | 대상 | 논문 (절) | 코드 (파일: 줄) | 판정 |
|---|---|---|---|---|
| D1 | ACT 학습 | 100 에폭, 배치 256, 영상 4프레임 (F.7) | 런처: 에폭 **10**, 배치 **128**, 프레임 **10** (`exp_scripts/act/run_exp_act.sh`: 25, 29–30, 마지막 세 호출의 에폭 인자 10; README 기준 명령도 동일). 데이터클래스 기본값만 100/256/4 (`act/train_act_imitator.py`: 83, 85, 131) | **차이** (기본값은 논문, 런처는 다름) |
| D2 | Diffusion Policy | 100 에폭, 행동 지평 16, 배치 256 (F.7) | 런처: 에폭 **10**, 프레임 10, U-Net 폭 `352 704 1408 2816` (`exp_scripts/diffusion_policy/run_exp_dp.sh`). `pred_horizon` 16이지만 `act_horizon` 4 (`diffusion_policy/train_dp_imitator.py`: 104–105) | **차이** |
| D3 | VQ-BeT | GPT 100 에폭, 배치 256 (F.7) | VQ-VAE 100 에폭·배치 256은 일치, **GPT 200 에폭, 배치 32**, lr 1e-4 (`exp_scripts/vqbet/run_exp_vqbet.sh`: 24–25, 51–52) | **차이** |
| D4 | RDT-1B, π0.5 에폭 | "모든 사전학습 10 에폭" (F.4, F.5), "matched compute" (B.4) | RDT와 π0.5는 **45 → 10, 30 → 15, 15 → 30 에폭** (`exp_scripts/rdt/run_exp_rdt.sh`, `exp_scripts/pi/run_exp_pi.sh` 마지막 줄). 과제 수 x 에폭 = 450으로 일정. ACT, DP, UniSkill, GR00T는 규모와 무관하게 10 에폭 | **차이.** 두 모델은 코퍼스가 작을수록 더 많이 학습해 "규모 효과"가 일정 연산에서 측정됨. 나머지는 연산이 규모에 비례 |
| D5 | OpenVLA | 사전학습 배치 256 (F.3) | 배치 **512**, effective 512, lr 1e-4, 10 에폭 (`exp_scripts/openvla/run_exp_openvla_cache_45.sh`: 26–27, 36). **15와 30 규모 런처 없음**(45만) | **차이** + 런처 누락 |
| D6 | VLA 캡션 T(V) | "고정, 결정적 캡셔너"이며 Qwen3-VL에 단일 프롬프트 템플릿, 모델별 조정 없음 (§3.2.2, B.4) | 영상이 아니라 **과제 단위의 미리 만든 JSON**(`lerobot_dataset/task_desc/human_desc.json`, 53과제)을 쓰고 과제마다 **53–65개의 패러프레이즈**가 있음. 학습과 평가 모두 `random.choice`로 뽑음 (`lerobot_dataset/evaluate_processor.py`: 455) | **차이.** 결정적이 아니고 영상별이 아님. 사람 작성 + 증강이라는 B.1 서술과는 부합 |
| D7 | PourKettle 성공 판정 | 성공 = 액체량 ≥ τ_fill ∧ 쏟은 양 ≤ τ_spill, τ = 20, 하위 단계 M = 4 (reach, grasp, transport, pour) (B.3 사례) | **액체 시뮬 없음.** `success = self.kettle_ever_poured`: 왼팔이 주전자를 쥔 채 컵 위(수평 거리 ≤ 0.16 + 0.01 m, 높이 > 0.08 m)에 **한 번이라도** 있으면 성공(래치). 기울임 판정(`is_tilted`)은 계산만 하고 쓰지 않음. 하위 단계 3개 `["reach","grasp","pour"]` (`dual_tasks/_020_pour_kettle.py`: 33, 233–263) | **차이** |
| D8 | SR 정의 | 종료 시점에 모든 목표 술어가 참 (B.3) | 평가 스크립트는 `success_once`(에피소드 중 한 번이라도 성공)와 `success_at_end`를 둘 다 기록하고, 요약 SR로 `success_once_mean`을 사용 (`act/eval_act_imitator.py`: 762, 771, 938). 과제 다수가 `*_ever_*` 래치로 성공 정의 (`_013`, `_015`, `_017`, `_020`, `_022`, `_024` 등) | **차이** (정의가 "한 번이라도"에 가까움) |
| D9 | Sub-SR | 순서 있는 M개 하위 목표 중, 피크 보상이 고정 임계값을 넘은 것의 비율 (B.3) | `RewardTracker.total()` = 단계별 **피크 보상(연속값)의 평균** (`utils/reward_utils.py`: 306–311). 임계값으로 세는 코드는 저장소에 없음 (집계 파이프라인은 비공개) | **차이 (공개 기록으로 확인)**: 프로젝트 페이지 기록의 SubSR는 연속값(2,767개의 서로 다른 값, `03` V12) |
| D10 | 궤적 유사도 지표 | "궤적 유사도 지표를 정의하지 않는다" (B.3) | 평가 스크립트에 TSS, nDTW, wTSS(= SR x TSS) 계산이 있음 (`act/eval_act_imitator.py`: 170, 229–236, 770–780 부근). 논문 표에는 안 나옴 | **차이** (보고 안 한 지표가 코드에 존재) |
| D11 | L1 정의 | 배치 재배열 "rearrangement" (§1, 표 1) | 모든 과제 공통 **고정 오프셋 (0.1, 0.1) m** (`utils/L0_L3_utils.py`: 19, 40–52). 매 에피소드 무작위 재배치가 아님(0–0.02 m 균일 섭동만 별도). 오프셋이 어떤 물체에 적용되는지는 과제마다 다름 (`_020_pour_kettle.py`: 197은 주전자만) | **차이(표현)** |
| D12 | L2 정의 | 같은 범주의 외형·형상 변경 (§1, 표 1) | 모델 ID 교체(`apply_l2_ycb_model_id`)에 더해 **좌우 거울 반사**(`is_lr_mirror_enabled()`는 L2에서 켜짐, 주전자 회전 변경 포함). 런처는 `--eval_lr_mirror auto --eval_lr_mirror_robot_pose false`(로봇 자세는 반사 안 함) (`utils/L0_L3_utils.py`: 86–112, README `mani_skill/README.md`) | **일치(부록 C.5가 팔 교체를 인정)**, 단 §3.1·표 1의 L2 정의에는 없음 |
| D13 | 10개 시연 | §4.1과 부록 F는 "과제당 10 시연", 부록 B.2는 "4개 레벨로 나눈 예산이 아님" | `sim_test_config_unseen.json`은 과제 5 x 레벨 4 = 20개 항목이 각각 `train: 0:10` → **과제-레벨 쌍당 10개, 과제당 40개**. 사전학습은 `0:50` (`sim_train_config_45.json`: 180항목) | **일치(B.2)**, §4.1·F 문구가 모호 |
| D14 | Scr와 P+FT | 표 3의 Scr, P+FT는 10개 시연으로 학습 (§4.1). Scr 집계 방식은 서술 없음 | `exp_scripts/`는 15/30/45 사전학습과 평가만 있고 **Scr, P+FT 학습 런처가 없음**(UniSkill 체크포인트 이름에 `unseen_finetune/scratch`가 있고 Zero-WAM에만 문서가 있음). README는 "모든 수치는 런처로 만든다"고 씀. 공개 기록(`03` V12)상 Scr는 변형 11개가 규모 표지 3회, 4개(GR00T, π0.5, RDT, OpenVLA)가 1회이며 표 3 값은 3회의 **셀별 최솟값** | **차이/확인 불가** (런처 없음, 집계 비공개) |
| D15 | 시연 영상 선택 | 같은 에피소드 쌍으로 A/B 비교 (§3.2.2) | 평가 때 영상은 `random.randint`로 **롤아웃마다 임의 에피소드**를 뽑음(`evaluate_processor.py`: 293–307), 시드 고정은 확인하지 못함 | **확인 불가** (같은 영상으로 짝지어지는지 코드로는 보장 안 됨) |
| D16 | 평가 모델 수 | 9개 모델, 15개 변형 (§3.2) | 코드에 Zero-WAM, Astra 추가(`README.md` 뉴스: 10-28, 11-07). 프로젝트 페이지는 "Nine models, sixteen variants" | **차이** (논문 이후 확장, 표 3에는 없음) |
| D17 | 체크포인트 | "논문에서 평가한 체크포인트 공개" (README) | HF API 목록에는 `uniskill/` 58개 파일만 보임 | **확인 불가** (다른 모델 체크포인트 미확인) |
| D18 | 학회 표기 | CoRL 2026 (공식 목록은 포스터) | README 뉴스는 CoRL 채택 + **NeurIPS ODI 워크숍 구두** | 일치 (구두는 워크숍) |

### 일치한 항목 (코드와 논문이 같은 값)

| 항목 | 값 | 파일 |
|---|---|---|
| XSkill | 128 프로토타입, lr 2e-4, 배치 128, Sinkhorn 3회, 30프레임/슬라이드 3, 정책 obs 4, pred 16, act 8, 확산 60, lr 1e-4 | `xskill/config/stage1_pretrain_encoder.yaml`, `stage2_skill_transfer.yaml` |
| GR00T-N1.6 | VLM(llm, visual) 동결, 프로젝터·DiT·vlln 학습, lr 1e-4, wd 1e-5, warmup 0.05, 배치 256, 추론 4스텝, 10 에폭 | `gr00t/gr00t/configs/finetune_config.py`, `exp_scripts/gr00t/run_exp_gr00t.sh` |
| π0.5 | lr 3e-5, warmup 1000, 배치 16 | `pi/train_pi_lerobot_jax.py`: 1552–1553, `run_exp_pi.sh` |
| RDT-1B | LoRA rank 576, alpha 1152, 배치 128 | `exp_scripts/rdt/run_exp_rdt.sh`: 48–49 |
| 영상 조건 VA | 작업 임베딩 256차원, 어댑터 한 층, KL 가중 10, 예측 지평 24(ACT), 16(DP) | `act/train_act_imitator.py`: 88, 148 등, `diffusion_policy/train_dp_imitator.py`: 133 |
| 평가 | 쌍당 10회(`--num-episodes 10`), 에피소드 최대 500 스텝, 레벨은 정책에 비공개(과제와 영상만 매핑) | `exp_scripts/act/run_eval_act.sh`: 37, `evaluate_processor.py` |
| 데이터 | 과제 50개 시뮬, L3 없는 실제 과제 6개(H29, H33, H36, H38, H39, H43), 시뮬 대응 없는 실제 과제 3개(H25, H37, H38) | `task_mapping.json`, `task_desc/sim_desc.json` (200 항목) |

## 3. 판정 요약

- **가장 영향이 큰 차이:** D4(규모 효과의 연산 조건), D7·D8·D9(성공·하위 목표 정의), D14(Scr와 P+FT 레시피 비공개).
- D1–D3, D5는 논문 부록 F의 하이퍼파라미터가 공식 런처와 다르다는 점입니다. 기본값이 논문 값이고 런처가 다르다는 것(D1)은 논문이 나온 뒤 런처가 바뀌었거나, 부록이 기본값을 옮겨 적었을 가능성을 모두 열어 둡니다. 코드만으로 어느 쪽이 실제 실험인지는 가를 수 없습니다.
- 프로젝트 페이지 JS 수치가 논문과 정확히 일치하므로(`03` V8), 논문 수치가 실제로 존재하는 파이프라인에서 나왔다는 점은 지지됩니다. 문제는 **그 파이프라인이 저장소 밖**에 있다는 것입니다.

## 4. 구현 가능성 ("구현할 수 있는지")

**판정: 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 벤치마크의 시뮬 평가와 학습은 사실상 불가능합니다. 정적 재계산과 통계 분석은 끝냈고, 작은 구조 재구현(영상 조건 정책 토이)은 조건부로 가능합니다. 실제 로봇 평가와 Imitator Arena는 하드웨어와 사람 평가자가 필요해 불가입니다.**

| 구성 요소 | 필요한 것 | 이 맥에서 | 근거 |
|---|---|---|---|
| ManiSkill3/SAPIEN 시뮬 환경 | Linux/Windows, Python 3.12, Vulkan 렌더링 | **불가에 가까움 (미확인 일부)** | `pyproject.toml`이 `sapien`, `mplib`, `fast_kinematics`를 Linux/Windows 조건으로만 지정. `jax[cuda12]`, `deepspeed`, `bitsandbytes`, `pynvml` 의존. macOS 렌더링은 확인하지 못함 |
| 베이스라인 학습 (15변형 x 3규모) | CUDA GPU 다수(런처가 GPU 0/1/2 병렬), 대용량 캐시 | **불가** | 7B OpenVLA, π0.5(JAX CUDA), GR00T-3B, RDT-1B LoRA는 맥에서 비현실적. ACT/DP 크기도 hidden 1024, 12층이라 큼 |
| 데이터 | IG-10K ~ 880 GB(API 표기), 자산 20 GB | **불가** | 24 GiB 메모리와 디스크 제약. 받지 않음 |
| 체크포인트 평가 | 위 시뮬 + CUDA | **불가** | HF 3.5 GB지만 평가 환경이 필요 |
| 실제 로봇 | 듀얼암 Realman, VR 원격조작, 사람 평가자 | **불가** | 논문 B.1 |
| Imitator Arena 사람 평가 | 평가자 10명 이상 | **불가** | 웹 플랫폼은 저장소에 없음 |
| 정적 재계산, 통계, 셀 데이터 재구성 | 논문 표, 프로젝트 페이지 JS | **가능, 완료** | `analysis/recompute.py` (수 초) |
| 소형 재구현: 레벨별 장면 불일치를 흉내 낸 영상 조건 정책 토이 | torch (이미 설치), MuJoCo 4.0 | **조건부 가능 (미구현)** | 이 논문 수준의 현실적 영상 조건·사람 영상 없이, 목표 지정 모듈이 "장면 불일치에서 의도를 추론"하는지만 보는 축소 실험. 이 맥에서 MuJoCo는 사용 가능 |

**노력 추정 (측정 전):** 정적 분석은 완료. 소형 토이는 1–2주. 시뮬 하나를 Linux GPU 기기에서 재현하려면 설치와 자산 다운로드 약 2–3일 + 변형당 학습(런처 기준 GPU 3개 병렬) 수일, 15변형 전체는 수 주. 실제 로봇 재현은 이 저장소 범위 밖입니다.

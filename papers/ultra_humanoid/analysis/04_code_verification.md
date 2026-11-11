# ULTRA: 논문 ↔ 공식 코드 대조 (정적 읽기)

## 0. 대상과 방법

- **저장소:** github.com/Sirui-Xu/ULTRA (Apache-2.0, InterMimic 파생 MIT). 얕은 클론 `repo/ULTRA/`(gitignore). 커밋 `146ddd5f63987f9b41304b368e40f69990e63ec2` (2026-10-17, "Fix project page link in README").
- **읽은 것:** `README.md`, `docs/install.md`, `docs/deploy.md`, `docs/teacher-model-card.md`, `requirements.txt`, `scripts/train_retarget_smplx.sh`, `train_teacher.sh`, `train_student.sh`, `train_finetune.sh`, `ultra/data/cfg/g1_retarget_smplx.yaml`, `g1_teacher.yaml`, `g1_student_vae.yaml`, `g1_student_finetune.yaml`(일부), `ultra/env/tasks/ultra_g1.py`, `ultra_g1_retarget.py`(보상 함수 1053–1245행), `humanoid_g1.py`(보상 함수 516–745행), `ultra/learning/ultra_agent_distill_vae_rl.py`(손실 657–690행), `ultra/run_teacher_inference.py`, `ultra/sim2sim_teacher.py`(임포트).
- **하지 않은 것:** 코드 실행, `pip install`, 데이터·가중치 다운로드(Google Drive 링크 포함), GPU 사용, IsaacGym 설치. 모든 결과는 읽기에서 나온 것입니다.
- **표기:** 논문은 절·표·부록 번호(IV-A, Table B 등), 코드는 `파일:줄`로 씁니다.

## 1. 저장소에 있는 것과 없는 것

| 구성 | 저장소에 있나 | 비고 |
|---|---|---|
| 리타게팅 학습 코드(UltraG1) | 있음 | `scripts/train_retarget_smplx.sh` → `--task UltraG1` |
| 리타게팅 증강 내보내기 | 있음 | `scripts/export_retarget_smplx.py` (`--xyz`, `--asset-scale`) |
| 교사 학습 코드(UltraG1Retarget) | 있음 | `scripts/train_teacher.sh` |
| 학생 증류 코드(VAE) | 있음 | `scripts/train_student.sh` → `UltraDistillObjV2Point` |
| RL 미세조정 코드 | 있음 | `scripts/train_finetune.sh` → `UltraDistillObjV3RL` |
| **교사 추론 가중치** | 있음 | `ultra/weights/teacher_ultra_inference.pth`, 45.9 MB |
| **학생 가중치** | **없음** | 표 I, III, IV의 학생 결과를 저장소만으로 재현할 수 없음 |
| 리타게팅 가중치 | 없음 | 학습해야 함 |
| 데이터 | 없음 | 두 개의 Google Drive 링크(InterAct OMOMO 준비본, ULTRA 리타게팅 증강본). 받지 않음 |
| 실물 배포 스택 | **없음** | `docs/deploy.md`: ROS 노드, G1 저수준 제어, OptiTrack, 깊이 점군 추출, 키보드 목표 인터페이스는 저장소 밖 |
| 시뮬 대 시뮬(MuJoCo) 검증 | 있음 | `ultra/sim2sim_teacher.py`, `sim2sim_vae.py`, `sim2sim_vae_keyboard.py` |
| 로봇 자산 | 있음 | G1 29-DoF URDF·XML·메시, 물체 4종(+크기 변형) |

## 2. 논문 ↔ 코드 D-items

판정: **일치**, **불일치**(논문과 코드가 다름), **추가**(코드에만 있음), **확인 불가**(읽기로는 근거를 못 찾음).

| ID | 주제 | 논문 | 코드 | 판정 | 근거 |
|---|---|---|---|---|---|
| D1 | 리타게팅 보상 구조 | 곱 형태 단일식 r_track = r_p·r_r·r_obj·r_int·r_ct·r_eng (IV-A) | 두 구현이 있음. 리타게팅 단계(UltraG1)는 보상 = reward_dof·contact_energy·energy·obj_energy·w1 + rb·ro·rig·w2 + rb·ro·rig·rcg·w3. w1, w2, w3는 시간에 대한 sigmoid로 "정지 → 접근 → 추적" 단계를 나눔. 교사 단계(UltraG1Retarget)는 단일 곱에 **1.6 배율**을 곱함 | 불일치(추가 항목 있음) | `humanoid_g1.py:743`, `ultra_g1_retarget.py:1238, 1243`, 주석 "I: standing; II: approaching; III: tracking" |
| D2 | r_p 대상 (발과 손바닥) | "F은 발과 손바닥"(IV-A) | 리타게팅 단계: `local_key_pos = key_pos[:, [2, 5]]`(발만)으로 rp를 계산. 손은 접촉과 r_int에만 씀. 교사 단계는 **39개 전신 링크**의 위치를 씀 | 불일치 | `humanoid_g1.py:631–632`, `ultra_g1_retarget.py:1155–1156` |
| D3 | r_r (링크 방향 대 관절각) | 핵심 링크 방향의 정규화 벡터 일치(IV-A) | 리타게팅 단계: 링크 방향 벡터 `rot_id` 9쌍으로 일치(일치). 교사 단계: **관절각** `dof_pos`를 직접 일치 | 일치(리타게팅), 불일치(교사) | `humanoid_g1.py:634–635`(local_key_rot), `ultra_g1_retarget.py:1159–1161` |
| D4 | r_obj: 물체 위치 추적 | "물체 자세와 속도 추적"(IV-A) | 리타게팅 설정 `op: 0.`이라 **물체 위치 보상이 꺼짐**(`rop = exp(−0·eop) = 1`). 회전과 속도만 남음(or 0.1, opv 0.01). 물체 위치는 종료 조건(물체 점군 차이 > 0.5 m)으로만 제한 | 불일치 | `g1_retarget_smplx.yaml:67`, `ultra_g1_retarget.py:1177, 1195` |
| D5 | r_int 가중치 (kint = 20) | Table B: kint = 20 | 설정 파일 `ig: 20`은 **읽히지 않음**. 리타게팅 단계는 `w_ig = 5`로, 교사 단계는 `w_ig = 1`로 **함수 안에 하드코딩**. `run.py`/`run_distill.py`가 `args.ig`를 넣어도 사용처가 없음 | 불일치(구성 값 무효) | `humanoid_g1.py:616`, `ultra_g1_retarget.py:1140`, `run.py:251` |
| D6 | r_ct 가중치 (kct = 5) | Table B: kct = 5 | 설정의 `cg1`, `cg2 = 5`는 **읽히지 않음**. 리타게팅 단계는 손 접촉에 `w_cg_left = 0.5`, 교사 단계는 전신 접촉에 `w_cg = 1.0`으로 하드코딩 | 불일치(구성 값 무효) | `humanoid_g1.py:683`, `ultra_g1_retarget.py:1208` |
| D7 | r_eng (노력·발 배치 정규화) | "관절 노력과 발 배치 정규화"(IV-A) | 행동 노력 항(eg1)은 교사 설정에서 0.0이고, 리타게팅 함수는 행동 항을 아예 쓰지 않음. 접촉 에너지(eg3 = 1e−10)는 사실상 1. 발 배치 정규화 항은 없음 | 불일치(사실상 없음) | `g1_teacher.yaml:86–88`, `ultra_g1_retarget.py:1231–1235` |
| D8 | 기준 궤적의 발 높이 보정 | 논문에 없음 | 교사 단계는 기준 발 높이에 **발 수평 속도 × 0.1**을 더함(발이 움직이면 기준 높이를 올림) | 추가(미기재) | `ultra_g1_retarget.py:1126` (`vel_to_height_coef = 0.1`) |
| D9 | 교사 보상 가중치 (Table B) | kp = 10, kr = 5, kpv = 0.1, kop = 5, kor = 0.5, kopv = 0.1, kint = 20, kct = 5 | 교사 설정 `p 10, r 5, pv 0.1, op 5, or 0.5, opv 0.1, ig 20, cg 5` (`g1_teacher.yaml:67–78`)는 Table B와 같음. 그러나 `ig`와 `cg`는 D5·D6에서 보듯 **적용되지 않음** | 부분 일치 | D5, D6 |
| D10 | 자세 사전 보상 | 논문에 없음 | 리타게팅 단계의 자세 사전 `exp(−20·mean((q − q_init)²))`는 **정지 단계에서만** 씀(w1) | 추가(미기재) | `humanoid_g1.py:528, 743` |
| D11 | 교사 학습 데이터 | OMOMO 데이터(V-A) | **공개 교사 가중치**는 README와 model card에 따르면 OMOMO 외에 **AMASS와 BONES-SEED를 추가로** 학습. 학습 코드는 OMOMO 설정을 따름 | 불일치(공개 가중치 한정) | `README.md`(Inference), `docs/teacher-model-card.md` |
| D12 | 관측과 행동 차원 | 교사 4052(Table D), 학생 1496(Table H), 행동 29, 잠재 64 | 교사 추론 가중치 설명: 입력 4052, 출력 29 (`teacher-model-card.md`). 배포 설명: 학생 입력 1496, 출력 29, 잠재 64 (`deploy.md`) | 일치 | `docs/teacher-model-card.md`, `docs/deploy.md` |
| D13 | 증류 손실의 prior-only 행동 항 | Table F (V): prior-only 행동 맞춤 손실 포함 | 계수 `prior_action_coeff = 0.0·(…)`으로 **항상 0**. 항은 계산하지만 손실에 기여하지 않음 | 불일치(코드에서 꺼짐) | `ultra_agent_distill_vae_rl.py:663, 683` |
| D14 | KL, 평활, 보조 손실 일정 | 0.001→0.1 (KL), 0.0001→0.001 (평활), λ_E = λ_A = λ_G = 1, 에포크 500–3500 (Table F) | KL: `0.001 + 0.099·(1 − cos)/2`, 평활: `0.0001 + 0.0009·…`, 에포크 500–3500. 같음 | 일치 | `ultra_agent_distill_vae_rl.py:657–667` |
| D15 | RL 미세조정 손실 | PPO는 배포용 학생(사전 분포만)에 적용, 크기는 "작은 상수", 액터는 "100 에포크 동안 점진 활성"(App. C) | RL 항에 **0.2** 배율. 액터 계수는 시작 후 100–200 에포크에 걸쳐 0에서 1로 올림. 학생 사전 분포만 쓰는지는 `UltraDistillObjV3RL`에서 확인 필요 | 일치(수치 추가) | `ultra_agent_distill_vae_rl.py:674, 688` |
| D16 | 점군 잡음 | Table K: 가우시안 0.02 m, 점 누락 0.15, 이상점 0.05, 가림 0.1 | 설정: `pointNoiseStd: 0.02`, `pointDropoutProb: 0.15`, `pointOutlierProb: 0.05`, `pointOcclusionProb: 0.1` | 일치 | `g1_student_finetune.yaml:103–114` |
| D17 | 목표 성공 보너스 | Table G: 성공 보너스 | `goal_success_bonus: 10.0` | 추가(값 공개) | `g1_student_finetune.yaml:72` |
| D18 | 실물 배포 | 실물 G1 실험(V-E) | 배포 스택이 저장소에 없음. 대신 `export_jit.py`(TorchScript, 원시 1496차원 관측 입력, 행동 ×3.0 PD 목표)와 MuJoCo 키보드 데모만 있음 | 불일치(재현 불가) | `docs/deploy.md`, `ultra/export_jit.py` |
| D19 | 리타게팅 제어기 | "이상적인 저수준 제어, 제어 주파수 = 시뮬 주파수"(IV-A) | 설정은 `controlFrequencyInv: 17`(교사와 같음)이고 README는 "position target PD control"이라고 씀. 이상적 제어 여부는 확인 못함 | 불일치(부분) | `g1_retarget_smplx.yaml:16`, `README.md`(stage 1) |
| D20 | 리타게팅 학습 길이 | 논문에 없음 | "최대 50,000 에포크", 4096 환경 | 추가 | `README.md`(stage 1) |
| D21 | 물체 종류 | largebox, plasticbox, smallbox, suitcase (V-A) | 같은 4종 + 크기 변형 `080_080_080`, `100_100_100` | 일치 | `ultra/data/assets/objects/` |
| D22 | 증강 (축별 스케일, 물체 크기) | 재학습 없이 한 정책으로 적용 (IV-A) | `export_retarget_smplx.py --xyz X Y Z`(반복 지정)와 `--asset-scale`로 같은 정책을 실행 | 일치 | `README.md`(stage 2) |
| D23 | 종료 조건(접촉 불일치 20프레임) | "접촉 불일치 20프레임" (IV-A, [3]) | 종료 로직은 확인했으나 프레임 수는 줄 단위로 찾지 못함 | 확인 불가 | — |
| D24 | 깊이 점군 추출 파이프라인 | 역투영, 전방 ROI, 바닥 제거, 최대 군집 (V-E) | 학습 쪽은 점군 잡음만 있고 실물 추출 코드는 없음 | 불일치(배포 스택에 있음) | `docs/deploy.md` |

**핵심 결론.** 보상의 **구조**와 **일정**은 대부분 맞지만, **가중치 몇 개가 실제로 쓰이지 않고** **몇몇 항이 논문에 없고** **일부 논문 항이 0이거나 꺼져 있습니다.** 특히 D4(물체 위치 보상 0)와 D5·D6(표 B 가중치가 무효)은 논문의 "물체와 접촉 보존" 주장과 직접 관련됩니다. 저자가 실제로 쓴 설정이 논문의 표와 다르다는 뜻이므로, 재현할 때는 **코드를 기준으로** 해야 합니다.

## 3. 세부 확인

- **증강.** 논문은 증강을 "재학습 없이 한 정책으로"라고 씁니다(IV-A). 코드에서 `--xyz`와 `--asset-scale`로 같은 체크포인트를 재실행합니다(D22). 일치합니다.
- **교사 추론.** `run_teacher_inference.py`는 `isaacgym`을 임포트합니다. 반면 `sim2sim_teacher.py`는 `mujoco`, `rl_games`, `trimesh`를 씁니다. **MuJoCo 기반 교사 추론은 IsaacGym 없이 가능한 구조입니다**(확인은 실행하지 않음).
- **관측 빌더.** `ultra/utils/obs_vae.py`(`MujocoObs`)가 IsaacGym 밖의 학생 관측을 만듭니다. 학생 관측을 재구성하는 참조 구현으로 쓸 수 있습니다.
- **정규화.** 배포 문서는 학습 설정에서 `normalize_input: False`라고 씁니다. TorchScript 내보내기는 항등 정규화를 씁니다. 실행 중 정규화 통계는 없습니다.

## 4. 구현 가능성 판정

### 4.1 공식 재현 (학습 파이프라인 전체)

**판정: 이 맥에서 불가능합니다.**

- 학습은 IsaacGym Preview 4가 필요합니다(`docs/install.md`). IsaacGym은 NVIDIA GPU와 Linux를 전제로 합니다. 이 맥(Apple Silicon, CUDA 없음)에서는 실행할 수 없습니다.
- Python 3.8, CUDA 11.x, torch cu118을 쓰는 설치 절차입니다. `numpy<1.24`가 필요합니다.
- 단계별 필요 자원: 리타게팅 학습(최대 50,000 에포크, 4096 환경), 교사 학습, 증강 내보내기, 학생 증류(5500 에포크), RL 미세조정. 저장소에 소요 시간 기록은 없습니다. **단일 RTX급 GPU에서 수일에서 수주로 추정합니다**(근거 없는 추정).
- 데이터: OMOMO 준비본과 ULTRA 증강본을 Google Drive에서 받아야 합니다. OMOMO와 InterMimic 데이터 조건을 확인해야 합니다. AMASS와 BONES-SEED 조건은 공개 교사 가중치에만 해당합니다.

### 4.2 부분 재현 (이 맥에서 가능성 있음, 미검증)

- **교사 가중치의 MuJoCo 추론(sim2sim_teacher.py).** 구조상 IsaacGym이 필요 없어 보입니다. 이 맥에서 CPU로 한 모션을 돌려 보는 것은 **조건부로 가능할 수 있습니다.** 필요한 것: torch, rl_games==1.1.4, mujoco, trimesh 설치(사용자 승인 후), 모션 `.pt` 파일 1개(Google Drive), 교사 가중치(저장소에 있음). 실행 시간과 결과는 확인하지 않았습니다.
- 이것으로 얻는 것은 **교사의 추적 성능 일부**뿐입니다. 학생, 목표 추종, RL 미세조정, 실물은 검증하지 못합니다.
- 이 실험은 AMASS 또는 BONES-SEED로 학습된 가중치라는 점에서 논문의 교사와 다릅니다. 결과를 논문 Table I과 직접 비교하면 안 됩니다.

### 4.3 실물 재현

**판정: 불가능합니다.**

- 배포 스택(ROS 노드, G1 저수준, OptiTrack, 깊이 점군)이 저장소에 없습니다(D18).
- Unitree G1 실물, OptiTrack 시스템, 깊이 카메라, 가정용 상자 물체가 필요합니다. 현재 사용할 수 있는 환경에는 없습니다.
- 가장 가까운 대체: 시뮬 대 시뮬(MuJoCo)에서 학생 관측을 만들고 TorchScript로 내보내는 경로(`export_jit.py`)는 있습니다. 학생 가중치가 없으므로 이것도 학습이 먼저 필요합니다.

### 4.4 요약 표

| 수준 | 이 맥 (CPU, MPS, CUDA 없음) | 필요한 것 | 판정 |
|---|---|---|---|
| 리타게팅·교사·학생 학습 | 불가 | NVIDIA GPU, Linux, IsaacGym Preview 4, 데이터 | **불가** |
| 교사 MuJoCo 추론 (한 모션) | 조건부 | torch, rl_games, mujoco, trimesh, 모션 파일 1개 | 조건부(미검증) |
| 학생 추론과 TorchScript 내보내기 | 학생 가중치 없음 | 학생 학습 필요 | 불가(학습 선행) |
| 목표 추종 (sim2sim) | 학생 가중치 없음 | 학생 학습 필요 | 불가(학습 선행) |
| 실물 G1 | 불가 | G1, OptiTrack, 깊이 카메라, 배포 스택 직접 작성 | **불가** |

**총평.** 공식 코드는 **재현 경로가 열려 있지만**(Apache-2.0, 학습 스크립트, 교사 가중치) 학생 가중치가 빠져 있어 **표 I, III, IV의 핵심 결과를 저장소만으로 재현할 수 없습니다.** 이 맥에서는 교사 추론 일부만 조건부로 가능합니다.

## 5. 권리와 라이선스

- **논문.** arXiv 비독점 배포(nonexclusive-distrib 1.0)입니다. CC BY가 아닙니다. 원문 PDF에는 저작권 표시가 없고 arXiv 스탬프만 있습니다(1쪽, 확인함). 따라서 **논문 그림은 개인용 리포트와 PPT에만 씁니다.** 게시본에는 우리가 그린 그래프만 넣습니다.
- **코드.** 저장소 최상위 LICENSE는 Apache-2.0입니다. `LICENSE-InterMimic`은 InterMimic 파생 MIT이고, 일부 IsaacGym 예제 파일은 NVIDIA 헤더를 유지합니다.
- **교사 가중치.** 상위 Apache-2.0이 데이터 조건을 대체하지 않습니다. AMASS는 비상업 연구·교육으로 제한되고, BONES-SEED는 모델 배포 시 조건과 출처 표기가 필요합니다. 가중치는 비상업 연구·교육용으로 제공된다고 명시되어 있습니다(`docs/teacher-model-card.md`).
- **데이터.** OMOMO와 InterAct 준비본은 원 데이터 조건을 따릅니다. 이 분석에서 내려받지 않았습니다.
- **로봇 자산.** Unitree에서 왔습니다. 재배포 조건은 확인하지 않았습니다.

## 6. 열린 질문 (저자나 저장소에 물을 것)

1. 학생 가중치를 공개할 계획이 있는지.
2. 논문 Table I과 III의 학생 결과를 어떤 설정(D5, D6의 하드코딩 값, D13의 0 계수)으로 얻었는지. 공개 코드가 그 설정과 같은지.
3. 공개 교사 가중치가 AMASS와 BONES-SEED로 추가 학습된 뒤 논문 Table I의 교사 성능이 달라지는지.
4. 실물 배포 스택을 공개할 계획이 있는지.

# 04. 공식 코드 대조 (정적 읽기)

공식 저장소 `github.com/univtac/UniVTAC`의 메인 브랜치 커밋 `0dafa10`(2026-09-07)을 `repo/`에 얕게 복제해 **정적으로만** 읽었습니다. 설치, import, 실행, 데이터 다운로드는 하지 않았습니다. 복제된 내용은 신뢰할 수 없는 데이터로 취급했습니다. 대조 항목은 논문 본문과 부록의 주장에 맞춰 D1–D23으로 번호를 매겼습니다. "일치"는 코드가 논문 설명과 맞는다는 뜻이고, "차이"는 다르다는 뜻입니다. "확인 불가"는 저장소만으로 판단할 수 없다는 뜻입니다.

## 1. 확인 범위

| 영역 | 파일 | 읽은 내용 |
|---|---|---|
| 인코더 | `encoder/network.py`, `encoder/train.py`, `encoder/dataloader.py` | 백본, 디코더, 손실 가중치, 학습 루프, 데이터 전처리 |
| 촉각 제어와 성공 기준 | `envs/_base_task.py`, `envs/collect.py`, `envs/sensors/tactile.py`, `envs/*.py`(과제 8개) | 적응 그리퍼 제어, 데이터 수집 분기, 마커 수, `check_success` |
| 정책 설정 | `policy/ACT/*.yml`, `policy/Ablation/*.yml`, `policy/ViTAL/*`, `policy/SIM_TASK_CONFIGS.json` | 학습 하이퍼파라미터, 절제 설정, 카메라 구성 |
| 평가 | `scripts/eval_policy.py`, `parallel_eval.sh` | 시드 루프, 예외 처리, 전문가 검사 |
| 문서 | `README.md`, `docs/*.md`, `LICENSE` | 설치, 데이터 구조, 뉴스(CoRL 채택), 라이선스 |

읽지 않은 것: `policy/smolvla*`(lerobot 사본), `third_party/TacEx` 내부, `assets/`(USD 자산), `policy/UniT`의 세부 구현, 노트북 파일.

## 2. 대조 표

| 번호 | 논문 주장 (위치) | 코드 (파일) | 판정 |
|---|---|---|---|
| D1 | 그리퍼 속도 식 (1): 무접촉 `v_fast`, 접촉 시 `min(|d_min − δ_th|, v_slow)` (III-A-2) | `envs/_base_task.py`의 `adaptive_set_gripper`: 무접촉 0.0005, 접촉 시 0.00005를 상한으로 한 **위치 증분 루프**(최대 1000회). 목표 깊이는 mm를 m로 바꾼 `/1000`. 임계에 도달하면 `break` | **구조 일치, 세부 차이.** 속도 법칙이 아니라 위치 증분 루프이고, 상수 이름(v_fast, v_slow)은 없음 |
| D2 | Grasp와 Probe는 촉각 반응 제어를 쓴다. 데이터 수집은 세 센서 모두 접촉 폐루프 (III-A-2, III-A-4) | `envs/collect.py`: **GelSight Mini만** `use_adaptive_grasp = True`(촉각 폐루프). GF225와 XenseWS 분기는 고정 그리퍼 위치 `uniform(0.0118, 0.013)`로 **개방 루프** | **차이.** 논문의 "센서 무관 폐루프" 설명과 다름 |
| D3 | 접촉 깊이 임계 δ_th를 바꿔 "가벼운 접촉부터 깊은 압입까지" 데이터 생성 (III-A-4) | `envs/collect.py` L72: `uniform(27.7, 28.1)` mm, 폭 0.4 mm. 과제별 임계(`lift_bottle.py` 27.8 mm 등)와 같은 스케일 | **차이 (범위).** 수집 범위가 논문 설명보다 좁음. 단위와 기준 깊이가 맞다는 전제에서 판단 |
| D4 | 조작 API: Grasp, Move, Place, Probe, Rotate (III-A-2) | `envs/utils/atom.py`: `grasp_actor`, `place_actor`, `move_by_displacement`, `move_to_pose`. "probe"라는 이름은 없음. 회전은 `collect.py`의 `move_by_displacement` 반복 | **확인 불가.** Probe 대응 코드를 찾지 못함 |
| D5 | 인코더 사전학습: 학습률, 에폭, 배치, 분할 등은 **본문에 없음**. 손실 가중치만 있음 (III-B, 식 5) | `encoder/train.py`: Adam, 학습률 1e-3, **5 에폭**, 배치 64, 검증 20%(시드 42). 백본 `resnet18(num_classes=512)`로 **무작위 초기화**(사전학습 가중치 불러오지 않음). 샘플은 `data_length`(기본 1000)개를 무작위로 뽑음 | **일치 안 함(누락).** 에폭이 5로 고정이라 학습 스텝 수는 샘플 수에 거의 비례. 1K 샘플은 약 65 스텝, 200K는 약 1.25만 스텝. 표 III의 데이터 규모 효과와 학습량 효과가 섞임 (03 V9) |
| D6 | 205,826 샘플, 형상당 약 14,000 프레임 (III-A-4) | `encoder/dataloader.py`: 프레임 하나에서 **왼쪽과 오른쪽 센서 샘플 두 개**를 만듦. "샘플"의 단위는 프레임이 아니라 센서 이미지 | **확인 불가.** 14 × 14,000 = 196,000 프레임이면 샘플 수는 약 392,000. 205,826은 프레임 약 10만 개로도 읽힘. 데이터 메타데이터를 받아야 확정 가능 |
| D7 | 손실 λ_s = 1.0, λ_c = 0.5, λ_p = 0.5 (식 5) | `encoder/train.py` 전체 설정: `marked_rgb`와 `rgb` 1.0, `depth`와 `marker` 0.5, `pose` 0.5 | **일치.** λ_c를 깊이와 마커에 각각 0.5로 적용 (식 3의 합 구조와 같음) |
| D8 | 과제마다 100회 테스트 롤아웃 (IV-A) | `scripts/eval_policy.py`: 시드 루프. 예외가 난 롤아웃은 **세지 않고 다시 뽑음**(`test_num -= 1`). 전문가 검사(`--expert_check`)는 **기본이 꺼짐**이고, 켜면 전문가가 성공하지 못하는 시드를 제외. `parallel_eval.sh`의 기본 총 수는 100 | **차이 (보고 누락).** 기본 설정에서도 예외 롤아웃은 분모에서 빠짐. 실패로 세지 않음. 논문은 이를 밝히지 않음. 전문가 검사를 켠 설정을 썼는지는 논문에 없음 |
| D9 | 무효 시험: 최대 관통 깊이 초과 또는 큰 미끄럼이면 무효 (III-C) | `envs/`, `scripts/`, `policy/`에서 관통(penetration)이나 미끄럼(slip) 검사를 찾지 못함. 과제 성공은 물체 포즈 술어만 씀 (예: Insert Hole은 x, y < 1 cm, z < −4 cm, 기울기 cos > 0.99, 손 안 편차 < 4 cm) | **차이 (공개 코드 범위).** 논문이 말한 무효 규칙은 이 공개 저장소에 없음. 다른 곳에 있을 가능성은 배제하지 못함 |
| D10 | 마커 N개 (III-B, 명시 없음) | 시뮬 설정: GelSight Mini 64, GF225 81, XenseWS 220 (`envs/sensors/tactile.py`). 인코더는 **앞 63개만** 사용 (`dataloader.py`의 `[0, :63]`, `MarkerDecoder(marker_nums=63)`) | **차이 (설명 없음).** 64 → 63 절단 이유가 코드에 없음. 논문의 N 값도 없음 |
| D11 | 깊이 맵 D와 마커 좌표 M의 정규화 (III-B) | 깊이: `(d − 24) / (34 − 24)` 고정 범위. 마커: 320 × 240으로 나눔. 이미지: 320 × 240으로 리사이즈 | **확인 불가 (세부 누락).** 논문에 없는 고정 상수. 재현에 필요 |
| D12 | 경로 절제 (표 II): Contact만, Shape만, Contact+Shape, Full | 설정 파일: `contact_pathway.yml`은 깊이 1.0, 마커 1.0 (Full은 0.5, 0.5). `contact_shape.yml`은 깊이 0.5, 마커 0.5, 형상 1.0, pose 0 (= Full에서 pose만 뺀 것). `from_scrach.yml`은 `tactile_ckpt: null` | **부분 일치.** Contact 설정은 λ를 바꿔서, 경로 제거만 보는 절제가 아님. Scratch는 일치 |
| D13 | 데이터 규모 0, 1K, 5K, 10K, 50K, 100K, **200K** (III-C, 표 III) | `policy/Ablation/data_*.yml`: 1000, 5000, 10000, 50000, 100000 (+ `_all`). **200K 설정 없음** | **차이 (재현 경로 누락).** 200K 설정은 공개 저장소에 없음. `_all`은 손목 카메라를 추가함 |
| D14 | VITaL 기준선을 같은 조건에서 학습 (IV-A) | `policy/ViTAL/`은 원 VITaL(케이블 꽂기, arXiv:2403.11898) 저장소 기반. 행동 청크 **20** (ACT는 50), dropout 0.025 (ACT 0.1), z 차원 32, 배치 32 또는 64, **4000 에폭** (ACT는 4000 스텝). CLIP 사전학습 경로는 `lift_bottle_v3` | **차이 (공정성 확인 불가).** 논문은 "모든 정책을 같은 50 궤적으로 학습"이라고만 씀. 기준선 하이퍼파라미터가 다름 |
| D15 | 부록 B: 학습률 1e-5, 가중치 감쇠 1e-4, 배치 64, 4000 스텝, 인코더 4층, 디코더 7층, 청크 50, 시간 앙상블 | `policy/ACT/train_config.yml`: lr 1e-5, weight_decay 1e-4, batch 64, num_steps 4000, enc 4, dec 7, chunk 50, temporal_agg true, kl 10 | **일치.** |
| D16 | Table I의 "ACT + UniVTAC" 설정 (부록 B: 3인칭 카메라 하나, Lift Bottle과 Insert Tube는 다중 시점) | 설정이 세 개 있음. `train_config.yml`(카메라 `cam_high` + 촉각), `train_config_tactile_full.yml`(**카메라 없음**, 비전 학습률 0), `data_*_all.yml`(`cam_high` + `cam_wrist`). Table I이 어느 설정인지 표시 없음. `SIM_TASK_CONFIGS.json`의 Lift Bottle은 머리·손목·촉각 두 개 | **확인 불가.** 어느 설정이 Table I에 쓰였는지 판단 못 함. "tactile_full"을 쓰면 카메라 없이 촉각만 쓴 결과이므로 본문 설명과 어긋남 |
| D17 | 시뮬 벤치마크는 세 센서를 지원 (I, III-A-1) | README TODO: "데이터 수집과 평가는 지금 GelSight Mini만 지원". `docs`와 `task_config/demo.yml` 기본값도 gsmini. 논문은 벤치마크에 쓴 센서를 밝히지 않음 | **차이 (범위).** 공개 코드의 평가 범위는 GelSight Mini |
| D18 | 라이선스 (해당 없음, 저장소 메타) | `LICENSE` 파일 = Apache License 2.0 (GitHub API도 Apache-2.0). README 맨 아래는 "MIT license"라고 적음 | **불일치 (저장소 내부).** 법적 판단은 하지 않음. 재배포 전에 확인 필요 |
| D19 | 실로봇 실험 (IV-D, 부록 C): Tianji Marvin, Quest 원격조작, 과제당 시연 150개, 20회 평가 | 실로봇 배치 코드, 실로봇 데이터 설정, 시연 수집 스크립트를 찾지 못함 | **확인 불가.** 실로봇 결과는 저장소로 검증할 수 없음 |
| D20 | 시뮬 체크포인트 (논문 본문에 없음) | README: 공개 체크포인트는 **Isaac Sim 4.5** 데이터로 학습했고 5.1에는 쓰지 말라고 함. 사전학습 인코더는 `checkpoints/`(HF)에 있음. 설정 파일이 가리키는 인코더 경로는 `encoder/checkpoints/resnet18/20251128-125750` 등 | **확인 불가.** 가중치는 받지 않음. Table I 결과와 공개 체크포인트의 대응은 확인 못 함 |
| D21 | Lift Bottle 성공: 최종 밑면을 벽에서 5 cm 이내 (부록 A) | `envs/lift_bottle.py`: `rel_pose[0] > −0.02`(물체를 목표 기준 x로 2 cm 여유), y 범위 ±0.1, 기울기 cos > 0.99 | **확인 필요.** 좌표 기준이 다르면 직접 비교가 안 됨. 5 cm와 2 cm의 관계가 불명확 |
| D22 | Lift Can 성공: 들어 올려 미끄럼 없이 (III-C, 부록 A) | `envs/lift_can.py`: `can_pose[2] < 0.01` 과 축 정렬(cos > 0.99). 위치 z가 1 cm 미만이면 성공이라 "들어 올림"과 맞지 않아 보임 | **확인 필요.** 기준점이 다를 가능성. 논문 설명과 코드 판정이 같은지 확인 안 됨 |
| D23 | 과제 설명과 물체 (부록 A, Table V) | 과제 8개 모두 `envs/`에 있고, 지시문은 `instructions/*.json` | **일치 (정의 수준).** 과제 이름과 개수가 맞음 |

## 3. 구현 가능성 (이 맥)

이 맥: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS 사용 가능, CUDA 없음.

| 구성 | 맥에서 가능? | 이유 |
|---|---|---|
| Isaac Sim 4.5 또는 5.1 시뮬레이션 실행 | **불가** | 저장소가 NVIDIA RTX GPU와 Isaac Sim, Isaac Lab, cuRobo를 요구함. 5.1 README는 RTX 40·50 시리즈를 말함 |
| UniVTAC 인코더 사전학습 (ResNet-18, 320×240, 5 에폭) | **조건부** | 코드는 `device = cuda`로 고정됨 (`encoder/train.py`). MPS로 바꾸면 가능할 것으로 보이지만 검증 안 됨. 데이터는 HF의 접촉 사전학습 HDF5(638 에피소드)가 필요하고 크기는 확인 안 함 |
| ACT 정책 학습 (시뮬 벤치마크 시연 50–100개) | **조건부 (느림)** | 4000 스텝, 배치 64, 청크 50. CPU로는 실용적이지 않음. MPS는 추정으로 며칠. 시연 데이터 받기가 먼저 |
| 표 재계산과 통계 | **가능 (완료)** | `analysis/recompute.py`, 수 초 |
| 2-링크 토이 등 CPU 검사 | **가능** | 새 설계가 필요함 |
| 실로봇 재현 | **불가** | 로봇 하드웨어, 촉각 센서 두 개, 원격조작 장비 |
| 데이터 규모 곡선 (200K 포함) | **불가** | GPU 필요. 200K 설정도 없음 (D13) |

**노력 추정 (측정 전, 추정):**
- 인코더 사전학습을 맥(MPS)에서 재현: 데이터 받기와 코드 정리 포함 3–5일.
- ACT 벤치마크 재현 한 과제: 외부 GPU 기준 데이터 받기 포함 하루 이상, 여덟 과제와 세 시드면 1–2주.
- 이 맥에서의 완전 재현: 불가.

## 4. 결론

- 공식 코드는 논문의 학습 설정(부록 B, 손실 λ)과 큰 틀에서 맞습니다.
- 그러나 **논문의 핵심 주장을 뒷받침하는 세부**가 저장소에 없거나 다릅니다: 무효 규칙(D9), 센서 범위(D2, D17), 사전학습 설정(D5), 200K 설정(D13), 기준선 공정성(D14), 평가 분모(D8).
- 이 중 D5(에폭 5 고정과 학습 스텝 혼동)와 D8(예외 롤아웃 제외)은 표 III와 표 I의 해석에 직접 영향을 줍니다.
- 저장소는 연구 재현의 **출발점**으로 쓸 수 있습니다. 결과를 그대로 재현하는 근거로는 부족합니다.

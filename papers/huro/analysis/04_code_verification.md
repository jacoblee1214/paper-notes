# 04. 공식 코드 대조와 구현 가능성

대상: `github.com/3587jjh/HuRo` 커밋 **`082fb40`** (2026-11-10, "Enable the Hugging Face badge"), 약 13.3k줄(파이프라인 `pipeline/`, `common/`, `configs/`, `setup/`, `examples/`, 서브모듈 패치). 얕은 복제를 **정적으로만** 읽었습니다. 코드를 실행, import, 설치하지 않았습니다. 경로는 저장소 루트 기준이고 줄 번호는 이 커밋입니다. 분석일 2026-11-23.

## 0. 요약

- **공개된 것은 "로봇화 파이프라인"(원시 영상 → LeRobot v2.0 데이터셋) 10단계뿐입니다.** VLA 학습·평가 코드, 체크포인트, 하류 시연 데이터, 실로봇 설정은 없습니다 (D1).
- **논문 데이터셋의 55.5%(EgoDex)는 공개 코드로 만들어지지 않았고 공개되지도 않았습니다.** 저자의 열린 PR이 직접 밝힙니다 (D2). 데이터 일부(4개 출처, 44.5%)는 Hugging Face에 접근 제한으로 올라 있지만 README는 아직 "Coming soon"입니다 (D3).
- **파이프라인 핵심(IK 목적항, 정렬 변환, 행동 정의 `a_t = s_{t+1}`, 렌더링 방식)은 논문 서술과 맞습니다.** 맞는 항목은 §2에 모았습니다.
- **논문에 없는 설정과 처리가 코드에 많습니다**: 청크 창 길이 210프레임과 50% 겹침, 키포인트 평활화와 이상치 가중치, 목표 가중치 전부, 허리·목 자유도 최적화, 다중 인물 필터, 렌더 실패 구간 제거 등 (D4–D13).
- **구현 가능성:** 이 맥에서는 파이프라인도 학습도 재현할 수 없습니다 (§4).

## 1. 논문 ↔ 코드 차이 (D1–D15)

판정 표기: **차이** (논문과 다르거나 논문에 없는데 결과에 영향), **미기재** (논문에 없는 설정), **범위** (공개 범위 문제), **일치**는 §2.

| # | 논문 | 코드 (파일:줄) | 판정과 영향 |
|---|---|---|---|
| D1 | §3.2, §4: GR00T-N1.6-3B 기반 VLA, 80k 스텝 × 2048 사전학습, 30k 스텝 미세조정, 실로봇 평가 4과제 | `README.md`: "This repository releases HuRo's robotization pipeline, from raw video to a LeRobot V2.0 dataset". `pipeline/`, `common/`, `configs/`, `setup/`, `examples/`에 학습·평가·정책 코드 없음. 체크포인트, 미세조정 시연(43/40/16/20개), 평가 로그 없음 | **범위.** 논문의 모든 결과 표(2, 3, 4, 7)는 공개물로 재현할 수 없습니다 |
| D2 | §3.1.4, 부록 D.1: EgoDex는 55.5%(78.9M 프레임)이고 "released camera-geometry and hand-pose annotations"를 쓴다. 표 10의 카메라·손 진단은 EgoDex 재실행 | `README.md`: "The released code does not read annotations that a source dataset provides ... write what stage N-1 would write ... with the `.done` markers". 열린 PR #9(저자 `joocjun`)의 설명: EgoDex는 "none of it is published", 라이선스 CC BY-NC-ND 4.0, **EgoDex 부분은 이 저장소의 파이프라인으로 만들어지지 않았고** 카메라·손 주석은 EgoDex의 ARKit 정답을 썼으며 팔 분할·인페인팅은 "earlier code that is not part of the release". 이 저장소로 EgoDex를 돌리면 "comparable dataset, not the paper's" | **범위, 차이.** 규모 결과(표 2)의 데이터 55.5%는 공개 코드로 재생성되지 않습니다. 논문은 이를 본문(§3.1.4, D.1)에서 "available annotations를 사용"이라고만 적고 시각 변환 코드가 다르다는 점은 적지 않았습니다 |
| D3 | v1 초록: "Code and data are released on our website" (v5에서 삭제). 프로젝트 페이지 Datasets 링크 | `README.md` "HuRo Dataset: **Coming soon.**". 배지는 Hugging Face 컬렉션 `Joocjun/huro`로 연결: 4개 데이터셋(`HuRo-EPIC-KITCHENS`, `-Egocentric10K`, `-Ego4D`, `-EgoVerse`), 모두 gated, 최종 수정 2026-11-09. EgoDex 없음. 이슈 #1(데이터 공개 요청, 열림). 프레임 기준 **EgoDex 제외 시 44.5%** | **범위.** README가 최신 상태를 반영하지 않습니다. 내려받아 확인하지 않아 에피소드 수, 행 수, 형식은 검증하지 못했습니다 |
| D4 | §3.1.1: "bounded-length chunks", 최대 길이와 겹침은 미기재. §1: 약 630K 에피소드, 142.2M 프레임 | `pipeline/stage6_annot_narr.py:259-260`: `--uniform_max_seq_len 210`(7.0 s), `--uniform_overlap_ratio 0.50`. 줄 513–530: 연속 조작 구간이 210프레임보다 길면 stride 105(`int(210 × 0.5)`)의 **겹치는 창**을 만들고 마지막 창은 구간 끝에 맞춤. 창마다 별도 세그먼트로 저장(줄 664, `rows_to_parquet`). `--min_seq_len 50`(줄 253). stage 7 `--max_seq_len 200`(`stage7_annot_inpaint.py:304`) | **미기재, 불일치 가능.** (a) 에피소드가 50% 겹치면 142.2M "processed frames"에 중복 프레임이 포함됩니다. 논문은 중복 여부를 말하지 않습니다. (b) 142.2M ÷ 630K = 225.7프레임은 창 최대 210보다 큽니다(`03` V10). 코드 기본값으로 이 통계가 나올 수 없습니다. EgoDex 경로가 다르기 때문일 수 있습니다 |
| D5 | §3.1.2, G.2: IK 목적항 나열(손끝 위치, 국소 키포인트 방향, 자기 시점 카메라 일치, 관절 한계, 휴식 자세, 밀집 단계의 시간 평활) | `pipeline/stage8_robot_retarget.py:61-70`: 키포인트를 **가우시안 평활**(`--smooth_sigma 1.0`). 줄 72-105: 손목 속도·가속도 기반 **이상치 프레임 가중치 하향**(4 m/s, 지수 4승, 최소 0.01). 줄 325-336: 손목 중심 기반 해석적 정렬 초기값(작업 공간 앵커, `compute_workspace_anchors`, 줄 282)으로 첫 단계를 초기화. `pipeline/retargeting/retargeter.py:25`: 희소 단계는 **32개 대표 프레임**. 줄 42-53: 가중치(local 20, global 50, 평활 40, 손 평활 ×0.25, 자기 시점 회전 3·위치 10, 휴식 0.2/허리·목 1.2/손 ×0.5, 관절 한계 100). 줄 289, 458: 최대 1000회 반복, 패딩 길이 224/1024/3072 | **미기재.** 목적항 구성은 논문과 같지만 전처리(평활, 이상치 하향, 초기값)와 가중치가 논문에 없습니다. 재현에는 이 값이 필요합니다. 손 자세 오차(20.4 mm 중앙값, 표 10)보다 IK 잔차(21.2 mm)가 비슷한 크기라 가중치 선택이 결과의 상당 부분을 좌우할 수 있습니다 |
| D6 | A.1: "The actuated neck and waist are kept fixed during demonstration collection and evaluation" (실로봇 하류 평가 설명). §3.1.4: ALLEX는 목 2, 허리 2 자유도 | `configs/allex.yaml:50-57,64-65`: `joints`에 `waist`, `neck` 그룹이 **최적화 대상**으로 들어가고, `stiff_groups`로 휴식 자세 가중치만 1.2(기본 0.2)로 높임. `retargeter.py:504`의 `joint_mask`가 설정된 그룹의 관절을 모두 1(최적화)로 둠. `stage10_lerobot_convert.py:56-59,230-242`: 최종 48차원 상태·행동에 목(44:46), 허리(46:48)가 포함 | **미기재.** 데이터셋에는 목과 허리의 움직임이 들어 있는데 논문 본문은 사전학습 행동에 몇 차원이 쓰이는지 적지 않았습니다(§3.2는 손목 + 손 관절만 서술). 학습 쪽 코드가 없어 확인 불가 |
| D7 | F.3: 62.5%의 궤적에서 관절 한계 위반 없음. "Exact projection onto the URDF position limits brings all audited trajectories within the limits" | `retargeter.py:251-258, 402-409`: 관절 한계는 **연성 벌점**(가중치 100)만. 저장 전 투영·클램프 없음(`stage8`, `stage10`에 한계 클램프 코드 없음). 자기 충돌 항 없음(논문과 일치) | **일치하나 주의.** 논문은 "투영 전"을 감사했다고 적었고 코드가 실제로 투영하지 않으므로 공개 테이블의 37.5% 궤적이 한계를 넘습니다. 사용자는 직접 투영해야 합니다 |
| D8 | F.3, Fig. 10: p99 컷 | `stage10_lerobot_convert.py:129-142`: `_DIAG_FILTER`에서 **카메라 위치·회전 오차만 p99로 제거**, 손끝·손바닥·국소 방향·가속도 오차는 100(제거 없음). 최소 100개 에피소드가 있을 때만 적용 | **일치(부분).** 그림 10 범례의 p99와 일치하나 본문이 필터를 설명하지 않습니다. IK 잔차가 큰 에피소드도 걸러지지 않습니다 |
| D9 | §3.1.1: VLM이 "caption-frame consistency"와 손-물체 조작 없는 청크를 거름 | `stage6_annot_narr.py:199-268`: 다중 인물 필터(8프레임마다 사람 검출, 한 프레임에 3명 이상이 2회 이상이면 **구간 전체 제거**). `stage10_lerobot_convert.py:424-450`: 한 프레임이라도 로봇 렌더에 실패한 구간은 **통째로 제외**. 언어가 null인 구간 제외(줄 420-421) | **미기재.** 표 9의 "post-selection retention 97–98%"가 이 필터들을 모두 포함하는지 알 수 없습니다 |
| D10 | §3.1.1, G.1: Qwen3.5로 지시문 생성 + 검증 단계 | `pipeline/captioning/caption.py:31`: `MODEL_ID = "Qwen/Qwen3.5-9B"`. 줄 833-836: 샘플 온도 0.6, top_p 0.95. `stage6_annot_narr.py:256-258`: 프레임 stride 3(10 fps), 샘플 2 × 배치 2 = 4개. 지시문은 4개 샘플의 병합(`merge_narration`). 논문 참고문헌 [33]은 "Qwen3.5-omni technical report" | **미기재, 인용 불일치 가능.** 코드가 쓰는 9B 모델과 인용된 "omni" 보고서가 같은 계열인지 논문이 밝히지 않습니다. 샘플링 설정, 청크당 프레임 수는 논문에 없음 |
| D11 | §3.1.1: 카메라 내부 파라미터는 droidcalib, "falling back to AnyCalib for near-static clips" | `stage1_annot_intrinsics.py:44-48,276-300,390-398`: DroidCalib은 움직임이 충분한 **창이 3개 미만**이거나 추정이 안정되지 않으면 `None`을 반환하고 그때 AnyCalib(`anycalib_gen`)으로 대체, 선명한 정지 프레임 30%(최소 50개) 사용 | **미기재(약한 차이).** 대체 조건이 "정지에 가까운 클립"만이 아니라 "DroidCalib 불안정 일반"입니다. 어떤 비율의 클립이 AnyCalib로 갔는지 논문에 없음 |
| D12 | §6: "our current robot overlay does not explicitly model occlusion" | `pipeline/overlay/renderer.py:535-545`: 로봇 의미 마스크의 경계 1.5 px만 부드럽게 하고(`alpha = clip(dist/1.5)`) **깊이·가림 판정 없이** 로봇 픽셀을 사람 인페인팅 장면 위에 덮음 | **일치.** 한계 서술과 맞습니다 |
| D13 | G.3: 관측 해상도는 종횡비를 보존해 효율적으로 줄임, 해상도 수치 없음. 표 8: EPIC 456×256, Ego4D 342×256 | `stage10_lerobot_convert.py:281-282`: LeRobot 출력은 **짧은 변 192**(예: 192×342)로 리사이즈(`--shorter_side 192`). README는 입력 영상을 짧은 변 256으로 변환하도록 안내 | **미기재.** 정책 입력 해상도가 논문에 없음. 공개 데이터가 192 기준이면 사전학습이 그 해상도였을 가능성이 있으나 학습 코드가 없어 확인 불가 |
| D14 | 논문은 비상업 제한을 언급하지 않음 | `README.md` 경고: "This pipeline cannot be run commercially. Its dependencies impose the restriction". `THIRD_PARTY_NOTICES.md`: HAWOR CC BY-NC-ND 4.0, ProPainter S-Lab(비상업), MANO 연구 전용, DroidCalib·ultralytics AGPL-3.0, Isaac Sim NVIDIA Omniverse 라이선스, ALLEX 메쉬 평가 전용. 저장소 자체 코드는 Apache-2.0. 실행 조건: NVIDIA GPU VRAM 24 GB 이상, CUDA 12.8, **Isaac Sim 렌더에 RT 코어 필요(A100, H100 불가)**, 드라이버 R580 이하(R595, R610에서 세그폴트; PR #7이 Isaac Sim 6.0 지원 제안), 설치에 "several hours" | **범위.** 학술 용도 외 사용과 데이터 재배포에 제약이 많습니다 |
| D15 | 표 8: RTX 5090 한 장에서 15분 클립을 8–10배 시간에 처리 | `setup/README.md`: 24 GB 이상, RT 코어 요구. 표 8과 일치하는 조건(5090은 RT 코어 있음). 처리 속도를 코드가 보장하는 장치는 없음 | **일치(방향).** 원시 영상 출처만 8–10×: EPIC 96.2 h + Ego4D 1,589.1 h = 1,685.3 h → 약 13.5k–16k GPU-시간 규모(추정, 15분 클립 비율을 전체에 외삽) |

## 2. 논문과 맞는 부분 (일치 항목)

- **IK 구조 (G.2).** 희소 단계에서 4자유도 정렬 `[tx, ty, tz, yaw]`와 관절을 함께 최적화하고 (`retargeter.py:120-146`의 `Offset4DofVar`, `offset_4dof_to_se3`가 yaw만 회전, `solve_offset_and_joints` 줄 147), 정렬 고정 후 전체 궤적을 시간 평활 항을 더해 푼다(`solve_retargeting` 줄 303-460). 목적항은 국소 방향(코사인), 손끝 위치, 자기 시점 카메라 SE(3) 로그 오차, 관절 한계, 휴식 자세로 논문 G.2 서술과 같습니다. 희소 단계에는 평활 항이 없습니다(`retargeter.py:272` 주석).
- **행동 정의 (식 8).** `a_t = s_{t+1}`, 마지막은 자기 복사 (`stage10_lerobot_convert.py:608-614`, `_next_frame_shift`).
- **관측 카메라 (식 2, 7).** 카메라 궤적을 같은 정렬로 베이스 좌표로 옮겨 "고스트 카메라"로 쓰고(`stage8_robot_retarget.py:362-364`), 로봇 일치 항과 렌더링이 이 카메라를 씁니다.
- **자유도.** `configs/allex.yaml`: 팔 7+7, 손 15+15, 허리 2, 목 2.
- **카메라 진단 컷.** 그림 10의 p99와 일치 (D8).
- **파이프라인 단계.** 100DoH → BoT-SORT(ultralytics) → HAWOR → DROID-SLAM + MoGe-2 + GeoCalib → Qwen3.5 → SAM2/Detectron2/ProPainter → PyRoKi → Isaac Sim (`pipeline/README.md` 표).
- **상태·행동 레이아웃.** `observation.state_eef_*`는 손목 위치 3 + 6D 회전 6의 9차원(`stage10`:625-631), 카메라 포즈는 9차원 `xyz_rot6d`(`_cam_pose_to_xyz_rot6d`). 학습 시 "현재 손목 자세에 대한 `C_obs` 상대 목표" 변환은 이 정보로 가능하지만 변환 코드는 공개물에 없습니다(`03` T1: 변환은 정렬 `ξ`와 무관).

## 3. 라이선스와 공개 범위

- **논문:** arXiv 비독점 배포 1.0(abs 페이지 링크 `nonexclusive-distrib/1.0`, 2026-11-23 확인). CC BY가 아니므로 논문 그림은 게시본에 넣지 않습니다. PDF에서 ©, All rights reserved를 찾지 못했습니다.
- **코드:** Apache-2.0(자체 코드)이지만 파이프라인 실행은 비상업 제한 의존성에 묶입니다 (D14). 서브모듈 패치와 일부 이식 코드(PyRoKi 예제 MIT, 100DoH, Detectron2, SAM2 이식)는 원 라이선스를 따릅니다.
- **데이터:** HF 데이터 카드 라이선스는 출처마다 다르고(EPIC cc-by-nc-4.0 등) gated입니다. EgoDex 부분은 CC BY-NC-ND라 공개되지 않았습니다 (D2).
- **이 폴더:** 복제는 `repo/`에 두고 gitignore, 게시본에는 코드를 넣지 않습니다.

## 4. 구현 가능성 판정 ("구현할 수 있는지")

**판정: 이 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 논문의 결과도, 공개 파이프라인의 실행도 재현할 수 없습니다. 통계 재계산(완료)과 소규모 CPU 검증만 할 수 있습니다.**

### 4.1 이 맥에서 할 수 있는 것

| 항목 | 상태 | 비고 |
|---|---|---|
| 표·본문 수치 재계산, 정수 가능성, 구간·검정 | **완료** | `analysis/recompute.py` (`03`) |
| 정렬 변환 불변성 toy | **완료** | `analysis/toy_alignment_invariance.py` (`03` T1) |
| 공식 코드 정적 대조 | **완료** | 이 문서 |
| IK 단계(stage 8)를 공개 어노테이션 한 샘플에 CPU로 실행 | **조건부** | JAX, jaxls, jaxlie, yourdfpy, pyroki 설치(승인 필요), ALLEX URDF와 메쉬(서브모듈, 메쉬는 평가 전용 라이선스), 입력은 stage 1–7 출력이라 **Hugging Face gated 데이터의 일부 샤드를 받아야** 함(크기 보고 후 승인). JAX CPU 동작은 미확인 |
| LeRobot 데이터 읽기·통계 | **조건부** | pyarrow로 Parquet만 읽는 것은 가벼움. 데이터 접근 승인 필요 |
| 작은 PyTorch 정책 실험(MPS) | **가능(자체 구현)** | 예: 고정 스텝에서 데이터 크기와 에폭을 분리하는 소형 모방학습 toy. 논문 결과의 재현은 아님 |

### 4.2 GPU, 로봇, 데이터가 필요해 맥에서 불가능한 것

- **파이프라인 stage 1–7, 9:** 모두 CUDA 전제입니다 (DroidCalib, HAWOR, DROID-SLAM, SAM2, ProPainter, Qwen3.5-9B ~18 GB). stage 9는 Linux + RT 코어 GPU + Isaac Sim 5.1 + 드라이버 R580 이하입니다. macOS 지원 없음.
- **VLA 사전학습:** GR00T-N1.6-3B, 전역 배치 2048, 80k 스텝. 논문이 GPU 수나 시간을 적지 않았고 학습 코드도 없습니다.
- **평가:** ALLEX 실로봇(두 팔 15자유도 손 2개)과 OpenArm, 4+1과제 수백 롤아웃.
- **데이터 재생성:** 원시 영상 출처만 약 1,685 h(EPIC + Ego4D)이고 8–10× 처리에 약 13.5k–16k GPU-시간(RTX 5090 기준 외삽, 추정). 내려받아야 하는 원본 Ego4D 등도 수 TB 규모.

### 4.3 공수 추정 (측정 전, 추정)

| 작업 | 추정 |
|---|---|
| IK 단계 한 샘플을 맥 CPU에서 (설치 승인과 데이터 샤드 접근 후) | 2–4일 (JAX CPU 환경, 샤드 선택, 입력 형식 맞추기). 결과의 정량 검증(F.3 62.5%/55.2% 재측정)은 자기 접촉 검사를 직접 구현해야 해 +1주 |
| 소형 PyTorch 정책 toy로 에폭 혼입 검증 | 1–2주 |
| 파이프라인 전체를 GPU 서버에서 예제 클립 2개로 | 설치 하루 이상("several hours" 컴파일), 실행은 클립당 수 시간. 이 맥이 아닌 RT 코어 GPU 필요 |
| 논문 표 2 재현 | 해당 없음 (학습·평가 코드, EgoDex, 로봇 없음) |

## 5. 제3자 재현 시 필요한 확인 (저자에게 물을 것)

1. 컵 쌓기 점수 환산 규칙 (`03` V2), 표 4 컵 누들 롤아웃 수 (`03` V4).
2. 에피소드 평균 길이 225.7프레임과 창 210프레임의 관계, 겹침 에피소드의 중복 제거 여부 (D4).
3. EgoDex 경로의 시각 변환(인페인팅, 팔 분할)이 공개 코드와 같은지 (D2).
4. 사전학습 행동 차원(허리·목 포함 여부)과 입력 해상도 (D6, D13).
5. 학습 설정이 단일 학습 1회인지, 시드 분산이 있는지.

# 04. 코드 대조: 논문 ↔ 공식 코드 (정적 읽기)

*대상:* Mobile ALOHA 논문 (arXiv:2401.02117v1, CoRL 2024 PMLR v270). 공식 저장소 두 개를 얕게 복제해 **정적으로만** 읽었습니다. 설치, import, 실행, 데이터 다운로드는 하지 않았습니다. 복제본은 `repo/`에 있고 `.gitignore`로 제외합니다. 복제본의 내용은 신뢰하지 않는 데이터로만 다뤘습니다.

| 저장소 | 커밋 | 마지막 push | 라이선스 | 역할 | 비고 |
|---|---|---|---|---|---|
| `MarkFzp/mobile-aloha` | `0e40324` (2024-01-03) | 2024-06-22 | MIT | ROS 1 기반 4팔 + 기지 수집·재생, 과제 설정 (`aloha_scripts/constants.py`) | 얕은 복제 (6.2 MB) |
| `MarkFzp/act-plus-plus` | `26bab07` (2024-01-10 PST) | 2024-05-15 | MIT | ACT·DP·VINN 학습, 공동 학습 설정 (`imitate_episodes.py`, `utils.py`, `policy.py`) | 얕은 복제 (2.6 MB) |

코드 줄 번호는 위 커밋 기준입니다. 저장소 경로는 `mobile-aloha/…`와 `act-plus-plus/…`로 적습니다.

## 1. 요약

- **맞는 것:** 정책 구조 (인코더 4, 디코더 7, 헤드 8, 공동 학습 정규화 통계, 16차원 행동, 과제별 에피소드 길이, 카메라 3대).
- **다른 것:** 공동 학습 샘플 비율 (논문 0.5 대 공식 설정 0.11–0.5), 공식 설정에만 있는 보조 데이터셋 (논문에 없음), 기지 행동 레이블의 출처 (논문은 CAN, 코드는 바퀴 엔코더), 기지 행동 5프레임 평활 (논문에 없음), 학습률과 배치와 청크 (표 6과 저장소 예시 명령이 다름).
- **논문에 없는 값:** 기지 지연 d = 13 스텝 (코드 하드코딩), 기지 평활 창 5, 시간 앙상블 k = 0.01 (옵션), 백본 학습률 1e-5, weight decay 0.
- **이식 장벽:** `.cuda()` 47곳 (MPS 이식 필요).

## 2. 대조 표

| # | 항목 | 논문 (arXiv v1) | 공식 코드 | 판정 |
|---|---|---|---|---|
| D1 | 트랜스포머 구조 | 표 6: 인코더 4, 디코더 7, 헤드 8, 은닉 512, 피드포워드 3200 | `imitate_episodes.py:75–82`: `enc_layers = 4`, `dec_layers = 7`, `nheads = 8`, `hidden_dim`, `dim_feedforward`는 인자 | 일치 (은닉·피드포워드는 명령 인자로 받음, 예시는 512·3200) |
| D2 | 공동 학습 샘플 비율 | §3, 표 5: 두 데이터셋에서 **동일 확률 0.5 / 0.5** | `mobile-aloha/aloha_scripts/constants.py`의 `sample_weights`: wipe wine·wash pan·chair·shrimp `[5, 5]` (50%), **cabinet `[6, 1, 1, 2]` (정적 20%)**, **elevator `[3, 3, 2, 1]` (정적 11%)**, **high five `[7.5, 2.5]` (정적 25%)** | **불일치** (본문 주장과 다른 설정 3개: 논문 과제 3개가 여기 해당) |
| D3 | 정규화 통계 | §3: 모든 행동 정규화에 **Mobile ALOHA 데이터만** 사용 | 공동 학습 설정은 `stats_dir`로 모바일 데이터만 지정 (cabinet·elevator 포함). 설정이 없으면 `utils.py:249–251`에서 모든 데이터셋 통계를 씀 | 일치 (설정 기준). 기본값은 다름 |
| D4 | 보조 데이터셋 | §5: 과제당 50 시연 (Use Cabinet, Call Elevator 등), 정적 825 에피소드 | cabinet: `aloha_mobile_cabinet_handles` (200), `aloha_mobile_cabinet_grasp_pots` (200). elevator: `aloha_mobile_elevator_2` (1200), `aloha_mobile_elevator_button` (800). wipe wine: `aloha_mobile_wipe_wine_2` (별도 설정) | **미기재** (논문은 과제별 50 시연만 말함). 공식 설정은 보조 시연을 섞음 |
| D5 | 정적 데이터 | §5: 825 에피소드, 12 과제, 탁상 | 모든 공동 학습 설정이 `aloha_compressed_dataset` 이름만 씀. 저장소에 데이터 목록 없음 | **확인 불가** (데이터 미공개 또는 이름만 있음) |
| D6 | 카메라 구성 | §4: 손목 2 + 상단 1, 정적은 정면 카메라 제외 | 과제 설정 `camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']` | 일치 (모바일 설정). 정적 데이터 카메라 이름은 확인 불가 |
| D7 | 행동 16차원, 정적 기지 행동 0 패딩 | §2.2: 관절 14 + 기지 2, 정적 데이터는 기지 행동 0 | `act-plus-plus/utils.py:59–66`: `/base_action`이 있으면 `concat(action, base_action)`, 없으면 0 두 열을 붙임 | 일치 |
| D8 | 기지 지연 보정 | §6: 지연 d 스텝, 팔은 처음 k−d, 기지는 마지막 k−d 사용 (d 값 없음) | `imitate_episodes.py:312–313`: `BASE_DELAY = 13` (실로봇 경로에서만), `query_frequency -= 13`. `:401`: `cat(all_actions[:, :-13, :-2], all_actions[:, 13:, -2:])` | 구조 일치. **d = 13 (50 Hz에서 0.26 s)은 논문에 없음**. 하드코딩 |
| D9 | 기지 행동 평활 | 논문 언급 없음 | `utils.py:282–291`: `smooth_base_action` 창 5 이동평균 (`np.convolve(…, ones(5)/5, 'same')`). 선속도 보정 `calibrate_linear_vel`은 주석 처리 (`:288`) | **논문에 없는 전처리**. 학습 레이블 (데이터 로딩)에만 적용. 추론은 `imitate_episodes.py:450`에서 `action[-2:]`를 그대로 씀 (`postprocess_base_action`은 주석 처리) |
| D10 | 기지 행동 레이블 출처 | §3: 기지 선속도·각속도를 **Tracer CAN 버스**에서 기록 | `mobile-aloha/aloha_scripts/record_episodes.py:156`: `/base_action` ← `ts.observation['base_vel']` = `real_env.py:127–133` `get_base_vel()` = **Dynamixel 바퀴 엔코더** (`/dev/ttyDXL_wheels`) 속도로 계산. 추론 명령은 `real_env.py:198–199` `tracer.SetMotionCommand` | **불일치** (논문: CAN 기록, 코드: 바퀴 엔코더 측정). 레이블은 **명령이 아니라 측정값**이므로 지연을 학습이 떠안음 |
| D11 | 시간 앙상블 | 논문 언급 없음 (청크 + 지연 보정만) | `imitate_episodes.py:402–411`: `temporal_agg` 옵션, 지수 가중 `k = 0.01`, `.cuda()` | **논문에 없는 옵션**. 논문 실험에서 쓰였는지 불명 (act_aloha D2–D7과 같은 질문) |
| D12 | 학습률, 배치, 청크 | 표 6 (ACT): 학습률 **2e-5**, 배치 16, 청크 **45** | 저장소 예시 명령 (`act-plus-plus/commands.txt`): 모바일 사전학습 `pretrain_all` → `--chunk_size 50 --batch_size 24 --lr 1e-4`. 시뮬 예시 `--lr 1e-5`, 청크 50·100 | **불일치 또는 미공개**. 표 6의 조합과 맞는 예시 명령이 없음 |
| D13 | 백본 학습률, weight decay | 논문 표 6에 없음 | `imitate_episodes.py:72`: `lr_backbone = 1e-5` (고정). `policy.py:32`: `weight_decay = 0`, AdamW (`:91`) | **논문에 없는 값** |
| D14 | 공동 학습 샘플링 | §3: 두 데이터셋에서 같은 확률로 뽑아 미니배치 구성 | `BatchSampler` (`utils.py:209–218`)가 `sample_weights`로 데이터셋을 샘플마다 뽑음 | 일치. 배치 안에서 **샘플마다** 데이터셋을 `sample_weights` 확률로 고름 (`utils.py:209–218`). 손실 합산 방식은 논문 식과 같은 평균으로 해석 |
| D15 | 에피소드 길이 (스텝) | 그림 3: wipe 1300, shrimp 3750, wash 1100, cabinet 1500, elevator 2250, chair 2000, high five 2000 | `constants.py`: wipe 1300, shrimp_truncated 3750, wash pan 1100, cabinet 1500, elevator_truncated 2250, chair_truncated 2000, high five 2000 | **일치** |
| D16 | 기록 주파수 | §3: 카메라 480×640, **50 Hz**. 부록 A.2: 카메라 **30 fps** | `constants.py:236`: `FPS = 50`. `record_episodes.py:96, 106`: `DT = 1/FPS`, 주기 대기 `sleep(DT − elapsed)` | 기록 루프 50 Hz **일치**. 카메라 프레임레트는 코드에서 직접 확인 불가 (`real_env.py:66` 주석 "fps 202 vs 30"은 T265 추적 카메라) |
| D17 | 학습 검증 분할 | 논문 미기재 | 과제별 `train_ratio` 0.9 / 0.95 / 0.99, 검증은 첫 데이터셋에서만 뽑음 | 미기재 (논문 재현에 필요한 값) |
| D18 | 학습 장치 | §3: 수집·추론 노트북 RTX 3070 Ti 8 GB. **학습 장치 미기재** | `.cuda()` 전제 (D19) | 학습 시간·장치 **미기재** |
| D19 | 장치 하드코딩 | — | `act-plus-plus`의 `.cuda()`: `imitate_episodes.py` 11곳, `policy.py` 1곳, `detr/main.py` 2곳 등, 최상위 `.py`와 `detr/models`에서 합계 47곳 | 이식 장벽 (맥 MPS에서 장치 인자로 바꿔야 함) |
| D20 | 하드웨어 사양 | §3, 그림 2 | 저장소에 배터리·하중·카메라 사양 없음 (README는 설치 절차). `mobile-aloha/README.md`: Tracer 설정, 카메라 3대 | 확인 불가 (논문 내부 불일치 K12–K14와 연결) |
| D21 | 데이터셋 공개 | 프로젝트 페이지: "Datasets" (Google Drive 폴더) | 저장소에 데이터 없음. 다운로드 안 함 | **확인 불가** (크기·내용 미확인) |

## 3. 재현 관점에서 중요한 차이

1. **공동 학습 비율 (D2):** 논문은 "equal probability"를 공동 학습 방법의 기본으로 씁니다. 공식 설정은 과제 3개에서 다릅니다 (cabinet 20%, elevator 11%, high five 25%). 그래서 논문 표 1의 Use Cabinet·Call Elevator·High Five 값은 **다른 혼합 비율**에서 나온 것입니다. 표 3 (비율 30–70%)은 Wipe Wine 한 과제뿐입니다.
2. **보조 시연 (D4):** 공식 설정은 논문이 말하는 "과제당 50 시연"보다 많은 **보조 시연** (cabinet 400개, elevator 2000개)을 씁니다. 논문 표 1의 "50 demos" 표기가 과제 전체 데이터를 뜻하는지 불분명합니다.
3. **기지 레이블 (D10):** 학습 레이블은 바퀴 엔코더에서 계산한 **측정 속도**입니다. 명령과 측정이 다르면 정책은 명령이 아니라 측정을 흉내 냅니다. 논문 §6의 "기지 지연" 논의와 맞물리는 부분이라 재현에서 중요합니다.
4. **숨은 상수 (D8, D9, D11, D13):** 지연 13 스텝, 평활 창 5, 시간 앙상블 k = 0.01, 백본 학습률 1e-5. 논문 재현은 이 값들을 보고해야 합니다.
5. **학습 설정 (D12):** 표 6의 학습률 2e-5와 청크 45는 저장소 예시 명령과 맞지 않습니다. 실제 과제 학습 명령이 공개되지 않았기 때문입니다.

## 4. 구현 가능성 (코드 기준, 맥: Apple M4 Pro 약 24 GiB, MPS, CUDA 없음)

| 단계 | 필요 자원 | 맥에서 가능 여부 | 근거 |
|---|---|---|---|
| 실로봇 수집·재생 (`mobile-aloha`) | ViperX 300 ×4 (Dynamixel), Tracer AGV (CAN), USB 카메라 3대, ROS 1, `pyagxrobots` | **불가** | `real_env.py`, `launch/`, `setup.py`, `CMakeLists.txt` |
| 정적 코드 검사 | — | **완료** (이 문서) | §2 |
| 수치 재계산 | 보고 값만 | **완료** | `03_analysis.md` |
| 학습 코드 이식 (`.cuda()` 47곳 → 장치 인자) | 코드 수정, 설치 승인 | **조건부** (추정 0.5–1일, 측정 전) | D19 |
| 학습 (ACT 기본, 청크 45, 배치 16) | 데이터 (프로젝트 페이지 링크, 크기 미확인), MPS 학습 시간 | **조건부** (측정 전 판단) | D18: 논문 학습 시간 없음 |
| 공동 학습 재현 | 정적 데이터 (825 에피소드, 미확인) | **불명** (데이터 확보 후) | D5 |
| 시뮬레이션 기지 환경 | 기지 + 팔 2개 시뮬 모델 (저장소에 없음) | **불가 (새로 만들어야 함)** | `act-plus-plus`의 시뮬은 ALOHA 팔 2개 탁상 환경 |

- 요약 판정: **실로봇 재현은 불가. 학습 재현은 데이터를 받고 MPS 이식 후 시간을 잰 뒤 판단합니다.** 공식 코드의 가치는 "구조 대조와 숨은 상수 목록"에 있습니다.
- 노력 추정 (참고): 장치 이식 0.5–1일, 데이터 확보는 크기를 본 뒤 결정 (1 GB 이상이면 보고 후 승인).

## 5. 검증하지 않은 항목

- Diffusion Policy와 VINN 구현 (`act-plus-plus/vinn_*.py`, `detr`, DDIM 설정)은 대조하지 않았습니다 (표 7–9와 비교 필요).
- 기지 평활 (D9)은 학습 데이터 로딩에서만 쓰입니다. 추론 쪽은 `action[-2:]`를 그대로 씁니다 (직접 확인).
- 시간 앙상블 (D11)이 논문 평가에 쓰였는지는 알 수 없습니다.
- 학습 명령 (`commands.txt`)은 시뮬과 사전학습 예시만 있고 Mobile ALOHA 과제별 명령은 없습니다.
- 데이터 크기 (정적 825 에피소드, 프로젝트 페이지 데이터셋)는 확인하지 않았습니다.

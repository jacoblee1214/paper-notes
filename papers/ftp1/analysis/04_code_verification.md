# 04. 공식 코드 대조 (정적 읽기)

대상: `michaelyuancb/ftp1-policy`, 커밋 `89fa681d6c014cce28300946b7526db808e0b1c1` (2026-07-15), 얕은 복제 `repo/` (gitignore). 코드는 **읽기만 했고 실행, import, 설치를 하지 않았습니다.** 저장소 안의 `AGENTS.md`와 `CLAUDE.md`는 데이터로만 읽었습니다.

## 1. 라이선스와 공개 상태

- **코드:** Apache-2.0. `LICENSE` 파일 본문과 GitHub API(`spdx_id = Apache-2.0`)가 일치합니다. `LICENSE_GEMMA.txt`가 따로 있습니다. Gemma 가중치 사용 조건으로 보이며, 이 분석에서는 확인하지 않았습니다.
- **UniVTAC 복사본:** `UniVTAC/LICENSE`는 Apache-2.0입니다. 공식 UniVTAC 저장소(`univtac/UniVTAC`, Apache-2.0)와 같은 라이선스입니다.
- **체크포인트, 데이터셋:** 저장소 README가 HF와 ModelScope 링크를 줍니다. 두 곳 모두 라이선스 정보를 이 분석에서 확인하지 않았습니다. 받지 않았습니다.
- **공개 문장과 실제 공개 상태 (D9).** 논문 v4 초록은 "All model checkpoints, datasets, and training code will be released in the future"라고 씁니다. v1 초록은 이미 공개되었다고 씁니다. 저장소는 2026-06-02에 만들어졌고, 학습 코드와 체크포인트·데이터 링크가 있습니다. 논문 문장과 실제 상태가 맞지 않습니다.
- **하위 모듈:** `third_party/aloha`(Physical Intelligence), `third_party/libero`(LIBERO). 복제하지 않았습니다.

## 2. 대조 결과

판정은 **일치 / 불일치 / 미공개 / 차이 없음(기본값만)** 입니다.

| 번호 | 항목 | 논문 | 코드 (위치) | 판정 |
|---|---|---|---|---|
| D1 | 촉각 전문가 크기 | §2.3: "300M 파라미터". B.5: 폭 1024, 깊이 18, MLP 4096, 헤드 8, 헤드 차원 256 | `ftp1_model_config.py` L71: `tactile_expert_variant = "gemma_small"` (폭 512, 깊이 18, MLP 2048). `gemma.py`의 `gemma_300m`(폭 1024, MLP 4096, "311M params" 주석)은 논문 사양과 같지만 공식 스크립트가 쓰지 않음 | **불일치 (공식 스크립트 기준)** |
| D1a | 공식 스크립트의 변종 | — | `train_univtac_example.sh` L33, `train_univtac_example_swanlab.sh` L44, `pretrain_small/*.sh` L31, `univtac_example_real_machine/*gsmall*.sh` L57이 모두 `gemma_small` | 스크립트는 전부 폭 512 |
| D1b | 배포 체크포인트의 변종 | — | 체크포인트 config는 HF에서 확인하지 못함 (`resolve/main/config.json`이 404). 어떤 변종으로 학습했는지 알 수 없음 | **미확인** |
| D2 | 고유감각 주입 방식 | B.3: 적응형 RMSNorm | 스크립트는 `state_input_mode="adarms"`. 코드 기본값은 `"action_expert"` (`ftp1_model_config.py`) | 논문과 스크립트 일치, 기본값만 다름 (차이 없음) |
| D3 | 사전학습 설정 | App. D: 50k 스텝, 전역 배치 768, 학습률 1e-4 → 5e-5, 48× H20 | 공개 예제 `pretrain_small/train_ftp1_pretrain_small.sh`: 150,000 스텝, 로컬 배치 16 × 8 GPU = 128, 학습률 5e-5 → 3e-6, 워밍업 300. README는 "lightweight example"이라고 씀 | **불일치 (예제는 다른 설정)**, 논문 전체 레시피는 없음 |
| D4 | 파인튜닝 설정 | App. D: 배치 64, 20k 스텝, 학습률 5e-5 → 5e-6, A800 8장 | `train_univtac_example.sh`: 배치 64, 20000, 5e-5 → 5e-6, 워밍업 500 (논문에 없음), 그래디언트 클리핑 1.0 | **일치** (워밍업과 클리핑은 논문에 없음) |
| D5 | 행동 청크 길이 | 본문 §2는 H를 기호로만 씀. 값 없음 | `ftp1_model_config.py` L78: `action_horizon = 32`. README: "FTP-1 default is 32". UniVTAC 평가 스크립트는 청크의 앞 20스텝만 실행합니다(`eval_ftp1_batch.sh` L28, `CHUNK_FIRST_N=20`) | **논문에 없음**. 코드 기본 32 |
| D6 | 행동 공간 차원 | B.1: 팔 2 × (손목 9 + 팔 7 + 손 32) + 머리 9 + 예약 15 | `ftp1_model_config.py` 상수: 손 관절 32, 팔 관절 7, 팔 하나 48, 예약 15. 주석에 총합 120 | **일치** |
| D7 | MTTS 기능 영역 수 | §2.1: 24칸 (0–14 손, 15–20 힘/토크, 21–23 예약) | `FTP1_SINGLE_HAND_NUM_TACTILE_AREAS = 24` | **일치** |
| D8 | 이미지형 인코더 깊이 | B.2: 센서별 깊이 3, 공유 깊이 9, 폭 768 | `t3_tactile_encoder.py` L41 `encoder_depth = 3`, L48 `depth = 9` | **일치** (폭은 확인 안 함) |
| D9 | 공개 문장 | v4 초록: "will be released in the future" | 저장소 2026-06-02 생성, 체크포인트·데이터 링크 존재 | **불일치 (문장과 상태)**, §1 |
| D10 | 블록 어텐션 마스크 | B.4: 행렬 [[1,0,0],[0,1,0],[1,1,1]] (VL, 촉각, 행동) | `ftp1_attention_masks.py` `build_expert_attention_layout`: VL 행은 VL 열만, 촉각 행은 촉각 열만, 행동 행은 VL, 촉각, 자기 열을 봄 | **일치** (정적 읽기) |
| D11 | 데이터 샘플링 배율 | Tab. 8: 출처별 Sample-Scale과 최종 비율 | 저장소에 해당 값이 없음 (0.89, 132.83 등 검색 결과 없음). 예제 데이터 설정 `data_config_pretrain_small.json`만 있음 | **미공개** |
| D12 | UniVTAC 평가 과제 | §3.2, App. E.2: 6과제 (Pull-out Key, Insert Hole, Insert Tube, Lift Bottle, Lift Can, Put Bottle). Grasp Classify와 Insert HDMI 제외 | `UniVTAC/scripts/shell/eval_ftp1_batch.sh` L34–39: 같은 6과제 | **일치** |
| D13 | 시뮬 롤아웃 수 | §3.1: 과제당 100회 | `eval_ftp1_batch.sh` L18: `TOTAL_NUM=${TOTAL_NUM:-100}` | **일치** |
| D14 | 기준선 코드 (π0.5 외) | Tactile-VLA, FTP-π0.5, NTP-1, matched VLA | 저장소 안에 Tactile-VLA 또는 NTP 구현 없음 (검색) | **미공개** |
| D15 | 프로빙 평가 (F.3) | 고정 특징 선형 프로브와 어텐티브 프로브, 시드 42–44 | 저장소 안에 프로빙 코드 없음 (검색: linear_probe, attentive, TacBench, FeelAnyForce) | **미공개** |
| D16 | 실로봇 코드 | 실로봇 과제, 하드웨어 설정 (App. E.1) | 실로봇 제어 코드는 찾아보지 않음 | **미확인** |
| D17 | 사전학습 체크포인트 이름 | 논문: "50k" 사전학습 | README: `ftp1_pretrain_v0426_50kstep`. 논문 v4의 표나 부록은 체크포인트 이름을 적지 않음 | **대응 미확인** (`02` §5) |
| D18 | 추론 장치 | 논문 해당 없음 | `ftp1_inference_wrapper.py` L204, L336, L348: `device="cuda"` 기본, 문서는 `cuda`와 `cpu`만 명시 | 맥(MPS)용 기록 없음 (`mps` 동작은 확인 못 함) |

## 3. 대조 방법의 한계

- 파라미터 수는 공식 `gemma.py`의 사양으로 추정했습니다(`analysis/recompute.py` §10). 임베딩과 정규화는 뺐습니다.
- 체크포인트를 받지 않았으므로 D1b와 D17은 확정할 수 없습니다. 체크포인트 config.json의 HF 경로 404는 파일 경로가 다를 수 있다는 뜻이지, 없다는 뜻은 아닙니다.
- 실행하지 않았으므로 마스크나 정규화의 수치 동작은 확인하지 않았습니다.

## 4. 구현 가능성 (이 맥)

맥 사양은 Apple M4 Pro, 약 24 GiB 통합 메모리, MPS 사용 가능, CUDA 없음입니다.

**판정: 이 맥에서 논문 결과를 재현하는 것은 불가능합니다. 공식 추론을 MPS로 옮기는 것은 조건부로 가능합니다. 학습은 불가능합니다.**

| 작업 | 필요 자원 | 맥 가능성 | 근거 |
|---|---|---|---|
| 사전학습 (50k, 48× H20) | 다중 GPU, 약 3,000시간 데이터 | **불가** | App. D, D3 |
| 파인튜닝 (20k, 8× A800) | CUDA GPU 다수 | **불가** | App. D, README 학습 스크립트 |
| UniVTAC 시뮬 평가 (표 1) | Linux, NVIDIA GPU, Isaac Sim 4.5.0, Isaac Lab 2.1.1 | **불가** | `UniVTAC/Installation_FTP1.md` L10–13 |
| 실로봇 재현 (표 2–3) | Franka, Sharpa 손, Xense/Contactile 센서 | **불가** | App. E.1 |
| 공식 체크포인트 추론 (단일 샘플) | 체크포인트 다운로드 (크기 미확인), MPS 포팅, bf16 동작 확인 | **조건부** | 래퍼는 `cuda`, `cpu`만 문서화 (D18). 포팅과 다운로드는 승인 후 |
| 프로빙 재현 (F.3) | 데이터 (TacBench, FeelAnyForce, 크기 미확인), T3와 FTP-1 체크포인트 | **조건부** | 프로빙 코드는 공개되지 않음 (D15). 특징 추출은 MPS 가능성 있음 |
| 설계 검사 (마스크, 행동 차원, MTTS 슬롯) | numpy 또는 torch CPU | **가능** (재구현 검사) | D6, D7, D10. 다만 이것은 논문 주장의 검증이 아님 |

- **노력 추정 (추정, 측정 전):** 공식 추론의 맥 포팅과 단일 샘플 확인은 1–2주입니다. 프로빙 재현은 특징 추출과 프로브 학습까지 약 1주입니다. 둘 다 승인과 다운로드 크기를 먼저 확인해야 합니다.
- 학습 재현과 실로봇 재현은 이 저장소 밖의 장비가 필요하므로 추정하지 않았습니다.

## 5. 이 분석에서 실행하지 않은 것

- 공식 저장소의 코드, 스크립트, 테스트를 실행하지 않았습니다. `pip`, `uv`, `setup_env.sh`도 실행하지 않았습니다.
- 체크포인트, 데이터셋, 모델 카드를 받지 않았습니다. GitHub 저장소 메타데이터(`api.github.com/repos/...`)와 HF 페이지 응답 코드만 확인했습니다.
- 외부 서비스(SwanLab, Feishu, ModelScope)에 연결하지 않았습니다.

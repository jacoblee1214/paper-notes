# 04. 코드 검증: 페이지 주장 ↔ 저장소 코드, 키트 지표 상수, 구현 가능성

- **대상:** `nvidia-isaac/video_to_data` @ `5226e3474173129545d703aaf5a9761fcd9b3a3d` (2026-11-09). 희소 복제 `repo/` (git 제외). 튜토리얼 HTML은 `docs/v2d_corl_tutorial/index.html`, 챌린지 HTML은 `docs/v2d_challenge/index.html`이고, 둘 다 라이브 페이지와 **바이트 단위로 같습니다** (C1, C2).
- **범위:** 코드와 문서를 읽고 정적으로 대조했습니다. 키트 채점 코드는 "untrusted" 자료로 취급해 실행하지 않고 압축을 풀어 읽기만 했습니다. 저장소 스크립트는 import하거나 실행하지 않았습니다. 재구성, 학습, 추론은 하지 않았습니다.
- **표기:** C = 주장이 코드·문서와 맞음 (확인), D = 페이지·문서와 코드 또는 문서끼리 어긋남 (불일치, 번호는 D1부터).

## 4. 구현할 수 있는지 (판정)

**판정: 맥(Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)에서 튜토리얼의 파이프라인과 실습을 재현하는 것은 불가능합니다. 개념 점검과 검증은 가능합니다.**

근거:
- 저장소 전체가 CUDA 전제입니다. `video_ingestion_agent/configs/ingestion.yaml`은 `device: "cuda"`, vLLM 백엔드를 씁니다. 재구성은 Docker 기반이고 RTX A6000(48 GB)과 L40S로 검증했다고 적혀 있습니다. 학습은 Isaac Lab 2.3.1 Docker와 드라이버 580.126.09 / CUDA 13.0을 권장합니다. 저장소에서 `mps`나 `darwin` 지원을 찾지 못했습니다 (`grep`, 0건).
- 여러 자원이 **등록과 수락**을 요구합니다. MANO (`mano.is.tue.mpg.de`), GR00T 모델 EULA, 허깅페이스 토큰 (`robotic_grounding/docs/SETUP.md` §10).
- 공개된 사전 산출물 중 실제 파일로 받을 수 있는 것은 아직 확인하지 못했습니다. 샘플 영상과 ONNX 정책은 LFS 포인터(132바이트)였고, 예시 체크포인트 저장소는 메타 조회가 인증 오류를 냈습니다 (D24).

| 실습·자산 | 필요한 것 | 맥에서 | 노력 (정성) |
|---|---|---|---|
| 실습 1: 영상 질의 (`video_ingestion_agent`) | vLLM + Cosmos3-Nano (CUDA), SigLIP-2 | **불가** (VLM). SigLIP-2 임베딩만 MPS 조건부 (미검증, 가중치 다운로드 승인 필요) | 임베딩 포팅 수 일 |
| 실습 2: 재구성 (`reconstruction`) | Docker, 모듈별 CUDA 가중치, 48 GB급 GPU | **불가** (Docker 경로). 개별 모듈(MoGe 등)을 MPS에서 돌리는 것은 조건부 (미검증) | 모듈당 수 일 |
| 실습 3: 리타겟 (`robotic_grounding/scripts/retarget/`) | MANO (등록), 데이터셋 6종, Docker 이미지 2개 | **조건부.** 접촉 라벨러와 불변식 검사는 CPU (trimesh 설치 승인 후). 데이터와 MANO 없이는 전체 불가 | 라벨러 포트 반나절 |
| 실습 4: Isaac Lab 학습·평가 | Isaac Lab 2.3.1, Isaac Sim, CUDA | **불가** | — |
| 실습 4 대안: 사전 체크포인트 평가 | HF 인증, Isaac Lab | **불가** | — |
| 실습 5: AI 에이전트 | 스킬 실행 시 Docker·GPU. 스킬 파일 읽기는 가능 | **읽기만 가능** (`.claude/skills` 16개, `.codex/skills` 7개 디렉터리 확인) | — |
| 노트북 대체 점검 경로 (페이지의 "laptop-only") | CPU, numpy | **가능.** 이 폴더의 `page_checks.py`, `gate_toy.py`, 키트 수식 대조가 이에 해당하며 이미 수행함 | 완료 |
| 키트 채점 실행 (Track 1·2·3) | 기준 데이터(호스트 기준값, 키트에서 제외), 메시, MHR 자산, 제출 | **불가** (기준값 없음, 키트 코드 실행은 정책상 금지). 수식 포트는 가능 | 수식 포트 1–2일 |
| 챌린지 데이터 (HF, CC BY 4.0, 파일 551개) | 대용량 다운로드 | **보류** (다운로드 승인 전) | — |

**재현 대신 할 수 있는 것 (맥, CPU):**
1. 게이트 개념과 임계값 표 (03 §1–§2).
2. 기하 게이트 toy (03 §4). 관통·떨림·튐·떠 있음의 FAR/FRR을 임계별로 보기.
3. 키트 지표 수식의 포트와 합성 입력 검사.
4. 저장소 코드 대조 (이 문서).

**규모 감각:** e2e 문서 기준 단계는 9개, 외부 체크아웃 1개 (Isaac-GR00T), 등록형 자산 2개 (MANO, GR00T EULA)입니다. 단일 48 GB GPU에서 "전체 파이프라인 가능"이라고 적혀 있으나 실행 시간을 측정한 기록은 저장소에 없습니다 (주장 미확인).

## 1. 페이지 주장 ↔ 저장소 (C 항목: 확인됨)

| 번호 | 페이지 주장 | 저장소 확인 결과 |
|---|---|---|
| C1 | 튜토리얼 페이지가 저장소에 있음 | `docs/v2d_corl_tutorial/index.html`과 라이브 페이지가 바이트 단위로 같음 (59,314 bytes). 확인 |
| C2 | 챌린지 페이지가 저장소에 있음 | `docs/v2d_challenge/index.html`과 라이브 페이지가 같음 (73,465 bytes). 확인 |
| C3 | 게이트 순서가 저장소 구조와 대응 | `video_ingestion_agent` → `reconstruction` → `robotic_grounding` (retarget) → Isaac Lab / `groot_finetune`로 대응됨. 확인 |
| C4 | 트랙 2 지표 SP-SR 0.10 m·0.5 rad, MP-SR 0.03 m·30° | `metric_code/track_2/SP-SR.py`·`MP-SR.py`의 상수와 `tracks/track_2/README.md` 표가 일치. 확인 |
| C5 | 트랙 1 PEN은 관통 깊이 (cm) | `track_1/PEN.py`: `np.maximum(-distances, 0.0).mean(axis=1) * 100.0`. 확인 |
| C6 | ACC는 가속도 오차 (2차 차분) | `track_1/PEN.py`의 `acceleration_error_per_frame`: `x[:-2] − 2x[1:-1] + x[2:]`. 확인 |
| C7 | MPPE는 물체 축 방향 5 cm 점 6개 | `track_2/MPPE.py`: 반지름 0.05 m, 방향 ±x, ±y, ±z. 확인 |
| C8 | AUC는 1–9 cm 임계 적분 | `track_2/AUC.py`: `np.arange(0.01, 0.10, 0.01)`, 사다리꼴 (x = linspace 0–1). 확인 |
| C9 | 성공 판정 lift-hold 임계 (0.10 / 0.05 m, 20 샘플) | `closed_loop/evaluation.py` 115–118행, `tissue_box_lift_hold.json` 값과 일치. 확인 |
| C10 | 수집 시도 수와 미세조정 스텝 식 | `tools/e2e/stages.py` 676–700행, `cli.py` 409행 (안전계수 1.125). 문서 식과 일치. 확인 |
| C11 | 감사 통과 (`ok: true`) | `audit_groot_run.py` 710행 `"ok": not audit.errors`. 확인 |
| C12 | 리타겟 결과의 유한성 검사 | `ego_recon_to_dexmate_sharpa.py` 308행 `np.isfinite`. 확인 (프레임 수 일치는 부분 확인) |
| C13 | 실습 1 산출물 JSONL 클립 목록 | 분할 결과를 JSONL로 쓰는 함수가 있음 (`_write_segment_jsonl`, 테스트 `test_reconstruction_service.py`). 확인 |
| C14 | 실습 5 AI 에이전트 | `.claude/skills/` 16개, `.codex/skills/` 7개 (`ingestion_agent_onboard` 등). 확인 |
| C15 | Tier 3 기준선 CARI4D 제공 | `reconstruction/modules/v2d_cari4d/` 모듈 존재. 라이선스는 D8 |
| C16 | 트랙 2 G1 + Dex3, 트랙 3 Sharpa Wave | `robotic_grounding/scripts/retarget/arctic_to_dex3.py`, `taco_to_dex3.py`, `sharpa` 리타게터, `v2d_whole_body/` 존재. 확인 |
| C17 | Isaac Lab 2.3.1 | `robotic_grounding/README.md`, 루트 README. 확인 (문서 기준) |
| C18 | 게이트 2 다시점 QC 임계 | `mv_hoi/config.yaml`의 `qc_thresholds` 블록과 `campaign_controller.py` 725–740행의 기본값이 같음. 확인 |

## 2. 불일치 (D 항목)

표의 "심각도"는 튜토리얼 이해와 재사용에 주는 영향입니다. **높음**: 주장이 코드와 다르거나 검사가 기본값에서 꺼짐. **중간**: 문서가 코드와 다름. **낮음**: 오타·정리 문제.

### 2.1 실습 자료와 페이지 문구

| 번호 | 페이지·문서 주장 | 저장소 실제 | 심각도 |
|---|---|---|---|
| D1 | 실습별 "runnable scripts or notebooks", "laptop-only inspection path" | `.ipynb` 0개, "laptop" 0건. "exercise" 문자열은 페이지 HTML과 테스트에만 있음. 페이지는 자료를 "after the event" 공개한다고 적음. 2026-11-12 기준 미이행 | 높음 (실습 재현 경로 없음) |
| D2 | 트랙 2 지표는 "as defined in the CHORD manuscript" | (a) CHORD 본문은 지표 출처만 밝힘 (SPIDER, ManipTrans, DexMachina, `chord/paper/paper.txt` 394행). 임계 숫자는 찾지 못함. 정본은 키트 코드. (b) `chord/analysis/02_prior_work.md`는 SP-SR이 "궤적 평균 중심화"라고 해석하지만 키트 `SP-SR.py`는 첫 프레임 정렬 뒤 시간 평균만 씀. chord 분석과 다름 (읽기 전용이라 수정하지 않음) | 중간 |
| D3 | 우승자: "Three winners per track"과 "Two winners per track" | 같은 페이지 안에서 수 불일치. 상금 $54K은 트랙 2를 티어 3개로 셀 때만 맞음 (03 N1) | 낮음 |
| D4 | 제출 한도: "up to five times per week" | FAQ: "The limit applies per track and tier, not across the entire challenge" | 중간 |
| D5 | 코드 공개: 페이지 "agreed code release (research license)" | FAQ: 우승 제출은 "Apache 2.0 license"로 공개 요청 | 중간 |
| D6 | 트랙 1: "Each axis is weighted equally" (페이지) | FAQ: 최종 순위는 지표별 순위 합. 두 설명이 어떻게 맞는지 명시 없음 | 낮음 |

### 2.2 챌린지 데이터·기준선

| 번호 | 페이지·문서 주장 | 저장소·키트 실제 | 심각도 |
|---|---|---|---|
| D7 | 리더보드가 실시간 | 저장소 `docs/v2d_challenge/leaderboard.json`은 2026-09-29 생성, "No submissions scored yet". 실시간 데이터는 별도 저장소 (2026-11-09 스냅샷, 01 §4.4). 저장소 사본은 갱신되지 않음 | 낮음 |
| D8 | Tier 3 기준선 "CARI4D"를 "off the shelf"로 제공 | 저장소 체크포인트 이름 `nvidia/cari4d_commercial` (`v2d_cari4d/README.md` 95행). 리더보드 주석: "commercial friendly recreation of CARI4D with retrained weights and some non-commercial modules replaced. Performance may differ slightly". `reconstruction/NOTICE`에 항목 없음 | 중간 (원본 아님, 라이선스 미확정) |
| D9 | Tier 2 교란: "jitter, dropout, and contact errors ... sampled from Track 1's error distributions" | 교란 생성 코드와 분포가 저장소·키트에 없음. 키트에는 폴더명 `synthetic_random_walk`만 있음. 검증 불가 | 중간 |

### 2.3 게이트 1–5의 자동 검사 (코드 ↔ 설명)

| 번호 | 설명 | 코드 실제 | 심각도 |
|---|---|---|---|
| D10 | `min_entity_confidence`, `min_relationship_confidence` (0.5) — 문서 `configuration.rst`는 "Minimum confidence to keep an entity" | `config.py` 151–153행에서 정의만 함. `video_ingestion_agent/src`에서 읽지 않음 (grep 0건). **죽은 설정**. 필터링 없음 | 중간 |
| D11 | VLM 검증 점수 (`verification_score`) | 저장만 함. 유효 판정은 `is_correct` 불리언 (`critic.py` 138행). 리포트 HTML에는 점수가 표시됨 (`report.py` 196행) | 중간 |
| D12 | 페이지: 관통이 "physical plausibility"의 한 축 | 재구성 기본 파이프라인에서 관통 손실 가중치 0.0 (`run_ego_reconstruction.py` GSPLAT_DEFAULTS `w_hand_object_penetration`, `run_ego_wilor.py` `simple_refinement_w_hand_object_penetration = 0.0`). 관통 검사는 평가 지표 PEN에만 있음 | 높음 (기본값에서 꺼짐) |
| D13 | 시간 평활 (떨림 억제) | 정점 가속도 평활 가중치 100 (켜짐), 척도 0.01 m. 손실 가중치이며 통과·기각 임계가 아님 | 낮음 (해석 주의) |
| D14 | 접촉 라벨 = 손과 물체의 접촉 | `contact_labels.py`는 `query.on_surface`의 **부호 없는** 거리로 `dist < threshold`를 봄. 법선은 부호와 무관하게 `-face_normals` (177행). 관통을 구분하지 못함. toy에서 깊은 관통의 99%가 이 규칙을 통과 (03 §4) | 높음 |
| D15 | 손바닥–엔드이펙터 불변식 | `planner/utils/validation.py` 264–267행: dex3 0.01 m, sharpa 0.10 m. 주석은 sharpa의 고정 오프셋 약 4 cm라고 함. 10 cm 허용은 4 cm 오차를 통과시킴 | 중간 |
| D16 | 수집 타임아웃 상한 = `ceil(frames / fps / speed × control_fps)` 를 "retarget 보고서에서 추정" | 식은 문서에만 있음. 코드는 `--collection-max-steps`를 필수 입력으로 받음 (`cli.py` 411–412행). 자동 계산 없음 | 낮음 |
| D17 | 시뮬 폐루프 "87 of 100 full cycles ... trained with 100 rollouts" (그림 캡션) 대 본문 "200 successful simulation rollouts" | 같은 문서 안에서 롤아웃 수가 다름. 같은 실험인지 원자료 없음 | 중간 |
| D18 | 실로봇 V2D Sim200→Real10 4/5, 원시 0/5 (표본 5) | 로그와 영상이 저장소에 없음. Fisher p = 0.048 (03 N2). 검증 불가 | 중간 |
| D24 | 예시 체크포인트 `nvidia/video-to-data-robotic-grounding-example` (`SETUP.md` §10) | 허깅페이스 메타 API가 "Invalid username or password" 반환. 비공개이거나 토큰 필요로 보임. 접근 조건 확인 못함 | 낮음 |

### 2.4 문서 정리 문제

| 번호 | 내용 | 위치 | 심각도 |
|---|---|---|---|
| D19 | 설치 명령 `./scripts/install_pacakages.sh` (파일명 오타). 실제는 `install_packages.sh` | 루트 `README.md` 109행. `reconstruction/README.md`는 맞음 | 낮음 |
| D20 | "18 containerized modules" | 루트 `README.md` 57행. Docker 디렉터리를 가진 `v2d_*` 모듈은 31개 (정의 차이 가능) | 낮음 |
| D21 | 이미지 이름 `nvcr.io/your-org/your-team` | `robotic_grounding/README.md` 14행. 자리표시자 | 낮음 |
| D22 | 개인 홈 경로가 기본값으로 들어 있음 | `reconstruction/modules/v2d_mv/cli/tile_videos.py` 21행 (경로 자체는 여기 적지 않음) | 낮음 (재사용 시 수정) |
| D23 | 저장소는 공개 상태 | `video_ingestion_agent/docs/release_readiness.md`의 판정은 "partial; ... not yet safe to flip the repo public". 공개 전환 준비 문서가 함께 올라가 있음 | 낮음 (성숙도 신호) |

## 3. 키트 채점 상수 대조

`analysis/page_checks.py` (키트 경로를 인자로 주면 정규식으로 확인):

| 항목 | 결과 |
|---|---|
| SP-SR 위치 임계 0.1 m | 일치 |
| SP-SR 회전 임계 0.5 rad | 일치 |
| MP-SR 위치 임계 0.03 m | 일치 |
| MP-SR 회전 임계 30° | 일치 |
| AUC 임계 0.01–0.09 m (9점) | 일치 |
| MPPE 반지름 0.05 m | 일치 |
| PEN 식 (`relu(−sdf)` 평균 ×100) | 일치 |
| 접촉 정확도 임계 0.015 m (`PEN.py`, 리더보드 미사용) | 일치 |
| RPE 샘플 수 10,000 (`track_3/RPE.py`) | 일치 |

키트 `metric_code/README.md`는 "code-only copies ... host reference values, object meshes, the MHR model asset ... are omitted"라고 적습니다. 그래서 키트만으로는 채점 결과를 재현할 수 없습니다.

## 4. 페이지에 없고 코드에만 있는 사실

- 챌린지 채점은 Kaggle에서 계산됩니다. 키트의 `submit/v2d_submit.py`가 제출을 올리고, 제출 파일에는 정책 체크포인트 SHA-256, 평가 스크립트 해시, 커밋 URL이 들어갑니다 (`tracks/track_2/README.md`). 재현 가능성 요건이 제출물에 포함됩니다.
- 트랙 2 결과 파일은 **교란 입력에 대한 추적** 기준이고, 점수는 **깨끗한 기준 궤적**에 대해 계산됩니다 (키트 README). 페이지 설명과 같습니다.

## 5. 미확인 항목

- 실제 재구성 정확도 분포, 게이트 임계의 실데이터 민감도: 데이터를 받지 않아 확인하지 못했습니다.
- 개방루프 MSE·MAE 계산: 외부 Isaac-GR00T 환경.
- 리타겟 QA "warnings 없음"이 자동 실패로 이어지는지: 보고서 생성 코드 일부만 읽음.
- CARI4D 라이선스: 모듈 README와 NOTICE에 명시가 없음 (D8).
- MANO 라이선스 원문: 저장소에는 등록·수락 절차만 있음.
- 챌린지 Tier 2 교란 분포 (D9).
- 저장소 `robotic_grounding/` 전체 (chord 분석과 중복되므로 다시 읽지 않음).

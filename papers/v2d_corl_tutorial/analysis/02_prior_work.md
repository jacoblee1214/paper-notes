# 02. 선행연구 계보, 관련 저장소, GitHub 목록

## 1. 계보 (튜토리얼이 기대는 연구)

튜토리얼은 새 방법을 내세우지 않고 **이미 나온 구성 요소를 한 줄로 잇는 파이프라인**을 가르칩니다. 구성 요소와 출처는 다음과 같습니다.

| 단계 | 구성 요소 | 출처 (저장소 안 근거) |
|---|---|---|
| 영상 질의 | 구간 분할 VLM, 검증 루프, 개체 그래프, SigLIP-2 임베딩 | `video_ingestion_agent/` (LangGraph). VLM은 `nvidia/Cosmos3-Nano`, 임베딩은 `google/siglip2-base-patch16-256` |
| 재구성 | 깊이 MoGe·UniDepth·Depth-Anything 3, 마스크 SAM2, 물체 SAM3D·BundleSDF, 6D 자세 FoundationPose, 손 HaMeR·WiLoR·HaWoR, 몸 SAM3D-Body (MHR), 다시점 보정 | `reconstruction/` (`NOTICE`에 모듈별 라이선스) |
| 로봇 과제 자산 | 재구성 메시를 USD로, Isaac Sim 낙하 시험, 지지 자세 추정 | `reconstruction/modules/v2d_hoi_object_reconstruction/mesh_to_usd/` |
| 리타겟 | 사람 손 MANO 모델 → Sharpa·Dex3·Vega, 접촉 라벨 | `robotic_grounding/scripts/retarget/` (`ego_recon_to_dexmate_sharpa.py`, `contact_labels.py`) |
| 정책 학습 | Isaac Lab + RSL-RL PPO, 추적·접촉 보상, VOC 커리큘럼, FlashSAC 가속 | `robotic_grounding/source/`, `flash_chord/`. 논문 근거는 **`chord/`** |
| 후속 학습 | 전문가 롤아웃 → LeRobot 데이터 → GR00T N1.7 미세조정·폐루프 평가 | `robotic_grounding/groot_finetune/`, `docs/e2e_workflow.md` |
| 평가 지표 | AUC (DexMachina), SP-SR (SPIDER), MP-SR (ManipTrans), MPPE, RPE, PEN·ACC·CD (MHR 기반) | 키트 `metric_code/`. 페이지 "as defined in the CHORD manuscript" |

지표의 계보가 중요합니다. CHORD 본문은 "AUC from Dexmachina, SP-SR from SPIDER, MP-SR from ManipTrans"라고 적습니다(`chord/paper/paper.txt` 394행). 튜토리얼 챌린지는 이 셋과 MPPE를 그대로 씁니다. 다만 임계값은 각 원 논문과 키트에서 다를 수 있으므로, 숫자는 키트 코드를 기준으로 봅니다(`04` §3).

## 2. 같은 저장소의 기존 분석 (재사용)

- **`chord/`**는 같은 코드 저장소의 `robotic_grounding/`을 이미 분석했습니다. 대상은 CHORD (arXiv:2607.00033, NVIDIA, CoRL 2026)이고, 공식 코드는 `9cad120`입니다. 이 폴더는 그 결과를 다시 계산하지 않고 인용만 합니다.
  - 보상 식과 코드 차이 (D1: 논문 Eq. 3은 코드와 다르고 코드는 방향별 exp 평균) → `chord/analysis/04_code_verification.md` §2
  - 종료·성공 임계 세 가지 (D3) → `chord/analysis/04_code_verification.md` §2
  - PPO 경로와 FlashCHORD 경로의 예산 차이 → `chord/analysis/02_prior_work.md` §3
  - 구현 가능성: 맥에서 Isaac Lab 재현 불가 → `chord/analysis/04_code_verification.md` §4
- **`dust/`, `reasoning_retargeting/`**는 웹 출처 분석의 형식을 본떴습니다. `reasoning_retargeting`은 에이전트 루프를 쓴 블로그 리타게팅이라 튜토리얼 실습 5(에이전트)와 가장 가깝습니다.

## 3. 이 저장소의 다른 분석과의 관계

| 폴더 | 관련성 | 튜토리얼의 어느 게이트와 맞닿는지 |
|---|---|---|
| [`chord/`](../../chord/) | **가장 큼.** 같은 코드 저장소, 같은 학습 파이프라인, 같은 지표 계열 (AUC·SP-SR·MP-SR·MPPE) | 게이트 5 (정책 학습), 트랙 2 지표 정의 |
| [`phuma/`](../../phuma/) | 높음. 영상과 모캡으로 만든 휴머노이드 데이터셋의 **큐레이션 임계**와 리타게팅 품질 검사 | 게이트 4 (리타겟 품질 검사의 설계 방식 비교) |
| [`warp_retargeting/`](../../warp_retargeting/) | 중간. 사람 상체 시연 → 로봇 **전신** 행동 리타게터 | 트랙 2 전신 (G1 whole-body), 게이트 4 |
| [`reasoning_retargeting/`](../../reasoning_retargeting/) | 중간. 에이전트 루프로 리타게팅 기준을 고침 | 실습 5 (AI 에이전트), 게이트 4 검증을 자동화하는 방식 |
| [`autodex/`](../../autodex/) | 중간. 실제 로봇에서 파지 데이터를 모으는 시스템 | 게이트 3 이후 실제 로봇 검증, 시뮬레이션–실물 격차 |
| [`do_as_i_do/`](../../do_as_i_do/) | 높음. 단안 일상 영상 → 4D 손·물체 복원 → 손 리타게팅 (SPIDER 사용) | 게이트 2 (재구성 품질 검사) 와 게이트 4 |
| [`flashdexretarget/`](../../flashdexretarget/) | `chord/` 분석에서 자주 인용됨. SP-SR 지표가 관대하다는 지적 (`chord/analysis/02_prior_work.md` 30행) | 트랙 2 지표 해석 |
| [`goaldex/`](../../goaldex/) | 낮음. 이미지 목표 조건부 손 내 조작 정책. CoRL 2026 | 튜토리얼 직접 관계 없음 |

## 4. GitHub 목록

| 저장소 / 자원 | 역할 | 라이선스 | 최근 갱신 (2026-11-12 기준) | 이 폴더에서 쓴 방식 |
|---|---|---|---|---|
| [nvidia-isaac/video_to_data](https://github.com/nvidia-isaac/video_to_data) | 튜토리얼 저장소 (V2D 전체) | `LICENSE`: 문서·스킬 CC-BY-4.0, 소스 Apache-2.0 (GitHub 표시 NOASSERTION). 스타 968 | 2026-11-09 push, 커밋 `5226e34` | 희소 복제 `repo/` (작업 트리 약 30 MB, `.git` 약 193 MB, 합계 약 223 MB < 1 GB). 미디어 제외. 읽기 전용 |
| [Nvidia-V2D-Challenge](https://github.com/Nvidia-V2D-Challenge) (조직) | 챌린지 제출 저장소, 정책 코드 공개 대상 | 미확인 | — | 접근하지 않음 |
| [MVerghese/v2d-leaderboard-data](https://github.com/MVerghese/v2d-leaderboard-data) | 챌린지 페이지가 읽는 리더보드 JSON | 미기재 | 2026-11-09 스냅샷 (`generated_at`) | 집계만 인용 (01 §4.4) |
| [HF `nvidia/video_to_data_challenge`](https://huggingface.co/datasets/nvidia/video_to_data_challenge) | 챌린지 데이터셋 (트랙 1–3) | CC BY 4.0 (카드 메타) | 2026-09-22 수정, 파일 551개 | 메타데이터만. 다운로드 안 함 |
| [HF `nvidia/video-to-data-robot-dexterity-task-library-and-dataset`](https://huggingface.co/datasets/nvidia/video-to-data-robot-dexterity-task-library-and-dataset) | 과제 라이브러리 4,739개 (`chord/`에서 확인) | CC BY 4.0 | 2026-08-21 | 메타데이터만 |
| [HF `nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano) | 게이트 1 VLM (`vlm_model`) | OpenMDW-1.1 (카드 메타) | 2026-11-09 | 메타데이터만. vLLM CUDA 기본 |
| [HF `google/siglip2-base-patch16-256`](https://huggingface.co/google/siglip2-base-patch16-256) | 게이트 1 프레임 임베딩 | Apache-2.0 (카드 메타) | 2025-02-21 | 메타데이터만 |
| HF `nvidia/video-to-data-robotic-grounding-example` | 예시 RL 체크포인트 (`robotic_grounding/docs/SETUP.md` §10) | 조회 실패 | — | 메타 API가 "Invalid username or password"를 반환. 접근 조건 미확인 (D24) |
| [facebookresearch/MHR](https://github.com/facebookresearch/MHR) | 트랙 1 MHR 몸 모델 원본 (키트가 축소본 포함) | Apache-2.0 (키트 `NOTICE.md`) | — | 접근하지 않음 |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | 게이트 5 GR00T 미세조정·평가 환경 (외부 체크아웃) | 미확인 (EULA 수락 필요) | — | 접근하지 않음 |
| [NVIDIA/simready-foundation](https://github.com/NVIDIA/simready-foundation) | 자산 구조 검증 (`simready-validate`) | Apache-2.0 (`reconstruction/NOTICE` 386–390행) | 핀 커밋 `4d9f3bb` | 저장소 NOTICE만 확인 |

## 5. 라이선스 표 (재배포하지 않는 제3자 모델 포함)

저장소 안에서 확인한 항목만 적습니다. 출처는 `reconstruction/NOTICE` 행 번호입니다.

| 구성 요소 | 라이선스 | 비고 |
|---|---|---|
| 저장소 소스 (NVIDIA 작성) | Apache-2.0 | `LICENSE` 본문 |
| 문서·스킬 | CC BY 4.0 | `LICENSE` 머리글 |
| MoGe | MIT | 378행 |
| SAM 2 | Apache-2.0 | 384행 |
| Depth-Anything 3 | Apache-2.0 | 55행 |
| Depth-Anything (v1) | Apache-2.0 | 411행 부근 |
| Grounding DINO | Apache-2.0 | 396행 |
| DROID-SLAM | BSD-3-Clause | 402행 |
| HaMeR | MIT (MANO 별도) | 420행 |
| BundleSDF | Apache-2.0 (NVIDIA 작성, 제3자 구성 요소는 각자 라이선스) | 366행 |
| FoundationPose | NVIDIA Source Code License (비상업 연구·평가) | 320행, 저장소 안에 포함 |
| SAM3D-Body | SAM License (Meta, 상업 이용 가능하나 사용 제한) | 326행, 저장소 안에 포함 |
| UniDepth | CC BY-NC 4.0 | 348행, 내려받아 씀 |
| WiLoR | CC BY-NC-ND 4.0 | 354행, 내려받아 씀 |
| HaWoR | CC BY-NC-ND 4.0 (MANO 별도) | 360행 |
| FoundationStereo | NVIDIA Source Code License (비상업) | 372행 |
| CARI4D | **NOTICE에 항목 없음.** 체크포인트 이름 `nvidia/cari4d_commercial` | D8 |
| MANO 손 모델 | 등록 후 수락 (`mano.is.tue.mpg.de`) | 저장소에 재배포 안 함. 라이선스 원문은 확인하지 않음 (외부 지식으로는 비상업 연구용) |
| Cosmos3-Nano (VLM) | OpenMDW-1.1 | HF 카드 |
| SigLIP-2 base | Apache-2.0 | HF 카드 |
| vLLM 0.12.0, LangGraph 1.1.10 | Apache-2.0, MIT | `video_ingestion_agent/NOTICE` |
| GR00T N1.7 | NVIDIA Open Model License (수락 필요) | `docs/e2e_workflow.md`. 원문은 확인하지 않음 |
| Isaac Lab 2.3.1, RSL-RL | 외부 지식으로는 BSD-3-Clause (원문 미확인) | 저장소 안에서 확인하지 않음 |

**시사점:** 게이트 2는 NC·ND 모델을 여러 개 씁니다. 결과를 상업 제품에 넣는 경우 모델별 조건을 따로 봐야 합니다. 챌린지 Tier 3 기준선도 "commercial" 체크포인트라고 적혀 있지만 NOTICE에 항목이 없어 라이선스를 확정하지 못했습니다.

## 6. 참고: 챌린지 지표 계보와 CHORD의 해석 차이

처음 `chord/` 분석은 SP-SR을 "위치를 궤적 평균으로 중심화하는 관대한 지표"라고 적었습니다(`flashdexretarget/04` D3 인용). 이것은 SPIDER 원 스크립트의 양손 모드 동작입니다. CHORD 공개 평가 코드(`flash_chord/evaluation/metrics.py: spider_success`)와 키트 트랙 2 `metric_code/track_2/SP-SR.py`는 중심화하지 않은 원 위치 오차를 시간 평균(`mean(axis=0)`)해 임계와 비교합니다. 트랙 2는 `_ALIGN_INITIAL_POSE = False`라서 첫 프레임 정렬(`_warp_to_reference_start`)도 하지 않고, 정렬은 트랙 3 파일에서만 켜집니다. 세 정의의 차이는 `03` 추가 검증과 `04` D2에 있습니다. 이 확인에 따라 `chord/`의 해당 설명은 2026-11-12에 고쳤습니다(`chord/analysis/04_code_verification.md` D13).

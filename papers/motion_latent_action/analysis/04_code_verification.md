# 04. 코드 대조와 주장 검증

공식 코드는 얕게 clone해 분석 폴더 안에 두었습니다(`repo/`, 저장소 `.gitignore`로 제외). 대조 기준 커밋은 **`fd614f7`(2026-06-18, "update readme")**입니다. 저장소는 Apache-2.0(`LICENSE`), 별점 0, 마지막 push 2026-06-18입니다. v2 초록의 코드 링크는 2026-07-02에 추가되었으므로, 이 커밋은 v1 시점의 스냅샷에 가깝습니다. 태그나 릴리스는 없습니다. 대조 범위는 잠재 행동 모델(`latent_action_model/`), VLM 사전학습과 미세조정(`vla-scripts/`), 데이터 로더(`prismatic/vla/datasets/`)입니다. 실행은 하지 않았습니다(CUDA 전제, §4).

## 1. 저장소 구성

| 경로 | 내용 | 논문과의 관계 |
|---|---|---|
| `latent_action_model/genie/modules/lam.py` (229줄) | 이중 코드북 VQ-VAE, 세 디코딩 경로 | III-B 전부 |
| `latent_action_model/genie/model.py` (481줄) | Lightning 학습 루프, 손실, 코드 사용률 | III-B.4, Eq. 1 |
| `latent_action_model/genie/dataset.py`, `prismatic/vla/datasets/` | RLDS 로더, 쌍 샘플링, 마스크 전달, 증강 | III-B 입력, 실험 설정 |
| `latent_action_model/config/*.yaml` | EgoDex / Bridge 설정 | IV-A, IV-B |
| `vla-scripts/train.py` | VLM 사전학습(잠재 토큰 NLL) | III-C, Eq. 2 |
| `vla-scripts/finetune_libero.py` | LIBERO 미세조정(L1 헤드) | IV-A |
| `vla-scripts/finetune_robotwin_flow.py` (1166줄) | RoboTwin 미세조정(flow-matching 헤드) | III-D, IV-B |
| `vla-scripts/deployment_real.py` | 실물 배포 | IV-B 실물 |
| 없음 | 데이터 전처리, SAM2·RoboEngine 마스크 생성, 학습된 체크포인트 | README: "data preprocessing script will be released upon finalization" |

## 2. 논문 ↔ 코드 대조 (D 항목)

판정: **일치**(논문과 코드가 같음), **차이**(코드가 논문과 다름), **누락**(논문은 있으나 코드가 없음), **버그**(동작 오류), **미확인**(코드만으로 알 수 없음).

| ID | 논문 주장(위치) | 코드(파일:줄) | 판정 |
|---|---|---|---|
| D1 | SAM2로 사람 손 마스크 생성(III-B.3) | 로더는 `egodex_with_sam2_mask_all` 데이터의 `mask` 키를 읽음(`lam-stage-2-egodex.yaml`, `configs.py:79–80`). 생성 코드 없음 | 누락(생성 파이프라인) |
| D2 | 로봇 마스크는 RoboEngine(IV-A) | Bridge 설정은 `bridge_oxeall`의 `mask` 키를 읽음(`lam-stage-2-bridgeall.yaml`). 생성 코드 없음 | 누락 |
| D3 | 행동 경로는 "전경 영역"만 감독(III-B.3) | `model.py:291–299`: 주의 맵 임계값 0.5로 이진화. 실제 전경 조건은 마스크 면적 > 47%(03 §3.4). 맵의 연속 가중치(`lam.py:137–139`)는 쓰이지 않음 | 차이(임계값은 논문에 없음) |
| D4 | 전체 재구성 경로(III-B.3) | `lam.py:204–213`, `model.py:320`: 마스크 없는 전역 MSE, 가중치 1. 행동 코드가 배경·밝기를 설명하는 통로 | 일치(논문도 전역 항 포함) — 누수 통로(03 §4) |
| D5 | Eq. 1의 λ_recon, λ_vq, λ_commit | 재구성 항 가중치 1, 1, 1(`model.py:320–325`), VQ 가중치 1, 커밋 β = 0.25(`model.py:357`, `vq_beta`). 논문은 값을 주지 않음 | 미확인(논문 누락) |
| D6 | 코드북 크기 16 두 개, 쌍당 토큰 4(III-B.2) | `lam.py:68–79`(`num_latents=16`), `lam.py:46`(`num_codes=4`) | 일치 |
| D7 | VQ 재시작(논문 언급 없음) | `model.py:440–441`: 에포크마다 **배경 코드북만** `random_restart()`. 행동 코드북은 재시작하지 않음. 재시작은 `randperm`으로 기존 코드를 복사(`blocks.py:435–445`) | 차이(행동 코드북 사용률 관리 없음; 재시작 방식이 죽은 코드에 기존 코드를 복사) |
| D8 | "1초 간격 쌍"(III-B 서두) | `dataset.py:357–363`: 설정 목록에 없는 데이터셋은 창 크기가 기본값. `datasets.py:337`의 기본 `window_size=10`이 그대로 전달되어 쌍 간격은 **9프레임**. 목록에 없는 EgoDex와 Bridge 모두 해당. 영상 fps 확인 필요 | 차이(1초는 fps 9에서만 성립) |
| D9 | (논문 언급 없음) 증강 | `datasets.py:383–410`: 영상에 `random_resized_crop(scale 0.9)` 적용. 마스크 증강은 주석 처리(`#"mask": mask_augment_kwargs`). EgoDex 설정은 `image_aug: true`. `obs_transforms.augment`는 `image_*` 키 중 지정된 것만 자름. 두 프레임이 같은 자르기 변수를 쓰는지는 dlimp 내부 구현에 달려 있어 미확인 | **버그 가능성**: 이미지와 마스크 정렬 불일치(03 §3.5). 프레임 단위 독립 자르기라면 가짜 움직임 |
| D10 | 코드북 크기 16, 행동 토큰 4(III-C) | `vla-scripts/train.py:42`(`codebook_size=16`), 특수 토큰 `<ACT_i>` 16개 | 일치 |
| D11 | Prismatic-7B 백본(III-C) | `train.py:38`(`prism-dinosiglip-224px+7b`), README의 HF 링크 | 일치 |
| D12 | 디코더와 VLM 어휘 연결(III-C, Eq. 2) | `train.py:186–218`: 학습된 LAM으로 쌍의 토큰을 만들어 VLM 목표로 사용 | 일치(사전학습 데이터 규모 미기재) |
| D13 | 의도 예측 CE, λ_intent(Eq. 5) | `finetune_robotwin_flow.py:1076`: VLA 미동결 시 **L = 10·L_flow + L_CE**(상대 가중 0.1). 동결 시 L = L_flow로 CE 없음 | 차이(λ 값은 논문에 없음; 기본 설정은 동결) |
| D14 | VLM은 LoRA로 미세조정(III-D) | `finetune_robotwin_flow.py:855`(`freeze_vla=True`), `:857`(`use_lora=False`). `finetune_libero.py:175–176` 동일. 기본값은 **동결 + LoRA 없음**. LoRA 설정은 실행 경로 이름(`lora-r32`)에서만 드러남 | 차이(기본값이 논문 주 설정과 다름) |
| D15 | 미세조정 흐름: VLM 은닉 + DINOv2 + 고유 감각(Eq. 3), flow-matching(Eq. 4) | RoboTwin: DINOv2 특징(`x_norm_patchtokens`)을 주 시점과 손목 시점에 모두 씀(`:796`). 흐름 손실은 `:725`, `:811` | 일치 |
| D16 | LIBERO 행동 헤드: "UniVLA의 자기회귀 헤드"(IV-A) | `finetune_libero.py:40–60`: MAP 풀링과 MLP(tanh) 출력, L1 회귀(`:133`). 자기회귀 디코딩이 아님. 흐름 손실도 아님 | 차이(설명과 구현 불일치) |
| D17 | LIBERO는 3인칭 영상만 씀(IV-A) | `finetune_libero.py:100`: `wrist_dino = True`로 **손목 DINOv2 특징을 항상 사용**. 주 시점 임베딩은 VLM 은닉 상태(`:105`), DINO가 아님. 주 시점 DINO 분기(`:113–120`)는 `else`에만 있음 | **차이(중요)**: 논문의 "DINO로 상태 인지"와 LIBERO 설정 설명이 코드와 맞지 않음. 손목 입력을 쓴다면 IV-A 설명과 다름 |
| D18 | 잠재 토큰 위치 선택 | `finetune_libero.py`의 `action_gt > 32000` 마스크: 어휘 ID가 32000보다 큰 토큰을 잠재 행동으로 씀 | 미확인(32000 오프셋의 근거 없음) |
| D19 | 자기회귀 대 flow 헤드(RoboTwin) | `finetune_robotwin_flow.py:654–725`: HRDT 계열 FlowBlock 기반 헤드 | 일치 |
| D20 | 실험 설정 값: LIBERO 30k/40k 스텝, 배치 128(IV-A) | `train.sh`와 finetune 설정: 배치 8, 최대 40000 스텝(`finetune_libero.py:154–155`). 배치 128과 불일치하는 설정 확인 필요 | 차이(미확인) |
| D21 | 코드북 크기 "32"(설정) | `finetune_robotwin_flow.py:844`에 `lam_num_latents=32`가 있으나 `lam.py:70, 75`는 16을 고정. 인자는 무시됨 | 버그(인자 무시, 실행은 됨) |
| D22 | 학습 명령 | `latent_action_model/train.sh`는 `lam-stage-2-mix.yaml`을 부르지만 저장소에 없음(설정은 egodex와 bridgeall 두 개). 경로는 `MFVLA/…` 고정 | 버그 |
| D23 | 테스트 시 코드 사용률 그림 | `model.py:448–455`: `self.lam.vq`를 참조하지만 모듈 이름은 `vq_action`, `vq_bg`. 테스트 종료 때 AttributeError | 버그 |
| D24 | (실행 장치) | `model.py:363, 368`: `.cuda()` 하드코딩. `flash-attn==2.5.5`(README). `torch.hub`로 DINOv2 다운로드 | 제약(CUDA·네트워크) |
| D25 | 입력 마스크 채널 | 주석은 마스크가 2채널(`lam.py:131`)이라 하지만 로더는 1채널(`datasets.py:310–313`). 실제 형상은 (B,T,1,H,W)이고 동작에는 문제 없음 | 주석 오류 |
| D26 | 공개 상태 | README 제목 "Motion-Focused Vision-Language-Action Model via Hybrid Disentangled Action Tokenization"은 arXiv 제목과 다름. README는 데이터 전처리를 "released upon finalization"이라 함 | 누락·제목 불일치 |
| D27 | 평가 시행 수(IV-A, 과제당 20회) | `README`: `--num_trials_per_task 20 --seed 7`(`run_libero_eval.py`) | 일치 |

**죽은 코드 참고:** `model.py:325–349`의 행동 손실 분기는 `pred_action`이 없어서 실행되지 않습니다. `lam.py`의 `shallow` 분기(`dim_half`)도 입력이 768차원뿐이라 빈 텐서입니다. 두 경우 모두 결과에 영향은 없습니다.

## 3. 주장 검증 (C 항목)

논문 주장을 원문 기준으로 대조했습니다. 판정: **지지**, **부분 지지**, **미지지**, **검증 불가**.

| ID | 주장(위치) | 검증 결과 | 판정 |
|---|---|---|---|
| C1 | 초록: "pre-trained exclusively on unlabeled human videos" | IV-A(LIBERO, 헤드라인 평균 91.8)는 BridgeV2 **로봇** 영상으로 사전학습. 사람 영상은 IV-B(RoboTwin)와 실물에서만 씀 | 부분 지지(헤드라인과 불일치) |
| C2 | 초록: "requiring only ∼50 trajectories" | 과제당 50개. LIBERO는 10과제(군당 500개), RoboTwin 10과제(500개) | 부분 지지(과제당 수치) |
| C3 | Table I: 평균이 모든 기준선보다 높다 | 91.8 > 90.1(villa-x): 합산 Fisher p = 0.30. UniVLA(저자 재현)보다는 p = 0.020, 보정 0.039 | 부분 지지(villa-x 대비 유의하지 않음) |
| C4 | IV-A: Goal +2.0, Long +9.5(villa-x 대비) | 차이 값은 정확(93.5−91.5, 84.0−74.5). Long만 p = 0.026(과제군 내), Goal p = 0.57 | 지지(값), Long만 유의 |
| C5 | Table I 기준선 UniVLA "strictly reproduced" | 원 논문 Bridge 평균 92.5 대 재현 88.1 | 지지되지 않음(재현값이 4.4점 낮음) |
| C6 | Table II: "comparable to state-of-the-art" | 67.7 대 π0 65.2(p = 0.26), UniVLA 63.6(p = 0.06, 보정 0.12) | 지지(comparable), "앞선다"는 미지지 |
| C7 | 의도-인지 분리가 성능을 올린다(IV-A, IV-B) | w/o DINO 대비 LIBERO +6.4(p ≈ 8×10⁻⁵), RoboTwin +4.9(p = 0.024, 보정 0.073). 단일 시드로 보임 | 부분 지지(LIBERO 강함, RoboTwin 약함) |
| C8 | 동결 VLM도 RDT와 비슷하니 VLM 사전학습이 효과적(IV-B) | 대조(사전학습 없는 같은 구조)가 없음. RDT는 다른 아키텍처 | 미지지(근거 부족) |
| C9 | 실물에서 UniVLA를 앞선다(Fig. 3c) | 46/60 대 23/60, p = 4×10⁻⁵(시행 수 20회 가정). 과제 3개 중 2개만 p < 0.05(보정 0.029) | 지지(소표본, 시드 미보고) |
| C10 | CKA 0.9139 대 0.8659에서 "significantly higher" | 부트스트랩 표준편차는 시드 변동이 아님. 데이터 쌍 1개(BridgeV2 대 FurnitureBench), 사람 영상 없음 | 미지지(유의성 주장) |
| C11 | Fig. 5: 사람과 로봇이 같은 토큰을 받는다 | 세 예시의 정성적 나열. 정량 지표 없음 | 검증 불가(예시 3개) |
| C12 | 행동 코드는 배경과 카메라 이동을 분리한다(I, III-B) | 전체 재구성 경로가 마스크 없이 학습됨(D4). 분리 여부를 재는 probe가 없음. toy에서 전경 마스크 안에도 카메라 정보가 남음(03 §4.2 A) | 미검증(누수 가능성) |
| C13 | 코드북 사용이 건강하다(암묵) | 사용률·혼잡도를 보고하지 않음. 행동 코드북은 재시작하지 않음(D7) | 검증 불가 |
| C14 | 시드 반복, 신뢰구간 | 어떤 표에도 없음 | 미지지(보고 부재) |

## 4. 구현 가능성 (판정)

**결론: 공식 재현은 이 맥에서 불가능하고, 아이디어 재구현은 중간 난이도입니다.**

**(a) 공식 재현**
- 코드는 공개되어 있지만 실행 경로가 CUDA를 전제합니다(`.cuda()` 하드코딩, flash-attn 2.5.5, `torch.hub` 다운로드). 이 맥(M4 Pro, 통합 메모리 약 24 GiB, MPS, CUDA 없음, **torch 미설치**)에서는 실행 자체가 막힙니다.
- 자원: LAM 학습은 GPU 2장(설정 `devices: 2`, 배치 64/장치, 2 epoch, 16-mixed). VLM 사전학습과 LIBERO 미세조정은 GPU 4장(`train.sh`), RoboTwin은 2장. GPU 모델은 기재되지 않았습니다.
- 데이터: EgoDex 영상 + SAM2 마스크(**마스크 미공개**), BridgeV2 OXE + RoboEngine 마스크(**미공개**), LIBERO RLDS(HF), RoboTwin 2.0 생성 데이터, 실물 데이터(미공개). 체크포인트: Prismatic-7B(HF)만 공개, 잠재 행동 체크포인트와 미세조정 체크포인트는 공개되지 않음.
- 판정: **이 맥 불가**, 공식 재현은 CUDA 서버에서 4 GPU 수준 + 마스크 생성 파이프라인이 필요합니다.

**(b) 아이디어 재구현**
- 핵심(이중 코드북 VQ-VAE + 마스크 손실 + 토큰을 VLM에 넣기 + 의도-인지 분리)은 코드에서 수백 줄로 다시 쓸 수 있습니다. 재구현 장벽은 (1) 사람 손 마스크 생성 파이프라인(SAM2 실행), (2) 두 단계 사전학습 자원, (3) 자르기-마스크 정렬 같은 데이터 세부입니다.
- 이 맥에서는 DINOv2-B 특징 추출도 느리고, 7B VLM의 LoRA 학습은 24 GiB에 맞지 않습니다. 작은 실험은 가능합니다(toy 수준, 소형 인코더 대체).
- 판정: **중간 난이도**. 설계 요소를 떼어 검증하는 소형 실험(§5 R1, R2)은 맥에서 가능하고, 대규모 학습은 GPU가 필요합니다.

## 5. 빠진 정보 목록 (재현에 필요한데 논문·코드에 없는 것)

1. 사전학습 데이터 규모(EgoDex 쌍 수), 스텝, 배치, 학습률(LAM, VLM).
2. 손·로봇 마스크 생성 절차와 임계값(D1, D2, D3).
3. λ_recon, λ_vq, λ_commit, λ_intent 값(D5, D13).
4. 쌍 간격과 영상 fps(D8).
5. 증강이 프레임과 마스크에 어떻게 적용되는지(D9).
6. 사용률, 혼잡도, 코드 재시작 방식(D7, C13).
7. 시드 수, 표준편차, 평가 시행 수(RoboTwin, 실물).
8. LIBERO 헤드 구조와 시점 구성(D16, D17).
9. 기준선 재현 설정(UniVLA-Bridge, 기준선 값의 출처별 프로토콜).

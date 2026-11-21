# 04. 공식 코드 대조와 구현 가능성

## 0. 대상과 방법

- **대상:** `ByteDance-Seed/VideoWorld` 저장소의 `videoworld2/` 하위 폴더, 커밋 `3dc3e563e201cebab0e947177e79117c885ec3ba` (2026-02-25, "Update README.md"). 얕은 복제는 `repo/`에 있고 gitignore되어 있습니다.
- **방법:** 정적 읽기만 했습니다. 설치, import, 실행, 체크포인트 다운로드, API 호출을 하지 않았습니다. 저장소의 내용은 신뢰하지 않는 데이터로 취급했습니다.
- **읽은 파일:** `videoworld2/README.md`, `scripts/train_dldm*.sh`, `scripts/test.sh`, `videoworld2/configs/base/experiment/openx.py`, `videoworld2/configs/base/config_ldm.py`, `videoworld2/models/video2world_model.py`(dLDM 학습 단계, 손실 계산), `videoworld2/latent_dynamics/`(인코더·Q-Former), `videoworld2/models/ctrl_video2world_dit.py`(제어 분기), `install.sh`, `requirements.txt`, 두 라이선스 파일.
- **검색으로 확인한 것:** AR 학습 코드, CALVIN, 분류기, 기준선 관련 문자열.

## 1. 설치 요구사항과 라이선스

- **라이선스:** 루트 `LICENSE`는 Apache License 2.0입니다(GitHub API도 Apache-2.0). 소스 파일 머리말은 "Copyright (c) 2025 ByteDance Ltd. and/or its affiliates", Apache-2.0입니다. `videoworld2/ATTRIBUTIONS.md`는 Cosmos 계열의 오픈소스 구성요소 목록을 담고 있습니다.
- **기반 모델 라이선스:** Cosmos-Predict2-2B-Video2World는 Hugging Face 메타데이터에서 `other`이며 자동 게이트입니다(NVIDIA 라이선스). dLDM 체크포인트 HF 메타데이터는 apache-2.0입니다. 두 라이선스를 함께 지켜야 합니다.
- **설치 요구:** CUDA 12.4용 PyTorch 2.6.0 휠, `transformer-engine[pytorch]==1.12.0`, NVIDIA apex(CUDA 확장 빌드), `flash-attn==2.6.3`, `megatron-core==0.10.0`, `decord`, `lpips` 등입니다. `install.sh`는 개발 환경의 절대 경로를 가정합니다. 이 맥(CUDA 없음)에서는 설치 자체가 막힙니다.
- **설치하지 않았습니다.** 승인 없는 설치는 하지 않는 원칙입니다.

## 2. 논문 ↔ 코드 대조 (D 항목)

| 번호 | 논문 | 공식 코드 | 판정 |
|---|---|---|---|
| D1 | 표 4: dLDM 기본 학습률 1e-4 | `configs/base/experiment/openx.py`의 세 실험 모두 `optimizer.lr = 2 ** (-14.5)` ≈ 4.3e-5 | **불일치** |
| D2 | 표 4: dLDM 최대 반복 1e5 | 워밍업 실험 `max_iter=100000`, 본 학습 실험 `max_iter=30000`(`train_dldm.sh`가 쓰는 실험). 두 단계를 합친 총 반복은 1.3e5 | 불명확. 표 4는 한 단계만 말함 |
| D3 | §5.5: 2단계부터 VQ-VAE 디코더 재구성 손실을 버림. 알고리즘 1: `MSE(rec, video) + VDM(...)` | `models/video2world_model.py`의 dLDM 학습 분기: 디코더 입력 코드는 `quant_codes.detach()`로 끊지만, `ldm_net.loss(..., ori_recon_image)`의 `ldm_loss_value`를 `kendall_loss`에 더함. 재구성 경로로 인코더는 기울기를 받음 | **본문과 코드 불일치.** 알고리즘 1과 코드는 일치. 기울기 차단(§3.2)은 코드에서 확인됨 |
| D4 | §3.2: 움직임 힌트를 기울기 차단 ControlNet 유사 분기로 넣음 | `gray_video.detach()`를 힌트로 씀. 제어 가중치 `control_weight=0.5`, `layer_mask`는 28개 블록 중 인덱스 3 이상 | 구조 일치. 가중치 0.5와 블록 범위는 논문에 없음 |
| D5 | (논문에 없음) | `color_transform`: VDM 입력 영상에 색 증강(`self.color_aug`)을 적용 | **논문에 없는 학습 증강** |
| D6 | §3.2: VDM은 잠재 코드로 외형을 채움 | 학습 중 `t5_text_embeddings`를 0으로 고정. 잠재 코드는 `quant_to_dit_dim`으로 교차 주의 입력 512 토큰에 넣고 나머지는 0 패딩 | 방향은 일치. 텍스트를 쓰지 않는 것은 논문에 명시되지 않음 |
| D7 | 표 4: "L2 & Denoising loss" | 코드 변수명 `kendall_loss`는 실제로 EDM 가중 MSE(`edm_loss = pred_mse * weights`)이며, Kendall 불확실성 가중치 학습은 없음 | 이름만 다름. 논문 설명과 맞지 않는 이름 |
| D8 | 표 4: 배치 128 (dLDM), 256 (AR) | 데이터 로더 `batch_size=1`, 학습 스크립트 `--nproc_per_node=8`. 기울기 누적 설정은 찾지 못함 | **확인 불가.** 전역 배치를 128로 맞출 근거가 없음 |
| D9 | §3.2, §5.1, 표 4: AR Transformer(Cosmos AR 4B) 다음 토큰 예측 | 저장소에 AR 학습·추론 코드, AR 설정이 없음. README는 `latent_code_infos.pt`를 AR 학습용으로 제공한다고 씀 | **공개 범위 불일치.** 본 논문의 AR 단계는 코드로 재현할 수 없음 |
| D10 | §4.2, 부록 B: DINOv2 단계 분류기, 시험 정확도 96.1% | 분류기 학습·평가 코드 없음 | **재현 불가** (코드와 라벨 모두 없음) |
| D11 | §5.4: CALVIN 평가와 액션 헤드 미세조정, 기준선(LAPA, Moto, AdaWorld, iVideoGPT, CoLA) | CALVIN 평가 스크립트와 기준선 구현 없음 (검색 기준) | **공개 범위 불일치** |
| D12 | §3.2: 잠재 코드 인코더 인과 구조 | 인과 3D 합성곱 인코더(`CausalConv3d`, `CausalDiscreteVideoLatentDynamicTokenizer`)를 확인 | 확인됨 |
| D13 | §3.2: VDM의 교차 주의를 인과적으로 함 | 인과 마스크가 교차 주의 경로에 있는지 찾지 못함 (인과 3D 합성곱만 확인). 이 항목은 확정하지 않음 | **미확정** |
| D14 | §3.1, 표 4: 양자화 FSQ 레벨 [8,5,5,5] (1000), 임베딩 4, 질의 N=4, 시간 스트라이드 4, 93프레임 | `levels=[8,5,5,5]`, `embedding_dim=4`, `act_embedding_num=4`, `temporal_compression=4`, `num_frames=93` | **일치** |
| D15 | §5.1: Cosmos DiT 2B를 VDM으로 사용 | `predict2_video2world_ctrl_ldm_fsdp_2b_480p_16fps`, 기반 체크포인트 `Cosmos-Predict2-2B-Video2World` | 일치 |
| D16 | §5.1: 트레이너블 파라미터는 AR, dLDM, DiT, 투영 층 | 워밍업 단계는 `train_architecture='lora'`, 본 학습은 `'base'`에 `finetune_base_model=True` | 부분 일치. 워밍업에서 VDM을 LoRA로 두는 설정은 논문 설명에 없음 |
| D17 | 표 3f: 동결 31.7, LoRA 50.9, 전체 미세조정 68.8 | 코드는 LoRA와 전체 미세조정 경로를 모두 가짐. 절제 설정 파일은 없음 | 구현 가능. 수치 재현은 불가 |
| D18 | 초록: "all code, data, and models to be open-sourced" | 공개 코드와 HF 체크포인트 (dLDM), HF 데이터 (Video-CraftBench, OpenX 메타) 공개. 기반 Cosmos 가중치는 NVIDIA 쪽. AR 4B와 평가 코드는 없음 | **부분 충족** |
| D19 | (저장소 위생) | `install.sh`와 `git_push.sh`에 개발 환경의 절대 경로와 내부 브랜치 이름이 남아 있음. README의 클론 주소와 논문 링크(2501.09781)가 본 논문과 다름 | 재현에 직접 영향은 작음. 문서 오류 |

### 2.1 논문 주장의 코드 쪽 확인

- **외형 분리(핵심 주장):** 코드에서 dLDM 손실은 재구성 손실 + 확산 손실이고, 확산 쪽은 VDM이 잠재 코드를 교차 주의로 받아 외형을 채우는 구조입니다(D6, D12, D13). 구조는 논문과 맞습니다. 다만 기울기 차단(D3)과 학습 증강(D5)이 추가로 있어, 외형 분리의 효과가 논문의 절제만큼 깨끗하게 분리되었는지는 확정할 수 없습니다.
- **학습 기본값 차이:** 공식 코드의 학습률(D1), 색 증강(D5), 제어 가중치(D4)는 논문에 없습니다. 논문 수치를 재현하려면 이 설정을 알아야 합니다.

## 3. 안정성과 한계

이 논문에는 부록 수준의 안정성 증명이나 한계 절이 없습니다. 결론(§7)은 "continued scaling은 future work"라고만 씁니다. 따라서 이 절은 비어 있습니다.

## 4. 구현 가능성 (이 맥)

**판정: 이 맥에서 논문 수치를 재현하는 것은 불가능합니다. 구조 확인과 공식 체크포인트 추론 일부만 조건부로 가능합니다.**

- **이 맥의 자원:** Apple M4 Pro, 약 24 GiB 통합 메모리, MPS 사용 가능, CUDA 없음.
- **공식 저장소 학습 (dLDM 2단계 + AR 4B): 불가.** CUDA 전용 패키지, 8-GPU FSDP 설정, 기반 DiT 2B 미세조정이 필요합니다. AR 4B 코드가 없습니다(D9).
- **공식 체크포인트 추론 (VideoCraft 데모): 조건부.** dLDM 체크포인트(HF, Apache-2.0 메타데이터)와 Cosmos 2B 가중치(NVIDIA 게이트)를 받고, CUDA 전용 의존성(flash-attn, transformer-engine, apex)을 MPS용으로 바꿔야 합니다. 단일 GPU 추론 설정이 있지만 MPS에서의 속도와 메모리는 측정하지 않았습니다.
- **단계 분류기와 Video-CraftBench 평가: 불가.** 분류기 코드가 없습니다(D10).
- **CALVIN 평가: 불가.** 평가 코드, 액션 헤드, CALVIN 시뮬레이터 설정이 저장소에 없습니다(D11).
- **Video-CraftBench 데이터 다운로드:** HF 저장소는 파일 3개이지만 용량은 메타데이터에서 확인하지 않았습니다. 약 7시간 분량의 영상이므로 큰 용량으로 보고 승인 없이 받지 않았습니다.
- **노력 추정 (미측정):** 추론 경로만 MPS 포팅하는 데 수일, 학습 재현은 GPU 클러스터와 데이터 준비가 필요해 이 맥 범위 밖입니다.

## 5. 재실행

정적 대조는 저장소 파일을 직접 읽는 것으로 끝납니다. 이 파일의 D 항목은 `repo/videoworld2/` 아래 경로와 커밋 해시로 다시 확인할 수 있습니다. 실행 스크립트는 없습니다.

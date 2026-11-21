# 04. 공식 코드 대조 (논문 ↔ 저장소), 권리, 구현 가능성

대상: `repo/` (github.com/spmem/spmem, 커밋 `89e1f70e`, 2026-05-11). **정적으로만 읽었습니다.** 실행, import, 설치, 가중치와 데이터 다운로드는 하지 않았습니다. 저장소 내용은 신뢰하지 않는 외부 자료로 취급했습니다.

## 1. 권리와 라이선스

### 1.1 논문 (arXiv 2506.05284)

- **abs 페이지 (2026-11-21 직접 조회):** 페이지 하단 라이선스 링크의 글은 **"view license"**이고, 링크 주소는 `http://arxiv.org/licenses/nonexclusive-distrib/1.0/`입니다. 링크 옆 문구는 "Rights to this article"입니다. 즉 arXiv의 **비독점 배포 라이선스(arXiv non-exclusive distribution license 1.0)**이고, **CC BY가 아닙니다.** 페이지 어디에도 "Creative Commons" 링크는 없습니다.
- **판단:** 논문 그림과 원문을 게시본에 재배포하려면 저자의 별도 허락이 필요할 수 있습니다. 그래서 **논문 그림은 개인용 리포트와 PPT에만 넣고, 공개본과 블로그에는 넣지 않았습니다.** 공개본 그림은 모두 우리가 그린 그래프입니다.
- **저작권 표시 점검 (PDF 텍스트 전체, `pdftotext`):** `©`, "Copyright", "All rights reserved", "Inc.", "Corporation", "confidential", "proprietary" 같은 표현은 없었습니다. 본문의 "We thank Google for their support"(§6, 감사)는 연구비 감사 문장이지 저작권 표시가 아닙니다. 저자 소속(Stanford, SJTU, CUHK, Shanghai AI Lab, NTU)은 기관 표기이고 기업 저작권 고지가 아닙니다. 그림 속 로고와 사진은 텍스트 검사에서 잡히지 않아서 이미지 자체는 검사하지 않았습니다 (`pdfimages -list`로 196개 이미지 객체가 있음을 확인).
- **NeurIPS proceedings 페이지:** 라이선스나 저작권 줄은 없습니다 (이름 변경 정책 문구만 있음).

### 1.2 공식 코드 (spmem/spmem)

- `LICENSE.txt`: **Apache License 2.0** (표준 본문, 저작권 줄은 저장소 쪽에 없음). GitHub API도 `apache-2.0`으로 보고합니다.
- 벤더링된 `Depth-Anything-3/`: LICENSE가 Apache 2.0입니다. 원저작자는 ByteDance Seed 쪽입니다.
- `Depth-Anything-3/da3_streaming/loop_utils/salad/`: LICENSE 파일이 있습니다 (종류는 읽지 않음).
- **모델 가중치와 데이터의 라이선스는 확인하지 않았습니다.** 체크포인트 `ysmikey/spmem_ckpt`, 데이터 `ysmikey/spmem_megadata`, 기반 모델 CogVideoX-5B-I2V, Qwen2.5-VL-7B-Instruct는 각각 별도 라이선스가 있을 수 있습니다. 받기 전에 확인해야 합니다.

## 2. 논문과 코드의 차이 (D1–D12)

| 번호 | 항목 | 논문 | 저장소 (위치) | 판정 |
|---|---|---|---|---|
| D1 | 학습 설정 | 학습률 2e-5, 6,000 이터레이션, 미니배치 8, A100 8장 (§4) | `train_example.sh`: 학습률 **1e-5**, **8,000** 스텝, 스케줄러 `cosine_with_restarts`, AdamW (β 0.9, 0.95), 가중치 감쇠 0.001, 노이즈 이미지 드롭아웃 0.05, 그래디언트 클리핑 1.0, GPU당 배치 1 × 8 | **다름**. 예제 스크립트라고 적혀 있고 논문 설정과 같지 않음. 드롭아웃과 스케줄러는 논문에 없음 |
| D2 | 추론 중 공간 기억의 복원기 | CUT3R로 온라인 복원 (부록 A, 그림 S1) | `infer_stream.py`: **Depth-Anything-3** (`DA3NESTED-GIANT-LARGE`, 처리 해상도 504)와 TSDF (`tsdf/demo_da3.py`). **CUT3R 호출 없음** (저장소 전체에서 `cut3r` 문자열 0건) | **다름 (중요)**. 논문 부록 A의 추론 방식과 다른 복원기를 씀 |
| D3 | 데이터 구축 복원기 | MegaSaM으로 카메라와 깊이 (§3.4) | 저장소에 MegaSaM 코드 없음. `tsdf/run_data.py`는 `${name}.npz`를 입력으로 받고 그 생성 방법은 적혀 있지 않음. `--iqr_filter`는 논문에 없는 옵션 | **확인 불가 (생성 단계 누락)** |
| D4 | 에피소드 기억 선택 기준 | 생성 중 **새로 드러난 미지 영역의 넓이**가 임계값을 넘으면 프레임 저장 (§3.2) | `infer_stream.py` `select_keyframe_indices`: **카메라 시야 뿔(frustum) 겹침**에 따른 탐욕적 선택 (기본 N=10, 샘플 32, 겹침 임계 0.3, 실패 시 마지막 N장). `--naive_ref`는 마지막 N장. 가시성 마스크 계산은 없음 | **다름 (중요)**. 기준이 자세 기반이고 논문은 영상 기반 (미지 영역) |
| D5 | 추론 중 텍스트 프롬프트 | Qwen은 데이터 주석에만 씀 (§3.4), 추론 설명에는 없음 | `infer.py`와 `infer_stream.py`가 Qwen2.5-VL-7B로 프롬프트를 만듦. **오프라인 `infer.py`는 정답 target 영상 `Vid_target.mp4`에서 프롬프트를 만듦** | **논문에 없는 단계**. 오프라인 경로로 평가했다면 정답 정보가 새 들어갈 수 있음 (평가 방식은 확인 불가) |
| D6 | 조건 DiT 블록 수 | 앞 18개 블록 복사 (§3.3) | `--num_tracking_blocks 18` | **일치** |
| D7 | 해상도, 프레임, 작업 기억 | 480×720, 49프레임, 최근 5개 과거 프레임 (§4) | `--height_buckets 480 --width_buckets 720 --frame_buckets 49 --cond_frames 5` | **일치** |
| D8 | 역사 교차 주의 간격 | 논문에 간격 값이 없음 | `--cross_attn_interval 14` | **논문에 없는 설정** (값은 저장소에만 있음) |
| D9 | 데이터셋 크기 | 90K 샘플 (§3.4) | 저장소 README: "~80K video clips" | **불일치** (`03` K4) |
| D10 | 평가 코드 | Table 1–3 (뷰 리콜, VBench, 사용자 연구) | **없음**. PSNR, SSIM, LPIPS, VBench 계산과 사용자 연구 스크립트가 없음. 기준선 (TrajectoryCrafter, DaS, Wan2.1-Inpainting) 코드도 없음 | **표 1–3 재현 불가** (이 저장소만으로는) |
| D11 | 설정 파일 잔재 | 없음 | `config/base_cfg.py`, `config/ssm_cfg.py`는 CoTracker 계열 설정 (`model_name="cotracker"`). 어느 스크립트도 import하지 않음 | 죽은 코드 (무해하지만 혼동 요인) |
| D12 | 장치 하드코딩 | 논문에 없음 | `infer.py`, `infer_stream.py`: `device="cuda"`, `torch.bfloat16`. 학습 스크립트는 CUDA 전제 (`torch.cuda.set_device`). `tsdf/fusion.py`는 기본 `use_gpu=True`이고 PyCUDA를 씀. `training/cogvideox_image_to_video_sft_ref.py`는 MPS에서 bf16 학습을 막는 주석이 있음 | **CUDA 전제**. 맥 포팅에 직접 영향 (§4) |

참고: 논문의 "CUT3R로 온라인 복원" (부록 A)과 저장소의 DA3 사용이 가장 큰 차이입니다. 논문 본문만 읽으면 이를 알 수 없습니다. 이 차이는 Table 1–2 결과가 어느 복원기로 만든 것인지에 따라 달라질 수 있습니다.

## 3. 저장소에서 확인된 것

- 학습 구조는 논문과 맞습니다. 조건 DiT 18블록 (`num_tracking_blocks`), 정적 점군 렌더링 조건, 5프레임 문맥, 49프레임, 480×720은 일치합니다 (D6, D7).
- 추론 흐름은 논문 그림 2와 맞습니다. 정적 지도 누적 → 목표 궤적 렌더링 → 조건 생성 → 프레임 추가 → 에피소드 프레임 재선택 순서입니다 (`infer_stream.py` 약 1000–1030행).
- 정적 지도는 TSDF로 합칩니다 (`tsdf/fusion.py`, 복셀 최대 1200, 논문 부록 A와 같은 값).
- 예제 입력 4개 (`examples/`)가 있습니다. 영상은 `Vid_source`, `Vid_target`, `Vid_mask`, `Vid_masktarget`, `Vid`로 구성됩니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)

**판정: 학습은 불가, 추론은 포팅 후 조건부, 평가 재현은 자체 구현이 필요합니다.**

| 작업 | 판정 | 근거 | 노력 추정 |
|---|---|---|---|
| 공식 학습 (8 × A100, 6,000 이터레이션) | **불가** | CUDA 전제 (D12). 5B 확산 트랜스포머를 49프레임으로 학습하는 메모리와 시간이 맥으로 불가능 | 해당 없음 |
| 공식 추론 (`infer.py`, 오프라인) | **조건부** | CUDA 하드코딩 (`device="cuda"`, bf16). 맥에서 fp16 또는 fp32, MPS로 바꿔야 함. 가중치 약 10 GB (CogVideoX-5B 계열 bf16 추정) + Qwen2.5-VL-7B (약 15 GB bf16 추정, `infer.py`는 프롬프트 생성 뒤 삭제). 한 번에 올리면 24 GiB를 넘을 수 있음. 속도는 측정 전 | 포팅 2–4일 (추정) + 메모리와 시간 측정 |
| 스트리밍 추론 (`infer_stream.py`) | **조건부 (더 어려움)** | DA3 GIANT 대형 모델 추가, pytorch3d와 open3d 설치 (맥 빌드 가능성 미확인), `tsdf/fusion.py`에 `use_gpu` 옵션이 있으나 `demo_da3` 경로에서 끌 수 있는지는 확인하지 않음 | 포팅 3–5일 (추정) |
| 데이터 구축 (MegaSaM + TSDF + 렌더링) | **불가 또는 조건부** | MegaSaM 코드 없음 (D3). `run_data.py`는 pytorch3d와 PyCUDA를 씀. 데이터 MiraData 원본이 필요 | 미정 |
| 논문 표 1–3 재현 | **불가 (코드만으로)** | 평가 코드와 기준선 코드가 없음 (D10) | 자체 구현 2–3일 (추정) + 기준선 구현 불가 |
| 공간 기억 제거 실험 (논문에 없음) | **조건부** | 저장소의 추론 경로에서 공간 기억 조건을 끌 수 있는지 아직 확인하지 않음 | 확인 필요 |

**재현 가능성 요약.** 이 맥에서 할 수 있는 최대는 **공식 추론 코드를 MPS로 옮겨 예제 한 개를 생성해 보는 것**입니다. 이것도 메모리와 시간이 맞는지 측정해 봐야 합니다. 논문 수치를 다시 얻는 일은 평가 코드가 없어 불가능합니다. 실행과 설치는 사용자 승인 뒤에 해야 합니다 (`CLAUDE.md`).

## 5. 설치와 실행 전 확인 목록 (실행은 하지 않음)

- 가중치 세 종류의 라이선스 (§1.2).
- `requirements.txt`는 `torch==2.7.1+cu126`, `flash_attn==2.7.4.post1`, `xformers==0.0.31.post1`, `deepspeed`, `bitsandbytes`를 고정합니다. 맥에서는 대부분 쓸 수 없습니다. 추론에 필요한 부분만 골라야 합니다.
- `pytorch3d`는 GitHub에서 설치하라고 적혀 있습니다 (README). 맥 빌드는 확인하지 않았습니다.
- 다운로드 용량: 체크포인트 (1 GB 이상으로 보임)는 크기를 보고하고 승인을 받아야 합니다.

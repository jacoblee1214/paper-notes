# 04. 공식 코드 대조와 구현 가능성

대상은 `SpatialVLA/SpatialVLA` 저장소의 commit `18fd74b2a633`(2025-06-23)입니다. `repo/`에 얕게 복제했고, **정적으로만 읽었습니다.** 실행, import, 설치, 체크포인트 다운로드는 하지 않았습니다. 복제본 안의 내용은 신뢰하지 않는 데이터로 취급했습니다.

## 1. 라이선스와 저장소 상태

- GitHub API `license: null`. LICENSE 파일이 없습니다.
- `pyproject.toml`의 classifier는 "Apache Software License"라고 씁니다. 그러나 라이선스 파일이 없으므로 **법적으로 확정할 수 없습니다.**
- 체크포인트 `IPEC-COMMUNITY/spatialvla-4b-224-pt`의 HF 태그는 MIT입니다. 코드와 가중치의 라이선스가 다릅니다.
- 마지막 push 2025-06-23. 저장소 공지는 2025-03-16 "단순화"가 마지막 변경입니다. lerobot 브랜치는 "개발 중"이라고 적혀 있습니다.
- 결론: 코드를 내 저장소에 복사하지 않습니다. 논문 구현 참고는 읽기로만 합니다.

## 2. 논문 대 코드 대조 (D1–D22)

판정 기준. **일치**: 논문과 코드가 같습니다. **차이**: 논문이 다르게 쓰거나 값이 다릅니다. **미기재**: 논문에 값이 없습니다. **구현 누락**: 논문에 있는데 저장소에 없습니다.

| 번호 | 항목 | 논문 | 코드 (파일:위치) | 판정 |
|---|---|---|---|---|
| D1 | 깊이 소스 | ZoeDepth 추정 깊이 (§III-A) | `ZoeDepthForDepthEstimation` 사용, `torch.no_grad()`로 평가 (`model/modeling_spatialvla.py`, 약 313–318). 공개 설정은 `Intel/zoedepth-nyu-kitti` | 일치 |
| D2 | 카메라 내부 파라미터 | "로봇-카메라 외부 보정 불필요" (§III-A) | `scripts/intrinsics.json`에 bridge(640×480, fx 623.588) 하나와 `default`(같은 값). 목록에 없는 데이터셋은 `default`를 씀. 224 리사이즈에 맞춰 K를 축척하는 코드는 있음 (`processing_spatialvla.py` 91–95) | 차이 (주장과 다름: 새 카메라는 기본 K 가정이 필요) |
| D3 | 패치 내 샘플 점 수 | 미기재 (식 2의 "패치 내 점") | 설정 기본값 `ego3d_patch_reso=4` (`configuration_spatialvla.py` 39), 공개 `config.json`은 **2**. 역투영 docstring 예시도 2 | 미기재 (기본값과 공개값이 다름, 재현 시 2 명시 필요) |
| D4 | 주파수 인코딩과 MLP | "sinusoidal + learnable MLP" (§III-A) | `n_freqs=8`, 로그 스케일, 중심 (0, 0, 2), xyz를 2로 나눠 정규화. 2층 MLP (Linear–LayerNorm–ReLU–Linear), 가중치 Xavier gain 0.01 (`modeling_spatialvla.py` 41–98) | 미기재 (세부는 코드에만 있음) |
| D5 | 위치 임베딩 덧셈 | 식 (2): O³ᵈ = X + MLP(γ(P)) | `selected = last_hidden_state + pos_embed_3d` (약 327) | 일치 |
| D6 | 프로젝터 정규화 | 논문에 없음 | `image_features / sqrt(hidden_size)` (약 332) | 미기재 (논문에 없는 스케일링) |
| D7 | 행동 차원과 토큰 수 | V = M_trans + M_rot + 2 = 8194 (식 5 설명) | `num_bins`: 이동 θ16·φ32·r8 = 4096, 회전 16³ = 4096, 그리퍼 2. 합 8194 (`scripts/action_config.json`) | 일치 |
| D8 | 등확률 경계 | 식 (4): 잘린 가우시안을 등확률 분할 | `get_bin_policy`가 같은 방식 (`action_tokenizer.py` 343–370). `min_sigma` 기본 0 | 일치 |
| D9 | 사전학습 통계 | "전체 사전학습 혼합에서 가우시안 맞춤" (§III-A) | `scripts/gs_spatialvla_plus.json` (`--normalized_statistic_path`) | 일치 |
| D10 | 극좌표 정의 | (φ, θ, r) (§III-A) | θ = arctan2(√(x²+y²), z), φ = arctan2(y, x), r = norm. 범위 θ∈[0,π], φ∈[−π,π], r∈[0,√3] (`action_tokenizer.py` 93–102, 249–261) | 일치 |
| D11 | 회전과 그리퍼 | roll, pitch, yaw 등확률 격자, 그리퍼 토큰 2개 (§III-A) | 회전 범위 [−1,1]에서 등확률, 그리퍼는 0.5 임계 (`RotationTokenizer`, `GripperTokenzier`) | 일치 |
| D12 | 청크 길이 | T = 4, 12토큰 (부록 D) | 공개 `processor_config.json`은 `action_chunk_size: 4`. 코드 기본값은 **1** (`processing_spatialvla.py` 52) | 일치 (공개 설정 기준). 기본값 차이 |
| D13 | 디코딩 기본값 버그 | 논문에 없음 | `unnorm_key`가 None일 때 `next(self.statistics.keys())` (`processing_spatialvla.py` 238). `dict_keys`는 이터레이터가 아니어서 `TypeError`가 날 것으로 보임. README 예제는 키를 직접 주므로 영향 없음 | 구현 결함 (정적 확인, 실행하지 않음) |
| D14 | 텍스트 임베딩 동결 | "텍스트 임베딩은 사전학습 중 동결" (§III-B) | 사전학습 기본 `freeze_llm_embed=True` (`spatialvla_pretrain.py` 74). 새 행동 토큰 임베딩은 학습 (`use_spatial_token`) | 일치 |
| D15 | 사전학습 설정 | 64 A100, 배치 2048, 10일, lr 2e-5, 선형, weight decay 0, warmup 0.005 (부록 D) | `lr=2e-5`, `weight_decay 0.0`, `warmup_ratio 0.005`, `linear` 일치 (`torchrun_pretrain.sh` 75–78). 그러나 GPU 기본값 48 × 배치 32 = 1536. 2048은 GPUS=64를 줄 때만 맞음 | 일치 (GPU 수 인자 필요) |
| D16 | 데이터 혼합 비율 | 그림 9: Bridge 15.34%, Fractal 14.71%, DROID 11.66%, BC-Z 8.64%, Kuka 7.06%, RH20T 5.67%, Hydra 5.15%, Language Table 5.06% | `oxe_spatial_vla_plus`의 샘플링 가중치를 합으로 나눈 값: Bridge 3.6%, Fractal 1.9%, DROID 0.2%, BC-Z 0.7%, Kuka 1.4%, RH20T 0.05%, Hydra 7.2%, Language Table 0.4% (`data/oxe/mixtures.py` 148–177). 논문 비율이 에피소드 비율인지 샘플링 비율인지 미기재 | 차이 (정의 불명. 같은 정의라면 큰 불일치) |
| D17 | 두 단계 학습 | 1단계 160k, 2단계 40k에서 DROID 제거 (부록 D). 본문은 "최종 1/3" (§III-B) | `oxe_spatial_vla_plus_stage2`는 DROID 줄을 주석 처리 (`mixtures.py` 206). 단계 전환 스크립트와 스텝 수는 저장소에 없음 (`mixture` 환경변수로만 바꿈) | 일치 (stage2 혼합), 스텝 정보 미기재 |
| D18 | 사후학습의 격자 재이산화 | 새 가우시안 맞춤, 삼선형 보간으로 임베딩 초기화 (§III-B, 식 6) | `spatial_embedding_adaption`이 새 정책을 만들고 `scipy.interpolate.griddata(method='linear')`로 임베딩 보간 (`action_tokenizer.py` 390–430). 미세조정 스크립트가 호출 (`spatialvla_finetune.py` 232). 삼선형이 아니라 Delaunay 기반 선형 보간 | 차이 (보간 방식) |
| D19 | LIBERO 미세조정 | 200 epoch, LoRA r=32, α=32 (§IV-B) | `finetune_lora.sh` 기본 `epoch=50`, `lr=5e-4`, `lora=32`, `lora_alpha=32`, 대상 `linear` | 차이 (epoch 50 대 200) |
| D20 | 추론 설정 | GPU 8.5 GB, RTX 4090 약 20 Hz, ensemble 행동 (§IV, 부록 D) | README는 bf16, `.cuda()`, 8.5 GB. `predict_action`은 `max_new_tokens=256`, `do_sample=False`. **행동 앙상블 코드는 없음** | 구현 누락 (ensemble) |
| D21 | 실로봇과 평가 코드 | WidowX, Franka 배치 (부록 D, E) | 저장소에 배치 코드 없음. SimplerEnv만 외부 저장소 (`DelinQu/SimplerEnv-OpenVLA`) | 구현 누락 |
| D22 | 깊이 정책의 3D 특징 | "점군 또는 깊이" 언급 (§IV-C) | 저장소에 점군 입력 없음. 깊이는 추정만 (D1) | 미기재 |

## 3. 저장소 결함과 검증 항목 (요약)

1. **D13 기본값 버그.** 정적 확인입니다. 실행하지 않았습니다. README 사용법은 `unnorm_key`를 넘기므로 공개 예제에서는 문제가 없습니다.
2. **D3 기본값과 공개 설정이 다름.** `ego3d_patch_reso` 4와 2. 재현하면서 설정을 명시하지 않으면 다른 모델을 만듭니다.
3. **D2 카메라 가정.** 목록에 없는 카메라는 bridge의 K를 씁니다. 실제 로봇에 쓰려면 K를 직접 넣어야 합니다. 논문의 "보정 불필요"라는 설명과 다릅니다.
4. **D16 혼합 비율.** 그림 9와 코드 가중치가 같은 정의인지 확인되지 않았습니다. 같은 정의라면 DROID와 RH20T 비율이 크게 다릅니다. 데이터셋 크기를 알아야 판단할 수 있습니다.
5. **D18 보간 방식.** 논문의 삼선형 보간과 코드의 `griddata` 선형 보간은 격자 밖과 경계에서 다를 수 있습니다. 적응 결과가 달라지는지는 측정하지 않았습니다.

## 4. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)

**판정: 학습과 실로봇 재현은 불가능합니다. 추론 재현은 조건부입니다. 격자와 토크나이저 검사는 CPU로 완료했습니다.**

| 작업 | 판정 | 근거와 노력 추정 |
|---|---|---|
| 사전학습 (64 A100 × 10일) | **불가** | CUDA 전제, 규모가 맥 메모리와 시간에 맞지 않음 |
| 미세조정 (LoRA, 8 GPU 기본) | **불가** | `finetune_lora.sh`가 GPUS 8, CUDA 기준. 데이터 준비와 DeepSpeed ZeRO 필요 |
| 추론 (체크포인트 8.1 GB, bf16) | **조건부** | 메모리는 맞을 것으로 보이나 MPS에서 bf16·flash-attn(requirements에 CUDA 전용 `flash-attn==2.6.3`)과 속도는 측정 전. 체크포인트 다운로드와 설치 승인 필요. 추정 1–3일 |
| SimplerEnv 평가 | **불가 (맥 기준)** | 외부 시뮬 설치와 GPU 렌더링 가정. 추정 난이도 높음 |
| LIBERO 평가 | **조건부** | MuJoCo 설치와 추론 호출 수가 많음. 한 스위트 소표본만 가능 (추정) |
| WidowX, Franka 실로봇 | **불가** | 하드웨어 없음 |
| 행동 격자와 토크나이저 검사 (왕복, 등확률, 범위) | **가능, 완료 (부분)** | `gs_*.json`과 `SpatialActionTokenizer`의 수식을 numpy로 옮겨 확인. `analysis/toy_adaptive_grid.py` |
| Ego3D 역투영 기하 검사 | **가능 (CPU, 미수행)** | 깊이와 K에서 패치 점 좌표를 만드는 식은 numpy로 충분. 노력 반나절 (추정) |
| 카메라 K 오지정 민감도 | **가능 (CPU, 미수행)** | D2를 수치로 확인하는 기하 토이. 추론 없이 가능 |

노력 추정은 측정하지 않은 값입니다. 추론 1–3일, 기하 토이 반나절은 추정입니다.

## 5. 이 문서의 한계

- 저장소를 정적으로만 읽었습니다. 기본값이 실제 학습에서 쓰였는지는 확인하지 못했습니다.
- D16은 샘플링 가중치와 논문 비율의 정의를 확인하지 못해 확정할 수 없습니다.
- 체크포인트 가중치, 설정 파일의 나머지 값, 데이터셋 변환 코드는 읽지 않았습니다.

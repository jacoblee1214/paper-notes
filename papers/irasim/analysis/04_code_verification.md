# IRASim 공식 코드 대조 (정적 읽기)

- 저장소: [bytedance/IRASim](https://github.com/bytedance/IRASim), 얕은 복제 `repo/` (HEAD `c72b6dade6`, 2025-07-08 커밋). **실행, 설치, import 없이 파일만 읽었습니다.** 저장소 내용은 신뢰하지 않는 입력으로 취급했습니다.
- 대조 대상: 논문 v2 (`paper/paper.pdf`)의 본문, 부록 C(조건 방식), 부록 E와 표 10–12(학습 설정). 게재본과 v2의 수식과 설정은 같습니다.
- 확인 범위: `models/irasim.py` (forward, 임베딩, 블록), `main.py` (학습 루프, 옵티마이저, EMA, 체크포인트 로드), `configs/` (학습, 평가, 확산), `evaluate/compute_latent_l2.py`, `scripts/install.sh`, `README.md`, `LICENSE`.
- 확인하지 않은 것: `diffusion/mask_gaussian_diffusion.py`의 손실 세부, `dataset/` 전처리, `evaluate/compute_psnr_ssim.py`의 세부 평균 방식, `sample/pipeline_trajectory2videogen.py`의 샘플링 세부. 시간 제약으로 생략했습니다.

## 1. 요약

- **Frame-Ada의 핵심(프레임별 행동 임베딩이 공간 블록의 adaLN 계수를 만든다)은 코드에 있습니다.** 16개 조건 토큰(학습 가능한 마스크 토큰 1개 + 행동 15개)을 프레임과 1:1로 더해 공간 블록에 넣습니다 (`models/irasim.py`, extras == 3 분기).
- **그러나 시간 블록은 행동을 받지 않습니다 (D1).** 논문 부록 C.2는 시간 블록이 궤적 임베딩을 쓴다고 쓰지만, 코드는 확산 시각 임베딩만 넣습니다. 최종 레이어도 기본 설정에서 행동을 받지 않습니다 (D2).
- **논문에 없는 학습 기법이 있습니다.** 학습 중 행동 조건을 10% 확률로 지우는 마스킹이 있습니다 (D3). 추론의 분류기 없는 안내는 끕니다 (guidance_scale 1.0).
- **설정 수치의 일부가 어긋납니다.** 학습 스텝 (300k 대 3M, D5)과 유효 배치 (D6)가 표 10, 부록 E와 맞지 않습니다.
- **논문 실험 일부는 저장소에 없습니다.** 정책 평가(LIBERO), 모델 기반 계획(Push-T, 실로봇), 보상 모델, RoboNet 설정이 빠져 있습니다 (D8, D10).
- **이 맥(CUDA 없음)에서는 그대로 실행할 수 없습니다.** 학습과 응용 진입점이 CUDA를 가정합니다 (D14).

## 2. 논문과 일치하는 항목

| 항목 | 논문 | 코드 (파일과 설정) |
|---|---|---|
| 백본 크기 | XL: 28층, 1152차원, 16헤드, 679M (표 10, 11) | `IRASim_XL_2`: depth 28, hidden 1152, heads 16 (`models/irasim.py` 529행). 파라미터 산술 일치 (`recompute.py`) |
| 패치와 입력 채널 | 패치 2, 입력 4채널 (표 10) | `configs` 기본값과 `IRASim_XL_2` (patch 2, in_channels 4) |
| 조건 프레임과 행동 수 | 1프레임 + 행동 15개 → 16프레임 (§4.1) | `num_frames: 16`, `mask_frame_num: 1`, 조건 토큰 16개 |
| 행동 차원 | RT-1, Bridge 7차원, Language-Table 2차원 (§3.1, 부록 B) | `state_dim` 7 (rt1, bridge), 2 (languagetable) |
| 해상도 | RT-1, Bridge 256×320, Language-Table 288×512 (§4.1) | `configs/` 안 rt1, bridge, languagetable 설정의 `video_size` |
| 옵티마이저 | AdamW, 학습률 1e-4, 가중치 감쇠 0 (표 10) | `torch.optim.AdamW(lr=1e-4, weight_decay=0)` (`main.py` 214행) |
| 학습률 스케줄 | 상수 (부록 E) | `get_scheduler(name="constant")` (`main.py` 240행) |
| 기울기 클리핑 | 0.1 (표 10) | `clip_max_norm: 0.1` |
| EMA | 0.9999 (표 10) | `update_ema(..., decay=0.9999)` 기본값 |
| 예측 대상 | ε-예측 (식 2) | `learn_sigma: False`, 출력 채널 4 |
| 추론 | PNDM 50 스텝 (부록 E) | `infer_num_sampling_steps: 50`, `sample_method: 'PNDM'` |
| 노이즈 스케줄 | 논문에 명시 없음 (식 3의 α̅ 정의만) | `beta_schedule: linear`, 1e-4–0.02 |
| 평가 체크포인트 | 30만 스텝 (그림 5 끝점) | `evaluate_checkpoint: rt1/checkpoints/frame_ada/0300000.pt` |
| 메트릭 | Latent L2 (주 지표) | `evaluate/compute_latent_l2.py`: 첫 프레임을 뺀 SDXL VAE 잠재 공간의 MSE (D12) |

## 3. 차이 (D1–D17)

**D1. Frame-Ada 시간 블록의 조건 (중요).** 논문 식 6–7과 부록 C.2는 시간 블록이 "영상 단위와 같은 방식으로 만든 임베딩"을 쓴다고 씁니다 (궤적 임베딩 + 확산 시각 임베딩). 코드의 `forward`에서 `extras == 3`일 때 시간 블록의 `c`는 `timestep_temp` 하나뿐입니다. 행동 정보가 시간 블록에 들어가지 않습니다. 공간 블록만 프레임별 행동을 받습니다. 논문은 이 차이를 말하지 않습니다. 논문의 "프레임 간 정렬"은 공간 블록에서만 일어납니다.
- 근거: `models/irasim.py` forward, `if self.extras == 3: c = timestep_temp`.

**D2. 최종 레이어 조건.** `FinalLayer`의 `c`는 확산 시각 임베딩이며, `final_frame_ada`가 참일 때만 행동 임베딩을 더합니다. 설정 파일은 모두 `final_frame_ada: False`입니다. 코드 주석은 "IRASim-Frame-Ada 체크포인트는 마지막 ada 없이 학습되었다, legacy bug"라고 적습니다. 논문 그림 2(c)와 부록 C.2는 최종 레이어를 언급하지 않습니다.

**D3. 학습 중 행동 조건 마스킹 10%.** `models/irasim.py`의 `self.training` 분기는 배치 샘플의 10%에서 행동 임베딩 전체를 마스크 임베딩으로 바꿉니다. 논문은 이를 쓰지 않습니다. 추론에서 `guidance_scale: 1.0`이라 분류기 없는 안내는 쓰지 않습니다. 따라서 이 마스킹은 학습 정규화이며, 조건 없는 생성 모델을 함께 학습한다는 효과가 있습니다. 논문의 "행동이 조건"이라는 설명과 함께 읽을 때 영향을 줄 수 있습니다.

**D4. 이력 프레임 조건 방식.** 논문 §3.3은 과거 프레임을 "노이즈 없는 입력 토큰"으로 둔다고 씁니다. 코드는 이 외에 행동 앞에 학습 가능한 마스크 토큰을 한 개 붙여 16개 조건 토큰을 만듭니다 (`torch.cat([batch_front_mask_emb, state_embeddings])`). 이 토큰이 첫 프레임의 조건 역할을 합니다. 논문은 이 토큰을 설명하지 않습니다.

**D5. 학습 스텝.** 본문(§4.1, 부록 E), 그림 5 축(최대 300K), 설정 파일(`max_train_steps: 300000`)은 30만입니다. 표 10은 "Training steps 3000000"으로 300만입니다. 표 10이 오타로 보입니다.

**D6. 유효 배치 크기.** 부록 E는 배치 64라고 씁니다. 코드의 유효 배치는 `local_batch_size (1) × gradient_accumulation_steps × GPU 수`입니다. RT-1 설정은 accumulation 2이므로 GPU 32개에서 64가 됩니다. Language-Table 설정은 accumulation 1이므로 GPU 32개에서 32, 64개에서 64입니다. 표 12는 Language-Table도 동시 GPU 32개라고 씁니다. 이 경우 논문의 배치 64와 맞지 않습니다.

**D7. 드롭아웃 0.1 (표 10).** `Attention`의 `attn_drop`과 `Mlp`의 `drop`은 기본값 0입니다. `dropout_p=0.1`은 `flash_attn` 분기에만 있고, 설정은 `attention_mode: 'math'`입니다. 기본 경로에서는 드롭아웃 0.1이 적용되지 않습니다.

**D8. 논문 실험 중 저장소에 없는 것.** `grep`으로 확인한 범위에서 다음은 없습니다.
- §4.2 LIBERO 정책 평가 (확산 정책 학습, 시뮬 롤아웃, 성공 판정)
- §4.3 Push-T 계획 (확산 정책, IoU 가치 모델, 사후 롤아웃 수집, GPC 기준선)
- §4.3 실로봇 계획 (목표 이미지 점수, 샘플링 계획기)
- OpenSora 가중치로 초기화하는 코드 (`main.py`의 부분 로드는 평가 체크포인트용)
- 키보드와 VR 응용은 있습니다 (`application/languagetable.py`, `application/vive_controller_traj2video.py`).

**D9. FID와 FVD 구현.** `pytorch-fid`와 `stylegan-v`는 `.gitmodules` 없이 gitlink(서브모듈)로만 있고, 얕은 복제에서 비어 있습니다. FVD는 `stylegan-v` 코드를 씁니다 (README 감사 항목). `stylegan-v` 저장소는 GitHub API에서 라이선스가 없습니다(null). 이 코드를 가져다 쓰면 라이선스 문제가 생깁니다.

**D10. RoboNet 설정 없음.** `configs/`에는 rt1, bridge, languagetable만 있습니다. 표 2 (RoboNet, 2프레임 + 행동 5차원 패딩, 10프레임)를 설정만으로 재현할 수 없습니다. `dataset_structure.txt`와 데이터 로더도 확인하지 못했습니다.

**D11. README 오류.** (a) 인용 항목에 `arXiv:2406.12802`라고 적혔지만 이 논문은 `2406.14540`입니다. 제목도 v1 제목 "Learning Interactive Real-Robot Action Simulators"입니다. (b) 평가 명령이 `evaluate/evaluation_short_script.py`와 `evaluate/evaluation_long_script.py`를 가리키지만, 실제 파일은 `evaluate_short_script.py`와 `evaluate_long_script.py`입니다.

**D12. Latent L2의 정의.** `compute_latent_l2.py`는 예측과 정답의 잠재 표현에서 첫 프레임을 뺀 뒤 `F.mse_loss`를 씁니다. 평균 제곱 오차이고, 파일 단위로 평균을 냅니다. 잠재 공간은 SDXL VAE입니다. IRASim과 LVDM은 이 공간에서 학습하므로, 이 지표는 잠재 공간 모델에 유리할 수 있습니다. 픽셀 공간 모델인 VDM은 같은 VAE로 다시 인코딩한 뒤 비교합니다. 논문은 이 편향을 논의하지 않습니다.

**D13. 데이터와 체크포인트 용량.** README 표 기준으로 RT-1은 학습 86G, 평가 100G, 체크포인트 29G, Bridge는 31G, 63G, 32G, Language-Table은 200G, 194G, 34G입니다. 합계는 약 769GB입니다. 개별 체크포인트 파일의 크기는 확인하지 못했습니다.

**D14. CUDA 전제.** `scripts/install.sh`는 `cu118` 빌드의 torch를 설치합니다. `main.py`는 `assert torch.cuda.is_available()`이고, `application/languagetable.py`도 같은 assert와 `torch.device("cuda", 0)`를 씁니다. `models/irasim.py`는 CUDA가 없으면 CPU로 떨어지는 분기가 있지만, 학습 진입점은 CUDA만 허용합니다. 맥에서 쓰려면 장치 코드를 바꿔야 합니다 (MPS 포팅).

**D15. 선택 의존성.** `flash-attn`은 설치 스크립트에서 선택 사항입니다. 기본 `attention_mode: 'math'`는 flash-attn 없이 돌아갑니다.

**D16. 식 8–9의 표기.** 논문 식 8–9는 `γ × LN(x + β)` 형태로 β를 LayerNorm 안에 넣고 γ를 1 없이 곱합니다. 코드는 `TransformerBlock.forward`에서 `x + gate × attn(modulate(LN(x)))` 형태이고 `modulate(x) = x × (1 + scale) + shift`입니다 (adaLN-Zero, 0으로 초기화). 식의 표기가 다르고, 식 8–9는 오타일 가능성이 큽니다.

**D17. 체크포인트 부분 로드.** `main.py`는 체크포인트 키 중 모델에 있는 것만 불러오고, 없는 키는 로그에만 남깁니다 (`Ignoring: ...`, 로드 비율 출력). 키 이름이 어긋나도 오류 없이 진행될 수 있습니다. 재현할 때 로드 비율을 꼭 확인해야 합니다.

## 4. 구현 가능성 판정 (맥: Apple M4 Pro, 약 24 GiB 통합 메모리, MPS, CUDA 없음)

**판정: 부분 구현 가능. 아키텍처 재구현과 추론 검사는 가능하고, 학습과 정책·계획 실험은 GPU와 데이터가 필요해서 이 맥에서는 불가입니다.**

| 항목 | 판정 | 근거와 노력 추정 |
|---|---|---|
| 논문 표 재계산과 검정 | 완료 | `analysis/recompute.py`, 수 초 |
| 파라미터 산술 (표 11) | 완료 | DiT식 블록 산술이 33 / 132 / 461 / 679M과 맞음 |
| 정적 코드 대조 | 완료 | 이 문서 |
| 프레임 조건 아키텍처 재구현 (독립 구현, 소형 설정) | 가능 (측정 전) | `models/irasim.py`를 PyTorch로 옮기면 약 600줄. 1–2주 (추정). 행동 조건이 프레임에 맞는지 확인하는 단위 검사는 CPU에서 가능 |
| 공식 추론 파이프라인을 MPS로 포팅 | 조건부 | `cuda` 하드코딩 제거, `math` 어텐션 경로 사용, fp32 또는 fp16 검사. 체크포인트 다운로드 (묶음 29–34GB 중 Frame-Ada 파일 크기 미확인) 승인 필요. 16프레임 한 개 생성 시간은 측정 전. A100 30초의 몇 배가 될지 모름. 2–4일 (추정) |
| 학습 재현 (RT-1, Bridge, Language-Table) | 불가 | 표 12: 데이터셋당 약 2,370 GPU 시간 (동시 32 GPU면 약 74시간). 데이터 최대 200GB. GPU 클러스터 필요 |
| 정책 평가 (§4.2) | 불가 | 공식 코드 없음 (D8). 확산 정책 학습과 IRASim 학습 모두 GPU 필요. 사후 롤아웃 데이터 미공개 |
| Push-T 계획 (§4.3) | 불가 | 확산 정책, IoU 가치 모델, IRASim 학습 필요. 100회 × K 후보 × 여러 단계의 영상 생성. 맥에서는 시간이 현실적이지 않음 |
| 실로봇 계획 | 불가 | 실로봇과 데이터 필요 |
| 키보드 응용 (Language-Table) | 조건부 | 코드는 CUDA assert가 있어 MPS 포팅 후 가능. 체크포인트 필요 (D14) |

- **이 맥에서 가능한 가장 큰 검증:** 프레임 조건 블록을 PyTorch로 다시 쓰고, 행동 하나를 바꿨을 때 해당 프레임의 공간 블록 출력만 바뀌는지 확인하는 단위 검사입니다. 이것은 D1–D2를 직접 검증하는 방법입니다. 실행 전에 설치 목록을 승인받아야 합니다.
- **통계적으로 가치가 큰 재현:** 정책 평가 상관 (표 4)을 다른 과제에서 다시 재는 것입니다. 그러나 GPU 없이는 불가입니다.
- 설치와 다운로드, 공식 코드 실행은 사용자 승인 없이 하지 않았습니다.

## 5. 라이선스

- **공식 코드:** Apache License 2.0 (`LICENSE` 파일, 소스 머리말 "Copyright (2024) Bytedance Ltd. and/or its affiliates"). 복제본은 정적 읽기만 했고, 이 저장소에는 코드를 넣지 않습니다.
- **가져다 쓴 코드:** Latte (Apache-2.0), pytorch-fid (Apache-2.0), stylegan-v (라이선스 없음, D9).
- **데이터셋:** RT-1, Bridge, Language-Table, RoboNet은 각자 라이선스가 있습니다. 확인하지 않았습니다.
- **논문 PDF:** arXiv 페이지의 "view license"는 `http://creativecommons.org/licenses/by/4.0/`를 가리킵니다 (CC BY 4.0). 자세한 권리 검토는 `README.md`에 적었습니다.

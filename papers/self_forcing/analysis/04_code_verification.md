# 04. 공식 코드 대조와 구현 가능성

공식 저장소 `guandeh17/Self-Forcing`(커밋 `33593df`, Apache-2.0)을 얕게 복제해 **정적으로만** 읽었습니다. 설치, 실행, import, 체크포인트 다운로드는 하지 않았습니다. 저장소 내용은 외부 데이터로 취급했습니다. 대조 대상 파일은 `configs/self_forcing_dmd.yaml`, `configs/self_forcing_sid.yaml`, `configs/default_config.yaml`, `pipeline/self_forcing_training.py`, `pipeline/causal_inference.py`, `model/dmd.py`, `model/sid.py`, `model/gan.py`, `utils/wan_wrapper.py`, `wan/modules/causal_model.py`, `scripts/generate_ode_pairs.py`입니다.

## 1. 전체 판정
- **알고리즘 골격은 논문 Alg. 1과 맞습니다.** 무작위 단계 추출, 마지막 단계만 기울기, KV 캐시 기울기 차단이 코드에 있습니다.
- **학습 하이퍼파라미터는 Table 3과 맞습니다.** 학습률, 배치, 옵티마이저, 업데이트 비율, 교사 모델 크기가 DMD와 SiD 설정 파일에 그대로 있습니다.
- **논문 본문에 없는 차이가 여러 개 있습니다.** DMD 시간 범위 제한, DMD 기울기 정규화, EMA 시작 시점, GAN 정규화 식 차이와 기본값입니다.
- **가장 중요한 누락은 롤링 캐시 학습 설정입니다.** 논문은 첫 청크 차단을 학습 방법으로 제시하지만(§3.4, 부록 B), 공개된 학습 설정에는 그 옵션(`local_attn_size`)이 없습니다(D8).
- **GAN 설정 파일이 없습니다.** Table 2와 Table 3의 GAN 행은 저장소만으로 재현할 수 없습니다(D9).

## 2. 대조표 (D)

| 번호 | 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|---|
| D1 | 기울기 적용 범위 | 마지막 단계 출력만 기울기 (Alg. 1, §3.2) | 샘플링된 exit 단계에서만 `enable grad`. 추가로 `start_gradient_frame_index = frames − 21`로 21프레임보다 앞은 기울기를 끊음 (`self_forcing_training.py`) | 일치 (학습 기본값 21프레임에서는 추가 절단이 없음). 21프레임을 넘는 롤아웃에서 차이 |
| D2 | 디노이징 단계 무작위 추출 | s ~ Uniform(1..T), 영상마다 한 번 (Alg. 1 줄 4) | `randint` 4개 중 하나, `same_step_across_blocks: true`로 블록 간 공유 (`default_config.yaml`) | 일치 |
| D3 | KV 캐시로 들어가는 값 | 기울기 차단 (§3.2) | 캐시를 `no_grad`로 다시 씀 (timestep 0 재실행) | 일치 |
| D4 | 단계 스케줄과 시간 이동 | [1000, 750, 500, 250], k = 5 (부록 A) | `denoising_step_list` 동일, `timestep_shift: 5.0`, `warp_denoising_step: true` | 일치 |
| D5 | DMD 샘플링 시간 범위 | 본문에 없음 | `min_step = 2%`, `max_step = 98%`로 잘라냄 (`model/dmd.py` 36–37, 170) | **논문 미기재** |
| D6 | DMD 기울기 정규화 | 식 (2)–(3)에 정규화 없음 | `grad / mean(|p_real|)`로 나눔 (`dmd.py` 117–120, DMD 논문 식 8 방식) | **논문 미기재** |
| D7 | DMD 가이드 | 실제 점수 CFG 3.0 (Table 3) | `real_guidance = 3.0`, `fake_guidance = 0.0` | 일치 (가짜 점수 CFG는 논문에 없음) |
| D8 | 롤링 캐시 학습의 첫 청크 차단 | 학습 중 마지막 청크가 첫 청크를 못 보게 함 (§3.4, 부록 B) | 기능 코드는 있음 (`causal_model.py`의 `local_attn_size` 축출 로직). 그러나 공개 YAML의 `model_kwargs`에는 `timestep_shift`만 있어 기본값 `local_attn_size = -1`이 쓰이고, 축출 분기가 실행되지 않음 | **설정 누락 또는 미공개.** 공개 설정으로는 논문의 롤링 캐시 학습을 재현할 수 없음 |
| D9 | GAN 학습 설정 | 배치 768, λ = 30, σ = 0.05, 실제 점수 없음 (Table 3, 부록 A) | `configs/`에 GAN 설정 없음 (DMD와 SiD만). `model/gan.py` 기본값은 `r1_weight = r2_weight = 0.0`, `r1_sigma = r2_sigma = 0.01`, `gan_g_weight = gan_d_weight = 1e-2` | **재현 불가 (설정 파일 없음).** 기본값이 논문과 다름 |
| D10 | GAN 정규화 식 | 식 (5): ½‖f(x) − f(x+σε)‖² (λ 곱) | `r1_grad = (Δlogit)/σ`, `loss = w · mean(r1_grad²)` (`gan.py` 257–284). σ² 만큼 다른 스케일 | **식이 다름.** 같은 λ로도 실효 세기가 다름 |
| D11 | 생성기 GAN 손실 | 식 (7): −log σ(f(x̂) − f(x)) | `softplus(−(f_fake − f_real))` (`gan.py` 169–170), 가중치 `gan_g_weight` | 일치 (가중치가 추가됨. 논문 미기재) |
| D12 | SiD 손실 | 식 (4), α = 1 (부록 A) | `sid_alpha = 1.0`. 손실식은 α = 1에서 식 (4)와 같음. 코드에 "TODO: Add alpha" 주석 (`sid.py` 126) | 일치 (α ≠ 1은 수식이 맞는지 확인 안 됨) |
| D13 | 학습률, 배치, 옵티마이저 | Table 3 (DMD: lr 2e-6, critic 4e-7, β 0/0.999, wd 0.01; SiD: lr 2e-6, wd 0) | DMD와 SiD YAML 모두 같음. 전역 배치 64 | 일치 |
| D14 | 업데이트 비율, EMA | 비율 5, EMA 0.99 (Table 3) | 비율 5, EMA 0.99. `ema_start_step: 200`은 논문에 없음 | 일치 + 미기재 항목 1개 |
| D15 | 실제 점수 네트워크 크기 | DMD는 Wan2.1-T2V-14B, SiD는 1.3B (Table 3) | DMD `real_name: Wan2.1-T2V-14B`, SiD `real_name: Wan2.1-T2V-1.3B` | 일치 |
| D16 | 시간 이동 공식 | t' = k t/(1 + (k−1)t/1000), k = 5 (부록 A) | `DMD.timestep_shift` 적용 공식과 같음 (`dmd.py` 166–169) | 일치 |
| D17 | ODE 쌍 생성 설정 | 16k 쌍, 기반 모델에서 샘플링. 세부 설정 없음 (§4) | `generate_ode_pairs.py`: `shift = 8.0`, 48 단계, CFG 6.0. 학습은 시간 이동 5.0 | **논문 미기재, 학습 설정과 시간 이동 값이 다름** (의도일 가능성. 확인 불가) |
| D18 | 학습 반복 수 | 본문에 없음 (수렴 1.5 h) | README: 600 반복, 64장 H100에서 2 h 이내 | 논문 미기재 항목 |
| D19 | 프레임 단위 변형 | §4: frame-wise 변형 (Table 1, 2) | 두 YAML 모두 `num_frame_per_block: 3`. frame-wise용 설정 없음. HF 체크포인트 목록(dmd, sid, gan, sid_v2, 10s, ode_init)에 frame-wise 이름 없음 | **frame-wise 결과의 재현 경로가 공개되지 않음** |
| D20 | 평가 프롬프트 수 | MovieGenBench 1003 (부록 E) | `MovieGenVideoBench.txt` 1003줄 | 일치 |
| D21 | 실시간 주장 하드웨어 | H100에서 측정 (§4) | README: "RTX 4090에서 실시간" | **README 주장은 논문 근거가 없음.** 4090 수치는 저장소에도 없음 |
| D22 | 데이터 | 데이터 없이 DMD와 SiD 가능, GAN은 14B로 만든 영상 70k 필요 (§4) | README: 학습은 데이터가 필요 없음 (GAN 제외). `vidprom_filtered_extended.txt`는 HF에서 받음 | 일치. GAN용 14B 영상 70k를 만드는 절차는 확인하지 못함 (`scripts/create_lmdb_14b_shards.py`가 관련 있어 보이나 읽지 않음) |

### 2.1 추가로 확인한 것
- DMD 실제 점수에는 CFG 3.0을 쓰고 가짜 점수에는 쓰지 않습니다(`dmd.py` 81–110). 이 설계는 Table 3과 맞지만, 논문 식 (2)는 CFG를 보여 주지 않습니다.
- 캐시 크기: 학습 기본값은 21프레임 × 1560 토큰입니다. 추론 기본값은 32760 토큰(21프레임)으로 고정됩니다(`causal_inference.py` 280–288). 10초 영상에서 캐시를 밀어 내는 경로는 `local_attn_size`가 설정될 때만 실행됩니다.
- 공식 저장소의 명령 `torchrun --nnodes=8 --nproc_per_node=8`은 64장 학습을 가정합니다. README는 8장에서 그라디언트 누적으로 16 h 안에 재현 가능하다고 씁니다(추정, 검증 안 함).

## 3. 라이선스와 권리

| 대상 | 라이선스 | 근거 |
|---|---|---|
| 논문 PDF | CC BY-NC-SA 4.0 | arXiv abs 페이지 "view license", `creativecommons.org/licenses/by-nc-sa/4.0/` (`01` §0, 아래 "권리 판정" 참고) |
| 공식 코드 | Apache-2.0 | GitHub 메타데이터, LICENSE 파일 (2025-09-12 추가) |
| 체크포인트 | Apache-2.0 (모델 카드 메타데이터) | Hugging Face `cardData.license` = apache-2.0. 최상위 license 필드는 비어 있음 |
| Wan2.1 기반 코드 | Apache-2.0 | GitHub 메타데이터 |
| CausVid | NOASSERTION | GitHub 메타데이터. 확인하지 않음 |

**권리 판정 (논문 본문 기준):**
- arXiv abs 페이지(2026-11-21 확인)는 `creativecommons.org/licenses/by-nc-sa/4.0/`를 가리키고, 아이콘은 `by-nc-sa-4.0.png`입니다. 메타데이터 판정은 CC BY-NC-SA 4.0입니다. CC BY가 아닙니다.
- PDF 본문에 ©, Copyright, All rights reserved, 저작권 표시는 없습니다. 머리말은 "39th Conference on NeurIPS 2025"와 저자 소속(Adobe Research, UT Austin)입니다. 회사 고지는 소속 표기뿐입니다.
- PDF 메타데이터에도 저작권이나 권리 문구가 없습니다(`pdfinfo`는 제목과 저자만, 생성 도구는 arXiv GenPDF).
- **따라서 공개 산출물에 논문 그림을 넣지 않습니다.** 공개본은 우리가 만든 그래프와 텍스트 요약만 씁니다. 논문 그림은 개인용 리포트와 PPT에만 넣을 수 있는데, 이번 판은 넣지 않았습니다.

## 4. 구현 가능성 판정

**판정: 이 맥(Apple M4 Pro, 24 GiB, MPS, CUDA 없음)에서 학습 재현은 불가능합니다. 공개된 1.3B 추론은 조건부로 가능하며, 속도와 메모리는 측정하지 않았습니다.**

| 무엇 | 필요 자원 | 맥에서 가능 여부 | 노력 추정 |
|---|---|---|---|
| DMD 학습 (논문 Table 3) | 14B 교사 + 1.3B 가짜 점수 + 생성기, H100 64장 1.5–2 h (D15, README) | **불가.** 메모리와 CUDA 모두 부족 | — |
| SiD 학습 | 1.3B 교사 + 생성기, H100 64장 2–3 h | **불가.** 같은 이유 | — |
| GAN 학습 | 설정 파일 없음 (D9), 14B로 만든 영상 70k 필요 | **불가** (자원과 설정 모두) | — |
| ODE 초기화 16k쌍 생성 | 기반 모델 샘플링 16k회, 설정 불명 (D17) | **불가** (시간 과다, 추정) | — |
| 1.3B 추론 (공개 DMD 체크포인트) | Wan2.1-T2V-1.3B 가중치와 T5 인코더, 체크포인트 `self_forcing_dmd.pt`. 가중치 다운로드는 승인 필요 | **조건부.** 저장소가 CUDA 호출 12곳(`.cuda()`)과 flash-attn 설치를 가정함. MPS 포팅과 어텐션 대체 필요 | 포팅 1–2주 추정. 속도는 측정 전 |
| frame-wise 추론 | 설정 파일 없음 (D19) | 조건부 (위와 같음, 설정은 직접 만들어야 함) | 포팅 이후 추가 0.5–1일 추정 |
| 롤링 캐시 긴 영상 추론 | `local_attn_size` 설정을 직접 정해야 함 (D8) | 조건부 | 추정 1–2일 |
| 논문 표 1, 2 재계산 | 논문 수치만 있으면 됨 | **가능, 완료** (`03`) | 완료 |

- **재현하지 않은 이유:** 사용자 요청에 따라 학습과 무거운 실험을 하지 않았습니다. 체크포인트 다운로드와 공식 코드 실행도 승인 대상이라 하지 않았습니다.
- **CPU 토이:** 선택 사항이었고, 중심 주장을 직접 검사하는 토이가 없어서(CPU에서 2분 안에 노출 편향 효과를 보이는 설계가 명확하지 않아서) 만들지 않았습니다.
- **그럼에도 가져갈 수 있는 것:** 학습 식과 알고리즘은 저장소에서 확인되므로, 같은 구조를 작은 AR 확산 모델로 옮기는 것은 CPU에서도 가능한 연구 설계입니다. 이것은 

## 5. 확인하지 못한 것
- 체크포인트의 실제 가중치 내용은 보지 않았습니다.
- 공식 저장소가 논문 학습을 재현하는지는 확인하지 못했습니다(D8, D9, D17).
- CausVid 저장소와 ODE 초기화 절차는 읽지 않았습니다.

# 04 논문 ↔ 공식 코드 정적 대조

대상: `repo/` (cvlab-columbia/dreamitate, 커밋 `c162da1`). 코드는 읽기만 했고 실행, import, 설치는 하지 않았습니다. 클론한 코드는 신뢰할 수 없는 데이터로 취급했습니다.

## 1. 저장소 구성

| 경로 | 내용 | 논문과의 관계 |
|---|---|---|
| `video_model/` | Stability AI `generative-models` 기반 SVD 미세조정 (`sgm/`, `main.py`, `configs/basile_svd_finetune.yaml`, `scripts/sampling/simple_video_sample_gradio.py`) | §3.2 영상 생성 |
| `megapose/` | MegaPose 기반 도구 추적 (`src/megapose/scripts/run_video_tracking_on_rotation_example_stereo_{left,right}.py`, 예제 데이터, 768×448 리사이즈) | §3.3 추적 |
| `README.md` | 추론·학습 명령, 체크포인트·데이터 다운로드 | 재현 절차 |
| 없음 | 로봇 제어 코드, Diffusion Policy 기준선 코드, Scooping·Sweeping·Push-Shape 설정, 평가 스크립트, 스테레오 시점 임베딩 코드 | 논문 §4 전반 |

## 2. 대조 결과 (D1–Dn)

| 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|
| D1 과제 범위 | 4 과제 (Rotation, Scooping, Sweeping, Push-Shape), 과제별 f_θ (§3.2, Table 1) | 학습 설정은 `basile_svd_finetune.yaml` 하나이며 데이터 경로는 `dataset/rotate/`. 다른 과제 설정·데이터 목록 없음 | **차이**: 공개 코드는 Rotation만 재현 가능 |
| D2 추론 스텝 | "During inference, we use 30 denoising steps" (부록 A) | 학습 설정의 `sampler_config`는 `EulerEDMSampler`, `num_steps: 25`, 주석 "(not used during training)". Gradio 스크립트 기본값은 모델에 따라 14/25 (SVD) 또는 30 (SVD-XT) 프레임·스텝 | **불확실**: 30 스텝이 어떤 설정에서 쓰였는지 코드만으로 확정 못함 |
| D3 가이던스 | "constant classifier-free guidance of 1.0" (부록 A) | 학습 설정 `guider_config`는 `LinearPredictionGuider`, `max_scale: 2.5`, `min_scale: 1.0` (선형 증가) | **차이 가능성**: 논문은 상수 1.0, 코드 설정은 1.0→2.5 선형. 추론 시 실제 값은 Gradio 경로에서 확인 필요 |
| D4 스테레오 구조 | 25프레임 = 뷰1 13 + 뷰2 12, 첫 프레임 버림, 프레임별 임베딩을 시점에 맞춰 수정 (부록 A) | `video_model/` 코드에 "stereo" 문자열 없음. 스테레오는 README와 `megapose` 쪽 추적 스크립트에만 등장 | **구현 부재**: 스테레오 조건화 코드는 공개 저장소에서 찾지 못함 |
| D5 학습 설정 | Table 3: lr 1e-5, batch 4, 16384 steps, 2 s 클립, 6 fps, motion score 200 | lr 1e-5 (`base_learning_rate`) 일치, batch_size 1 (dataloader), `accumulate_grad_batches: 1`. max_steps는 설정에서 찾지 못함 | **일부 일치**: lr 일치, step 수·fps·motion score는 확인 안 됨 (`main.py`의 `scale_lr` 옵션 있음) |
| D6 고정·미세조정 범위 | 인코더·디코더 고정, spatial/temporal attention만 미세조정 (§3.2) | 설정에 `freeze: True` 항목 있음 (약 57행). 어떤 모듈이 고정인지는 전체 파일을 읽어야 확정 | **일치 추정**: 자세한 확인 필요 |
| D7 추적 | MegaPose, 768×448, Realsense 기본 내부 파라미터 (§3.3) | 추적 스크립트가 768×448 리사이즈 사용 (`cv2.resize (768, 448)`) | **일치** |
| D8 기준선 | Diffusion Policy, ResNet-18, 200 epoch, 12 스텝 예측 (§4.1) | 공개 저장소에 DP 코드 없음 | **확인 불가**: 기준선 재현 근거는 논문 설명뿐 |
| D9 평가 | 시도 수, 성공 기준 (Table 1, §4.2–4.5) | 평가 스크립트 없음 | **확인 불가** |

## 3. 요약

- 공개 코드는 **Rotation 과제의 SVD 미세조정과 MegaPose 추적 예제**에 한정됩니다. 논문 Table 2의 4개 과제 결과를 코드로 재현할 수 있는 상태가 아닙니다.
- 추론 스텝(30 vs 25)과 가이던스(1.0 vs 1.0–2.5)는 논문과 설정 파일이 다릅니다. 실제 추론 설정은 Gradio 경로와 체크포인트 조합에 따라 달라질 수 있어 확정하지 못했습니다.
- 스테레오 조건화 코드가 없습니다. 논문의 핵심 설계 중 하나(뷰별 임베딩)는 코드로 대조할 수 없습니다.

## 4. 라이선스 주의

- 저장소 최상위 `LICENSE.txt`는 **CC BY-NC 4.0**입니다 (GitHub API는 NOASSERTION으로 표시). 비상업 조건이 붙습니다.
- `video_model/LICENSE-CODE`는 MIT (Stability AI). 모델 가중치는 `model_licenses/`의 SDV·SDXL 라이선스를 따릅니다.
- 이 코드를 재사용하려면 저장소 전체 라이선스 조건을 따로 확인해야 합니다.

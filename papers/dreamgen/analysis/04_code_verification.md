# 04. 코드 대조: 논문 ↔ 공식 저장소 (정적 읽기)

- 대상: `NVIDIA/GR00T-Dreams` (Apache-2.0), 커밋 `ec3881d` (2025-10-23, PR #14 병합), `dreamgen/repo/`에 얕은 복제 (`--depth 1`).
- 방법: 파일을 읽기만 했습니다. 실행, import, 설치, 가중치와 데이터 다운로드, API 호출은 하지 않았습니다. 복제본과 그 안의 내용은 신뢰하지 않는 데이터로 취급했습니다.
- 주의: 저장소의 README는 영상 세계모델로 **Cosmos-Predict2**를 씁니다. 논문 주요 실험은 **WAN2.1**입니다(§2.1, Table 2). 이 차이는 D1에서 다룹니다.
- 확인 범위: 논문의 방법 정의(§2)와 평가 절차(§4, App. H)를 저장소의 스크립트와 대조했습니다. 논문 실험 결과의 재현 여부는 확인할 수 없습니다.

## 1. 대조 요약

| # | 항목 | 논문 | 저장소 | 판정 |
|---|---|---|---|---|
| D1 | 영상 세계모델 | WAN2.1 주로, 일부 CogVideoX·Cosmos·Hunyuan (§2.1) | Cosmos-Predict2 (README), 미세조정은 외부 cosmos-predict2 문서 참조 | **차이** |
| D2 | 신경 궤적의 행동 인코더·디코더 | 실제와 신경 궤적을 **별도 embodiment**로 취급 (§2.4) | 신경 궤적은 상태만 0으로 바꾸고, `embodiment_id`는 실제와 같음 | **차이 (이 경로에서)** |
| D3 | 신경 궤적의 상태 입력 0 | 0 (§2.4, 각주 4) | `is_dream_instance`이면 `state = zeros` | 일치 |
| D4 | IDM 창 이동과 집계 | "H개 예측 후 한 칸씩 밀기" (§2.3) | 시작 프레임마다 창 생성 (stride 1), 같은 프레임의 예측을 **평균** | **논문에 없는 세부** |
| D5 | IDM 입력 | 두 프레임, 언어·고유감각 없음 (§2.3) | 프레임 0과 16, 설정 파일에는 상태·언어 모달리티가 있지만 행동 헤드 순전파는 영상과 행동만 씀 | 일치 (설정에는 남아 있음) |
| D6 | IDM 백본 | SigLIP-2, flow matching (§2.3) | `google/siglip2-large-patch16-256`, flow matching 헤드 | 일치 |
| D7 | LAPA 잠재 행동 라벨러 | 1초 뒤 프레임, VQ 임베딩 (§2.3, App. A) | 모델 코드 없음. `lapa_action` 키 처리만 있음 | **공개되지 않음** |
| D8 | IF(GPT) 평가 설정 | 프레임 수 등 기재 없음 | 8프레임, 0.5배, `gpt-4o`, temperature 0, seed | 논문에 없는 세부 |
| D9 | IF(GPT) 실패 처리 | 기재 없음 | API 3회 실패 시 해당 영상을 **0점**으로 처리 (조용히) | **편향 가능성** |
| D10 | IF(Qwen) 평가 설정 | Qwen2.5-VL-7B (§4) | 49프레임, 0.3배, 이미지로 입력 | 논문에 없는 세부 |
| D11 | IF 프롬프트 | App. H.1 프롬프트 | GPT 스크립트 문구 = 논문 H.1 문구. Qwen 스크립트 문구는 다름 | 부분 일치 |
| D12 | 제로샷 프롬프트 | App. H.1 (제로샷) | 같은 문구 | 일치 |
| D13 | PA 점수 구성 | VideoCon-Physics와 Qwen의 평균 (§4, H.2) | `eval_qwen_pa.py`는 Qwen 부분만. VideoCon은 외부 submodule (`videophy`, 복제 안 됨) | 부분 구현 |
| D14 | PA 프롬프트 | "good physics dynamics … aligned with the physical world" (H.2) | "Please be a strict judge. If it breaks the laws of physics…" | **문구 다름** |
| D15 | 입력 방식 | 명시 없음 | IF는 프레임 이미지, PA는 영상 원본 | 두 지표의 입력이 다름 |
| D16 | `--zeroshot` 인자 | 해당 없음 | `type=bool`이라 `--zeroshot false`가 True로 읽힘 | 사용 주의 (기본값 False는 정상) |
| D17 | 공개 체크포인트·영상 | 없음 | README에 가중치 링크 없음 (정적 확인) | 재현 불가 |
| D18 | 정책 학습 | GR00T N1, 20k 스텝 권장 (README) | `IDM_dump/scripts/finetune/*.sh` (gr1, franka, so100, robocasa) | 일치 (실행 안 함) |

## 2. 항목별 근거

### D1 영상 세계모델
- `README.md` 1행 주변: "we provide the full pipeline for DreamGen, as Cosmos-Predict2 as the video world model in the repository."
- 미세조정 절차는 `cosmos-predict2` 저장소의 문서(`post-training_video2world_gr00t.md`)를 가리킵니다. 이 저장소에는 WAN2.1 미세조정 코드나 설정이 없습니다(파일 목록 확인).
- 논문의 LoRA 설정(lr 1e-4, rank 4, alpha 4, App. D)은 이 저장소에서 확인할 수 없습니다.
- 해석: 논문 Table 2의 WAN 결과와 Cosmos 결과를 같은 파이프라인으로 재현하는 코드는 공개 저장소에 없습니다.

### D2 신경 궤적의 embodiment 분리 (차이)
- 논문 §2.4: "we treat the two types of trajectories as separate embodiments by using separate action encoder and decoder."
- `gr00t/model/transforms.py` 268–298행:
  - 언어에 `<DREAM>`이 있으면 `is_dream_instance = True`.
  - `if is_dream_instance: transformed_data["state"] = np.zeros_like(...)` (D3 근거).
  - `transformed_data["embodiment_id"] = self.get_embodiment_tag()` 는 두 경우 모두 같은 태그를 씁니다.
- 저장소 전체에서 신경 궤적용 별도 embodiment 태그나 별도 인코더·디코더 선택 코드를 찾지 못했습니다.
- 판정: 이 저장소의 경로에서는 논문 설명대로 행동 인코더·디코더가 나뉘지 않습니다. 논문 실험에 쓴 버전이 다를 수 있습니다. 확정하려면 저자에게 확인이 필요합니다.
- 이 차이는 논문이 실로봇 향상을 "GR00T N1에서 더 크다"고 설명한 대목(§3.1, p.6 l.350–352)의 근거와 관련됩니다.

### D4 IDM 창 이동과 집계 (논문에 없는 세부)
- `IDM_dump/base.yaml`: `action_horizon: 16`.
- `IDM_dump/dump_idm_actions.py` 205행: `for step_idx in range(length):` — 프레임마다 창을 하나씩 만듭니다. 즉 stride 1입니다.
- 287–290행: 같은 프레임 s에 들어오는 모든 예측을 모아 `np.mean(action_dict[s], axis=0)`로 평균합니다. 최대 16개 예측이 평균됩니다.
- 논문 §2.3은 "H개 예측 후 한 칸 밀어 다음 H개를 예측"이라고만 씁니다. 겹치는 예측의 평균에 대한 설명은 없습니다.
- 이 평균은 행동을 부드럽게 만들고 잡음을 줄입니다. 신경 궤적 품질과 정책 성능에 영향을 줄 수 있습니다. 이 설정을 논문 결과와 맞춰 확인할 수는 없습니다.

### D5 IDM 입력
- `gr00t/experiment/data_config_idm.py`: `observation_indices = [0, 16]`, `action_indices = list(range(16))`. 프레임은 두 장입니다(논문과 일치).
- 같은 파일에 `state_keys`, `language_keys`가 있어 데이터 로더는 상태와 언어를 만듭니다.
- 그러나 `gr00t/model/action_head/flow_matching_action_head_idm.py`의 순전파(372–404행)는 `encode_images`와 행동 임베딩만 씁니다. 상태와 언어는 행동 헤드에 들어가지 않습니다.
- 판정: 실제 연산 경로는 논문 설명과 맞습니다. 설정 파일에는 상태와 언어가 남아 있어서 읽는 사람이 헷갈릴 수 있습니다.

### D6 IDM 백본
- `gr00t/model/idm.py` 및 `base.yaml`: `google/siglip2-large-patch16-256`, 확산 트랜스포머 DiT 8층, flow matching. 논문 §2.3과 일치합니다.

### D7 LAPA 라벨러 (공개되지 않음)
- 저장소에서 `lapa` 문자열은 `gr00t/data/dataset.py` 444행(키 검사 생략), `IDM_dump/raw_to_lerobot.py`, 노트북, 문서에만 나옵니다.
- LAPA 잠재 행동 모델의 학습·추론 코드, 가중치 경로는 없습니다.
- 판정: 논문 Fig. 4의 LAPA 실험 결과는 이 저장소로 재현할 수 없습니다. 원 LAPA 저장소(latentactionpretraining.github.io 프로젝트 페이지)는 이번에 확인하지 않았습니다.

### D8–D10 평가 스크립트 설정
- GPT-4o IF (`dreamgenbench/eval_sr_gpt4o_whole.py`):
  - 69행: `sample_video_frames(vid_path, num_frames=8, scale_factor=0.5)`.
  - 79–84행: `model="gpt-4o"`, `seed`, `temperature=0.0`, `top_p=1.0`, `max_tokens=4`.
- Qwen IF (`dreamgenbench/eval_sr_qwen_whole.py`):
  - 58–59행 주석은 "Sample 16 frames"인데 실제 호출은 `num_frames=49, scale_factor=0.3`입니다. 주석과 코드가 다릅니다.
  - 49장의 이미지를 한 번에 넣습니다. GPT 설정(8장)과 프레임 수가 다릅니다. 두 IF 열을 서로 직접 비교할 때 이 차이를 고려해야 합니다.
- Qwen PA (`dreamgenbench/eval_qwen_pa.py`): 영상 원본(`{"type": "video"}`)을 넣고 `max_new_tokens=4`, 첫 글자가 "1"이면 1, 아니면 0.
- 이 세 설정은 논문 본문과 App. H에 없습니다. 재현하려면 이 값을 써야 합니다.

### D9 GPT-4o 실패 처리 (편향 가능성)
- `eval_sr_gpt4o_whole.py` 94행: `pred = 0 if response is None else (...)`.
- 재시도 3회가 모두 실패하면 `response`가 None이 되고 그 영상은 0점으로 기록됩니다. 로그 문구는 "Skipping video"이지만 결과 CSV에서는 0으로 세어집니다.
- 실패가 많은 실행에서는 IF(GPT) 값이 낮아질 수 있습니다. 논문이 실패율을 보고했는지는 확인할 수 없습니다.

### D11, D12 프롬프트
- 논문 App. H.1의 "Success Rate" 프롬프트 문구는 GPT 스크립트 64행의 문구와 같습니다("Please evaluate: if the video follows the instruction to finish the task '{prompt}', give a positive score. Reply only '0' for No or '1' for Yes.").
- Qwen 스크립트(`eval_sr_qwen_whole.py` 54행)는 다른 문구("Does the video follow the instruction … Answer 0 for No or 1 for Yes")를 씁니다.
- 제로샷 문구는 논문과 코드가 같습니다(D12).
- 판정: "같은 프롬프트"라는 설명은 GPT 쪽에만 맞습니다. Qwen 쪽에서 쓴 문구는 논문에 적힌 문구와 다릅니다.

### D13, D14 물리 정합성 PA
- 논문 §4와 H.2: PA = (VideoCon-Physics 점수 + Qwen 점수) / 2, 프롬프트 "Does the video show good physics dynamics that is aligned with the physical world?".
- 저장소의 `eval_qwen_pa.py`는 Qwen 점수만 구합니다. VideoCon 부분은 `dreamgenbench/videophy` submodule(`Hritikbansal/videophy`, MIT)에 있고, 얕은 복제로는 받지 않았습니다. README는 "average the score of PA I & II"라고 씁니다.
- Qwen PA 프롬프트는 "Please be a strict judge. If it breaks the laws of physics, please answer 0."로 논문 H.2 문구와 다릅니다.
- PA의 `prompt` 변수는 파일 이름에서 만들지만 프롬프트에 쓰이지 않습니다(사용되지 않는 변수).

### D16 `--zeroshot` 인자
- `eval_sr_gpt4o_whole.py` 117행, `eval_sr_qwen_whole.py` 139행: `type=bool`. `--zeroshot false`는 Python에서 `bool("false") == True`가 되어 제로샷 모드가 켜집니다. README 예시는 `--zeroshot true`만 씁니다. 기본값은 False라 기본 실행에는 영향이 없습니다.

### D17 가중치와 데이터
- README에는 IDM 체크포인트, DreamGen 미세조정 체크포인트, DreamGen Bench 영상 링크가 없습니다.
- `demo_data/robot_sim.PickNPlace`는 소형 예제 데이터(에피소드 5개)입니다.
- 논문의 GR1 실로봇 데이터, Franka 데이터, RoboCasa 신경 궤적 240k는 공개되지 않았습니다.

### D18 정책 학습 스크립트
- `IDM_dump/scripts/finetune/gr1.sh` 등: `scripts/gr00t_finetune.py` 호출, GR1은 `--data-config gr1_arms_waist`.
- README: "boost your batch size to the max, and train for 20k steps." 논문의 학습 스텝은 본문에 없습니다.

## 3. 저장소에서 확인된 일치 항목
- 신경 궤적의 상태 0 (D3).
- IDM 두 프레임과 flow matching, SigLIP-2 (D5, D6).
- 제로샷 IF 프롬프트 (D12).
- DreamGen Bench의 영상 수 규모: Table 2의 48·50·47·30 프레임 설정은 README의 "~50 videos for each dataset"과 대체로 맞습니다. Env 30은 README의 "~50"과 다릅니다.
- 라이선스: 저장소 `LICENSE`는 Apache-2.0입니다.

## 4. 구현 가능성 판정

**결론: 이 맥(Apple M4 Pro, 24 GiB, MPS, CUDA 없음)에서 논문 실험을 재현할 수 없습니다. 공개 코드는 CUDA 기기와 외부 저장소(cosmos-predict2, Isaac-GR00T, videophy)를 전제합니다. 데이터와 가중치도 대부분 공개되지 않았습니다.**

| 구성 요소 | 필요 자원 (추정) | 이 맥에서 | 근거 |
|---|---|---|---|
| WAN2.1 / Cosmos-Predict2 14B LoRA 미세조정 | CUDA GPU 여러 장, 수십 시간 이상 | 불가 | 모델 크기와 메모리, 논문 App. D 에폭 수 (GR1 75, RoboCasa 100) |
| 영상 생성 (RoboCasa 240k, GR1 300/과제 등) | 논문 L40 1,500장 54시간 (RoboCasa 240k) | 불가 | §5 p.9 |
| IDM 학습 (SigLIP-2 large + DiT 8층) | CUDA GPU, 수 시간 이상 | 불가 (MPS에서도 시간 과다) | `base.yaml` 크기 |
| GR00T N1 정책 학습 (GR1, Franka, SO-100) | CUDA GPU, 20k 스텝 (README) | 불가 | README §4 |
| RoboCasa 시뮬레이션 평가 | 시뮬레이터 설치, GPU 추론 | 불가 (설치와 원 데이터 필요) | §3.1 |
| 실로봇 실험 (GR1, Franka, SO-100) | 해당 로봇과 장비 | 불가 | 하드웨어 |
| DreamGen Bench 재평가 | Qwen2.5-VL-7B (fp16 약 15 GB), 생성 영상 4×7k | 이론상 메모리 가능, 시간 과다, 영상 미공개 | D17 |
| 표 숫자 재계산 | CPU | **가능 (03에서 수행)** | 논문 표 값 |
| 소형 toy (IDM 창 평균, 상관 계산) | CPU, 수 분 | 가능 | 목적이 제한적일 때만 의미 |

- **노력 추정:** 공개 코드 전체 재현은 CUDA 환경 설정(cosmos-predict2, Isaac-GR00T, videophy) 1–2주, 학습·생성·평가에 GPU 수백 시간 이상입니다. 실로봇 재현은 하드웨어 확보가 먼저 필요합니다. 맥 로컬에서 할 수 있는 것은 수치 점검과 평가 절차 점검입니다.
- 이 판정은 공개 저장소 구조와 README, 논문 App.를 기준으로 했습니다. 실행해 보지 않았습니다.

## 5. 대조의 한계
- 정적 읽기만 했습니다. 실제 실행 결과와 맞는지 확인하지 않았습니다.
- 커밋 `ec3881d`(2025-10-23)는 GitHub 저장소의 현재 기본 브랜치 HEAD입니다. 논문 실험에 쓴 커밋은 알 수 없습니다.
- `videophy`와 `cosmos-predict2` 저장소는 대조하지 않았습니다.

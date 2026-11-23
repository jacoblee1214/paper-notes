# 04. 논문과 공식 코드의 대조

대상은 `github.com/Kenn3o3/Vista` 커밋 `a0381fb1f95498dda9b006344bbbe923d42b5ccc`(2026-10-05 16:03Z)입니다. 얕은 복제(127 MB)를 정적으로만 읽었고, 실행, 가져오기(import), 설치는 하지 않았습니다. 복제한 내용은 신뢰할 수 없는 데이터로 취급했습니다. 실로봇 저장소 `Kenn3o3/vt_franka`는 복제하지 않고 설정 파일 몇 개를 `gh api`로 읽었습니다. 경로는 저장소 루트 기준입니다. 번호 D는 논문과 코드가 다르거나 논문에 없는 항목, M은 일치 항목입니다.

## 1. 저장소 구성

| 경로 | 내용 |
|---|---|
| `policy/VISTA/vista/` | 논문 모델. `model/equi/vista_sphere_policy.py`(구면 올림, 융합, 보정, 합성곱), `spherical_attention.py`(교차 주의), `vista_image_to_sphere.py`, `vista_tactile_to_sphere.py`, `vista_util.py`(S2Conv, SO3Conv), `equi_group_sampling.py`(I60, C8 표본화), `policy/vista_so3_policy.py`, `vista_so2_policy.py`(손실, 추론), `workspace/train_vista_workspace.py`(학습 루프) |
| `policy/VISTA/vista/model/equi/vista_*concate*, *early_fusion*, *global_l0*` | 절제 변형 (so2, so3 각각) |
| `policy/ISP`, `policy/DP`, `policy/ACT`, `policy/ViTAL` | 기준선 (ISP, ManiFeel과 EquiDiff+Tac, UniVTAC ACT, VITAL) |
| `configs/vista/*.yaml`, `configs/inference/*.yaml`, `configs/experiments/MIDFOV_EXPERIMENT.yaml` | 실험 설정. 추론 프로필은 실행 horizon과 스케줄러를 정함 |
| `envs/`, `assets/`, `third_party/TacEx` | UniVTAC 계열 Isaac Sim 과제 8개와 촉각 시뮬레이터 |
| `scripts/`, `univtac/` | 데이터 수집, 준비, 학습, 평가 진입점 |

공개되지 않은 것: 학습된 체크포인트(README에 링크 없음), 실로봇 데이터와 체크포인트, 논문 표를 만든 로그. 시뮬 시연 데이터는 Hugging Face와 ModelScope에 있습니다(`02` §2).

## 2. 일치 항목 (M)

| 번호 | 논문 | 코드 |
|---|---|---|
| M1 | Ho = 2, Hp = 16, lmax = 6, 행동 10차원, 관측 시퀀스 (부록 A.1) | `config/train_vista.yaml`: `n_obs_steps: 2`, `horizon: 16`, `lmax: 6`, `shape_meta action [10]` |
| M2 | 손실은 노이즈 예측 MSE (식 5) | `vista_so3_policy.py` `compute_loss`: DDPM, `prediction_type: epsilon`, MSE |
| M3 | 군 I60(원소 60)과 C8(원소 8), 군 축을 배치로 합침, 같은 U-Net 공유 (A.5) | `equi_group_sampling.py`의 `SO3ToIco`, `SO3ToC8`, `vista_so3_policy.py` `N=60`, so2 `N=8` |
| M4 | 엔드이펙터 자세로 조화 계수를 회전 (식 D^ℓ(R_t)c) (§3.2, A.4) | `vista_sphere_policy.py` 190행 `irreps.D_from_quaternion(eef_quat)` |
| M5 | 융합 f_i = LN(v_i + ψ([v_i, m_i])), 시각 질의, 촉각 키·값 (식 2–4) | `spherical_attention.py` `SphericalCrossAttention.forward` (다중 헤드 8×64, 출력 투영, GELU MLP, LayerNorm) |
| M6 | 촉각 투영은 시각식 초기화 없이 처음부터 학습, 위치별 softmax (식 8–9) | `vista_tactile_to_sphere.py` 45–46, 65행: `proj_logits ~ N(0, 0.02²)`, `softmax(dim=0)` |
| M7 | 시각 투영은 가우시안 커널 초기화 (식 6–7) | `vista_image_to_sphere.py` 19, 42–54행 (σ = 0.2, 합 1로 정규화) |
| M8 | 구적 가중치는 균일, √P로 정규화 (A.2) | `vista_image_to_sphere.py` `harmonic_norm="sqrt_n"`, `omega = 1/√P` |
| M9 | 구는 HEALPix 격자를 앞쪽 반구로 제한 (A.2) | `vista_util.py` `s2_healpix_grid`, `rec_level: 3`, `max_beta = π/2` (P는 논문에 없고 코드에서 368 또는 400, `03` T5) |
| M10 | 행동의 이동과 두 회전축은 벡터, 그리퍼는 스칼라 (A.5) | `equi_group_sampling.py` `traj_irreps`: `1x0e + 3x1e` |
| M11 | 추론 지연 측정용 DDIM 8단계 (A.9) | `configs/inference/vista_exec8_ddim8.yaml` (`scheduler: ddim`, `num_inference_steps: 8`) |
| M12 | 시뮬 평가는 40회, 시드 1000–1039 (A.7) | README: `--total-num 40 --start-seed 1000` |
| M13 | 학습/검증 8:2, 검증 확산 손실 최저 체크포인트 (A.7) | `get_val_mask`(앞 80% 학습, 뒤 20% 검증), `checkpoint.topk monitor_key: val_loss, k: 1` |
| M14 | 두 이미지를 7×7 특징으로 맞춤 (A.1) | `vista_sphere_policy.py` 153행 `adaptive_avg_pool2d(fmap, (7, 7))` |
| M15 | 관측 출력 [B,Ho,128,455], 행동 [B,Hp,64,455] (A.1, A.5) | `enc_n_hidden: 128`, `act_emb_dim=64`, `SO3Conv(..., lmax)` |
| M16 | 고정 군 정규화 층 (GroupNorm 계열) (A.5) | `equi_conditional_unet1d.py`가 `ConditionalUnet1D`를 군 원소별로 공유 적용하고, 정규화는 `conv1d_components.py`의 GroupNorm (`n_groups: 8`) |

## 3. 차이와 논문에 없는 항목 (D)

| 번호 | 논문 | 코드 | 판단 |
|---|---|---|---|
| D1 | 과제별 군(표 5): HDMI·Hole·Tube는 I60, 나머지는 C8. 표 10 캡션은 "VISTA는 학습 전에 과제 기하로 고른 과제별 인스턴스"라고 함 | 과제와 군의 대응표가 없습니다. 학습 CLI는 `so3`/`so2`를 인자로 받고, README 예시는 Lift Can을 `so3`로 학습합니다(표 5는 Lift Can을 C8로 적음). 선택 절차(검증 손실, 사전 판단 등)는 코드에 없음 | **논문에만 있는 설정.** 같은 과제를 바꾸어 돌려야 표 1을 재현. 실로봇의 군도 논문에 없음 |
| D2 | 작업 공간 중심 c_ws = (0, 0, 0.8) 고정 (A.1) | `ws_center_source: auto` (`config/task/univtac_vista.yaml` 39행): 과제 데이터의 엔드이펙터 위치 최소와 최대의 중점을 중심으로 씀. 검증 에피소드도 포함해 계산(`univtac_replay_image_dataset.py` 435행, `common/univtac_util.py` 25행). (0, 0, 0.8)은 `manual` 모드의 값이고 정책 버퍼 초기값 | **다름.** 논문은 고정, 코드 기본은 과제별 데이터 중점. 대칭 중심이 달라져 학습된 정책의 등변 중심이 과제마다 다름 (`03` T4) |
| D3 | 위치에 "중심 이동"만 언급 (A.1) | 위치 정규화는 세 축에 **같은 크기의 배율**을 씁니다(`abs_action_only_symmetric_normalizer_from_stat`, `get_range_symmetric_normalizer_from_stat`). so2는 xy에만 같은 배율 | **논문에 없음.** 축별 배율은 SO(3) 동변성을 깨므로 필요한 처리이며 코드가 이를 지킴. 논문 부록에 적히지 않음 |
| D4 | "과제당 50 시연" (표 1, 본문) | `n_demo`개의 첫 에피소드를 읽은 뒤 **앞 80% 학습, 뒤 20% 검증**으로 나눔(`sampler.py` 55–67행). n_demo = 50이면 학습 40, 검증 10 | 일관됨(A.7은 8:2를 적음). 다만 "50 시연"에서 경사 갱신에 쓰이는 것은 40개. 기준선도 같은 분할(`univtac_tact_lr.yaml` `val_ratio: 0.2`) |
| D5 | 에폭, 배치, 학습률, 옵티마이저는 논문에 없음 | VISTA: 300 에폭, 배치 32, AdamW lr 1e-4 (β 0.95/0.999, wd 1e-6), 코사인 + 워밍업 500, EMA, DDPM 100단계(`squaredcos_cap_v2`, `clip_sample: True`), 체크포인트 후보는 `1000/n_demo` 에폭마다(50 시연이면 20 에폭마다 15개, 100이면 10 에폭마다 30개) | **논문에 없음.** 시연 수에 따라 후보 체크포인트 수가 달라짐 (`03` §4의 비단조 칸과 함께 볼 것) |
| D6 | "모든 방법이 같은 시연, 행동 표현, 평가 프로토콜을 공유"(A.7) | 배치 크기가 방법마다 다름: VISTA 32, ISP 64(`train_isp_so3.yaml`), EquiDiff+Tac 16(`train_diffusion_unet_equidiff_tact.yaml`). EquiDiff+Tac은 손목 이미지를 230×230으로 키워 224로 자르고, 좌우 촉각 둘을 입력. VISTA는 84×84, 자르기 없음, 왼쪽 촉각 하나. VISTA, ISP, EquiDiff+Tac의 에폭은 모두 300 | **부분적으로 논문에 있음**(촉각 수는 A.7에 명시). 해상도와 배치는 논문에 없음 |
| D7 | 검증 손실로 체크포인트를 고름 (A.7) | 검증 손실은 `self.model`(EMA 아님)로 계산하고, EMA 모델은 따로 `policy`로 잡아 표본화 평가에만 씀(`train_vista_workspace.py` 255–280행). 추론은 EMA 가중치를 불러옴(README) | 선택은 비EMA 검증 손실, 배포는 EMA. 확산 손실은 호출마다 무작위 노이즈와 시간 하나라서 10개 검증 시연의 손실이 거친 추정 |
| D8 | 융합 대안 변형 표 (표 4): 연결은 "channel-wise concat + linear" | 연결 변형은 `LayerNorm → Linear(2048→1024) → ReLU`(`vista_concate_sphere_policy.py`). VISTA 융합 블록은 8헤드 교차 주의 + 출력 투영 + MLP + LayerNorm. 정적 계산한 매개변수 수는 교차 주의 블록 약 5.25 M, 연결 블록 약 2.10 M (비 2.5배, `03` T5) | 절제 변형 사이에 융합 모듈 용량이 다름. 논문은 "거의 같은 파라미터 수"를 C8 대 I60에서만 말함 |
| D9 | 시각 투영 가중치는 가우시안으로 **초기화**(식 6) | 초기화 뒤 `nn.Parameter`로 자유롭게 학습하고 다시 정규화하지 않음(`vista_image_to_sphere.py` 54행). 촉각 쪽은 매 순전파에서 위치별 softmax | 서술과 일치하되, 시각 투영은 정규화 없는 가중치로 학습됨 |
| D10 | 추론 샘플러: 지연 표만 DDIM-8 (A.9). 시뮬 주 결과의 샘플러는 안 적음 | 학습 설정은 DDPM 100단계(`num_inference_steps: 100`). README가 주 프로필로 `vista_exec8_ddim8`(실행 8, DDIM 8단계)을 지정. 기준선 평가 프로필은 `isp_exec8_ddim8`, `temporal_agg`(실행 1) 등 방법마다 다를 수 있음 | **주 결과의 샘플러와 기준선별 평가 프로필은 논문에 없음.** ACT 계열 기준선의 실행 방식이 VISTA와 같은지 확인 필요 |
| D11 | 구면 격자 P, 구면 투영 보조 파라미터는 안 적음 | `rec_level: 3`, 반구, `lmax: 6`, 시각/촉각/융합 차원 1024, `so3_dim 64`, 헤드 8 × 64 | 논문에 없음 (코드가 정함) |
| D12 | 실로봇: 삽입은 6행동, 지우기는 8행동 실행 (A.8, §5) | 실로봇 저장소 `config/inference/vista_paper.yaml`의 `exe_horizon: 8`(과제 `pencil_insertion`), `usb_insertion_visuotactile.yaml`과 `pencil_insertion_visuotactile.yaml`은 `exe_horizon: 16`, `erasing_visuotactile.yaml`은 8. 초기 자세 무작위는 USB 설정에 ±0.01 m (논문 ±1 cm와 일치) | **표 2를 만든 설정이 어느 파일인지 확인할 수 없음.** 파일 사이 horizon이 6, 8, 16으로 갈림. 읽은 파일은 `gh api`로만 확인 |
| D13 | 실로봇 정책의 군(I60 또는 C8)은 안 적음 | 정책 설정 `vista_so2.yaml`, `vista_so3.yaml`이 둘 다 있고 기본 과제는 `pencil_insertion` | 논문에 없음 |
| D14 | 실험은 표 1–10 | 실로봇 저장소에는 표에 없는 설정도 있음: `erasing_pos_id`, `erasing_pos_ood`, `erasing_pos_far_ood`, yaw 이동 개입(`--workspace-yaw-shift-deg`) | 논문에 없는 추가 평가 설정이 코드에 있음. 결과는 논문에 없음 |

**참고(코드의 경고).** `deploy_policy.py` 226행은 실행 horizon이 1보다 크면 "열린 루프 청크 실행, horizon=1 권장"이라고 출력합니다. 논문의 실행 설정(6–8행동)은 이 권장과 다릅니다. 논문이 §5에서 직접 한계로 인정한 부분입니다.

## 4. 구현 가능성 판정 (이 맥: Apple M4 Pro, 약 24 GiB, MPS, CUDA 없음)

**판정: 논문의 시뮬과 실로봇 결과를 이 맥에서 재현하는 것은 불가능합니다. 모델 구조를 직접 다시 짠 소형 CPU 실험은 조건부로 가능합니다. 식 수준 검증(동변성 토이)과 표 재계산은 이미 끝냈습니다.**

| 수준 | 판정 | 근거 |
|---|---|---|
| 시뮬 학습과 평가 (표 1, 9, 10) | **불가** | README가 "Linux with an NVIDIA GPU is required"라고 적고 Isaac Sim 4.5.0, Isaac Lab v2.1.1, cuRobo, libuipc(vcpkg 빌드), PyTorch 2.5.1 cu124, pytorch3d(cu121 휠)를 요구. 이 맥에는 CUDA와 Isaac Sim이 없음 |
| 시연 데이터 받기 | 조건부(승인) | Hugging Face 데이터셋 **124.8 GB**(과제당 4.0–24.4 GB, 100 에피소드). 받지 않음. 승인과 디스크 확인 필요 |
| 공식 체크포인트로 추론 | **불가** | 공개 체크포인트가 없음. 또 오프라인 추론 진입점도 같은 의존성(e3nn 포크, escnn 등)을 설치해야 함 |
| 오프라인 학습(시뮬 없이 검증 손실만) | 조건부, 사실상 어려움 | 데이터 받기와 설치(e3nn 포크, escnn, healpy, diffusers) 승인 필요. 등변 ResNet-50과 SO(3) 합성곱을 MPS에서 쓸 수 있는지 확인하지 못함. 300 에폭 속도는 측정 전. 롤아웃 없이는 성공률을 못 냄 |
| 실로봇 (표 2) | **불가** | Franka, Orbbec 카메라, GelSight Mini, 컨트롤러 PC 두 대와 원격조작 장비 |
| 구조의 소형 재구현 (구면 융합, 동변성 검사) | **가능** | torch 2.15는 설치되어 있어 새 설치 없이 가능한 부분이 있음(구면 조화 ℓ ≤ 2 직접 구현). 동변성 토이는 완료(`03` §6). 학습 가능한 소형 정책은 합성 데이터로 가능하나 e3nn 없이는 근사 |
| 정적 대조, 표 재계산 | 가능, 완료 | 이 문서와 `03` |

**노력 추정 (측정 전, 추정).** 소형 torch 재구현(합성 촉각-시각 토이 과제, ℓ ≤ 2, 군 C8 한정) 1–2주. 공식 코드로 한 과제 학습만 해 보려면 NVIDIA GPU가 있는 Linux에서 설치 2–4일, 데이터 4–24 GB 받기, 학습 시간은 이 환경에서 측정되지 않음. 전체 표 재현은 8과제 × (50, 100 시연) × 기준선 4종 × 학습이라 수 주 이상으로 어림합니다(추정).

## 5. 라이선스와 이용 시 주의

- 저장소 루트는 Apache-2.0(`LICENSE`)이지만 `THIRD_PARTY_NOTICES.md`가 "ISP에서 가져온 부분(`policy/ISP/LICENSE`, `policy/VISTA/LICENSE`)은 학술·교육 한정 라이선스이며 상업 이용은 서면 허가가 필요하다"고 적습니다. 즉 **VISTA 모델 코드의 일부는 상업 이용이 제한**됩니다. 같은 저장소의 다른 부분(ViTAL은 MIT, Image2Sphere는 MIT, TacEx와 libuipc는 각자 라이선스)도 별도입니다.
- e3nn은 BoceHu 포크의 고정 커밋(`b02616237eb936ec8586fe7d803c4a3de2588c09`)을 외부 의존성으로 씁니다.
- 체크포인트는 pickle 객체를 포함하므로 신뢰하는 것만 불러오라고 README가 경고합니다.
- 이 폴더에서 실행하거나 설치하지 않았습니다.

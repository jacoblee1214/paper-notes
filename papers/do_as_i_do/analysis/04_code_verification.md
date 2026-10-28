# 04. 코드 검증: 논문 ↔ 구현 대조

- **대상:** `malik-group/do-as-i-do` @ `824591b`, 서브모듈 `malik-group/Fast-SAM3D` @ `823d478`, `malik-group/sam-3d-objects` (shallow clone)
- **범위:** GPU 실행 없이 코드 읽기와 레포에 포함된 결과 파일 분석만 했습니다. 재현 실험은 하지 않았습니다.

## 1. 모듈 매핑

| 논문 | 구현 위치 | 일치 여부 |
|---|---|---|
| Eq. 1 shape/pose blending | `Fast-SAM3D/track_object.py:796-818` (`batched_guided_sample_sparse_structure`) | ✅ 식과 정확히 일치합니다. `z_ref = (1-t)·noise + t·target`, `x = (1-α)·x_next + α·z_ref` |
| α_s ∈ [0.9, 1] | `run_pipeline.sh`: `--guidance_strength 1` | ✅ |
| 적응형 α_p = max(0.1, 0.7 − 0.09\|Δθ\|) | `track_object.py:1676` lambda, `--pg_floor/--pg_ceil/--pg_slope` | ✅ 단위는 **deg/frame**입니다(논문에는 단위가 없음) |
| BootsTAPIR 20점 + SVD 2D 강체 | `reconstruction/scripts/tapir_velocity_tracking.py` (`--num-points 20`, `np.linalg.svd`) | ✅ |
| D_p = 13 | `_POSE_LATENT_KEYS` = 6drotation_normalized(6) + translation(3) + scale(3) + translation_scale(1) | ✅ 논문 부록 수치와 일치합니다 |
| N = 25 후보, T = 25 Euler | `--num_pose_samples 25 --euler_steps 25` | ✅ |
| 가중 SE(3) 거리로 클러스터링 | `_pose_distance`, `cluster_pose_candidates` (average linkage, `dist_thresh 0.3`, `min_size 3`, `w_rot 1.5`) | ⚠️ 아래 D2 참조 |
| 로그우도 선택 (Eq. 2) | `pose_log_likelihood.py`, `--pose_selection pose_log_likelihood` | ✅ 비교용으로 구현되어 있습니다 |
| 손-물체 정렬 최소제곱 | `reconstruction/scripts/optimize_translation_scale.py` | ✅ 구조 일치(세부 줄 단위 대조는 생략) |
| Warmup H 스텝 + weld | `retargeting/utils/io.py:101-120` (프레임 0 복제를 앞에 추가), `config.py` `warmup_*`, `mjwp.set_weld_active` | ✅ 다만 본문에 없는 부가 장치가 있습니다(D4) |
| Random force perturbation | `sampling.py:make_rollout_fn` (N×K worlds, K = 4, 평균) | ✅ 다만 gate가 있습니다(D5) |
| Transition reward | `mjwp._rest_surface_penalty` (in-hand 구간에 손 접촉 없음 / rest 구간에 표면 접촉 없음 → 상수 페널티) | ✅ 의미는 같습니다. 다만 공개 설정은 **pedestal** 버전입니다(D6) |
| Table 4 하이퍼파라미터 | `config/override/do_as_i_do.yaml` | ✅ 수치가 전부 일치합니다(num_samples 1024, sim_dt 0.005, ctrl_dt 0.5, horizon 3.0, knot_dt 0.2, 노이즈 스케일, 보상 스케일, 교란 파라미터) |

## 2. 불일치와 논문에 적히지 않은 구현 요소

| # | 내용 | 영향 |
|---|---|---|
| D1 | **Post-optimization이 기본으로 켜져 있습니다** (`--post_optimize`). 최종 선택된 pose를 SAM 3D의 layout post-optimization(렌더 실루엣 정렬, shape ICP는 끔)으로 한 번 더 다듬습니다. 본문 Method에는 언급이 없습니다. | Table 2 수치에 이 단계가 포함됐는지 논문만 보고는 알 수 없습니다. 추적기 자체의 기여를 해석할 때 주의해야 합니다. |
| D2 | **클러스터 순위 기준이 다릅니다.** 논문은 "작은 클러스터를 버리고 나머지를 silhouette IoU로 순위"라고 설명합니다. 코드는 (클러스터 크기, 평균 점수) 순으로 정렬해서 **가장 큰 클러스터**를 고르고, 그 안에서 render IoU가 가장 높은 후보를 고릅니다. | 실질적으로는 "다수결 → IoU" 순서입니다. 모드를 찾는다는 논리와는 오히려 더 잘 맞습니다. |
| D3 | 회전 거리를 논문은 `2·arccos⟨q_i, q_j⟩`로 썼지만, 코드는 scipy `(r_a⁻¹ r_b).magnitude()`를 씁니다. | 코드가 맞습니다. 논문 식은 절댓값이 빠져 있어서 q와 −q(같은 회전)를 2π만큼 떨어진 것으로 계산합니다. 논문 표기상의 실수입니다. |
| D4 | Warmup에 본문에 없는 장치가 붙어 있습니다. 시작할 때 손가락을 펴고(finger qpos = 0), 손바닥 법선 방향으로 침투가 없어질 때까지 손목을 물린 뒤 10 cm 여유를 더합니다(`warmup_min_clearance 0.1`). 손목·물체와 손가락 레퍼런스를 3 s 동안 보간하고(`warmup_ref_*_interp_duration`), warmup 구간 보상에는 0.1을 곱합니다(`warmup_rew_multiplier`). | 논문은 "grasp 휴리스틱을 쓰지 않는다"고 강조합니다. 그러나 "펼친 손으로 접근"하는 사전 설정은 약한 휴리스틱입니다. Warmup의 +41%p에 이 장치들이 얼마나 기여했는지는 분리되지 않았습니다. |
| D5 | 교란은 in-hand 구간에서만, 그리고 그 경계에서 0.5 s 안쪽으로 깎인(erode) 구간에서만 가해집니다(`perturb_gate_lag_duration`). warmup 중에는 꺼집니다. | 본문에는 "rollout에 무작위 힘"이라고만 적혀 있습니다. gate가 없으면 잡기·놓기 순간에 교란이 들어가서 성능이 떨어질 수 있으므로, gate는 재현할 때 중요한 디테일입니다. |
| D6 | 공개 설정은 `pedestal_penalty_scale: 0.5`, `floor_penalty_scale: 0`입니다. 논문은 "object-floor contact"라고 설명하지만 실제로는 물체 밑에 붙인 pedestal(plate)과의 접촉을 봅니다. | 의미는 같습니다(부록 B의 Object Base와 연결됩니다). |
| D7 | 최적화 temperature는 1.0이고, softmax는 **상위 10% 후보**에만 적용합니다(`_compute_weights_impl`). 논문은 "MPPI-style"이라고만 적었습니다. | `03_derivations.md` §4 참조 |
| D8 | IK 단계(`solve_ik`)는 기본값 `ref_dt=0.02`로 실행되어 `trajectory_kinematic.npz`의 `frequency=50`이 기록됩니다. 실제 레퍼런스는 HaWoR 약 30 FPS이고 물리 단계는 `ref_dt 0.0333`을 씁니다. | 물리 단계 시간축은 맞습니다. IK 단계의 qvel 레퍼런스만 1.67배 크게 계산되지만, `vel_rew_scale = 1e-4`라서 영향은 미미합니다. 메타데이터 버그에 가깝습니다. |
| D9 | Fast-SAM3D의 Taylor cache(`--enable_ss_cache`, stride 3)와 `torch_compile`로 속도를 높였습니다. 일부 Euler 스텝의 velocity는 네트워크로 계산하지 않고 외삽합니다. | 논문은 Fast-SAM3D를 언급하지 않습니다. 정확도가 얼마나 떨어지는지 보고되지 않았습니다. 다만 §1.1의 항등식은 v의 출처와 무관하게 성립합니다. |
| D10 | 추적은 기준 프레임에서 **양방향**(backward, forward)으로 진행합니다. `--chain_on_diffusion`이 켜져 있어서, 다음 프레임 target으로는 post-opt 이전의 확산 출력 pose를 씁니다. | post-opt의 보정이 다음 프레임으로 전파되지 않습니다. 확산 latent 공간 안에서 일관성을 유지하려는 선택으로 보입니다. |

## 3. 레포에 포함된 whisking 결과 수치 확인 (CPU, numpy)

`retargeting/outputs/sharpa/right/whisking/0/`의 `trajectory_mjwp.npz`(실행된 궤적)와 `trajectory_kinematic.npz`(레퍼런스)를 비교했습니다. 레퍼런스 시간축은 30 Hz이고 warmup 3 s만큼 이동시켰습니다. 스크립트는 `analysis/whisk_tracking_check.py`입니다.

| 지표 | 값 | 기준 |
|---|---|---|
| 평균 물체 위치 오차 | **0.038 m** (최대 0.140) | < 0.1 ✅ |
| 평균 물체 회전 오차 | **0.283 rad** (최대 1.387) | < 0.5 ✅ |
| 물체 높이(z) 범위 | 레퍼런스 0.087–0.228 m, 시뮬레이션 0.087–**0.132** m | 성공 기준으로는 잡히지 않는 차이 |
| 반복 횟수 | 16개 제어 스텝 중 10개가 2회 반복에서 조기 종료 | 상한 32 |

**해석:** 데모는 논문의 성공 기준을 통과합니다. 그러나 레퍼런스에서 물체를 약 14 cm 들어 올리는 구간을, 시뮬레이션에서는 약 4.5 cm만 들어 올립니다. "평균 오차 < 10 cm"라는 기준은 이런 차이를 잡아내지 못합니다. 실제로 쓸 때는 구간별 최대 오차나 높이 같은 과제 특화 지표를 함께 봐야 합니다.

## 4. 재현할 때의 요구 사항 (코드 README 기준)

- **복원:** VRAM 32 GB 이상의 GPU, 4개 conda env(sam3, sam3d, hawor, tapnet), HF gated 가중치(facebook/sam-3d-objects, facebook/sam3), MANO 라이선스, SAM 3 클릭 GUI용 X display
- **리타겟팅:** CUDA GPU(MuJoCo Warp), conda env 1개(python 3.12)
- **라이선스 주의:** HaWoR 포크는 CC BY-NC-ND 4.0이고 SAM 계열은 SAM License입니다. 연구용으로는 괜찮지만 상업적 사용과 수정본 재배포에는 제약이 있습니다.

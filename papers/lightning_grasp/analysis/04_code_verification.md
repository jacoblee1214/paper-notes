# 04. 코드 검증: 논문 ↔ 구현 대조, 수치 재계산, 구현 가능성

- **대상:** `zhaohengyin/lightning-grasp` @ `af43818` (2026-06-12, Python 패키지 `lygra`, 파일 42개). Release v1.0.0(2025-11-11)의 커널 바이너리와 자산 zip은 메타데이터만 봤고 받지 않았습니다. 저장소 이슈 16건을 읽었습니다.
- **범위:** GPU 없이 코드 읽기, 설정 대조, CPU 재계산과 toy만 했습니다. CUDA 커널(`lygra/cpp/build/{geometry,lbvh}.so`)은 소스가 공개되지 않아 동작을 확인할 수 없습니다.

## 1. 모듈 매핑

| 논문 | 구현 위치 | 일치 여부 |
|---|---|---|
| Contact field 표본 (Sec. 4.3) | `contact_field.py: generate_contact_field` (`n_iter=80000`) | ✅ 관절 한계 안 균일 표본 |
| 손 표면 패치 분해 (Sec. 4.4) | `mesh_analyzer.py: generate_mesh_group` (링크당 표면점 3,000개, 1 cm·36° 안에서 탐욕적으로 묶음, 패치당 keyvector 6개) | ✅ "확률적 표면 덮기" 그대로 |
| Contact field BVH (Alg. 1) | `contact/accel/lbvh_s2bundle.py` (`box_w=0.01`, `max_leaf_size=256`) → `bvh_s2bundle.py` → `lbvh.so` | ⚠️ box 생성과 Morton 정렬은 Python에 있고, 구성과 질의는 비공개 커널(D1) |
| 물체 접촉 질의, 법선 정렬 θ_hit | `compute_interaction(angle_threshold=0.2)` | ⚠️ 0.2가 각도(rad)인지 내적 문턱인지 커널이라 확인 불가 |
| 물체 전처리 (Sec. 5.1) | `mesh_analyzer.get_support_point_mask(r=0.01, z=0.002, n>5)` | ✅ |
| 물체 배치 (Sec. 5.2) | `pipeline/module/object_placement.py` (canonical box, 손바닥 배치 50%) | ✅ 손바닥 배치 확률은 논문에 없음 |
| 의존 그룹과 접촉 영역 (Sec. 5.3) | `kinematics.get_dependency_sets`, `contact_collection.py` (영역 해상도 500, 최소 5점) | ✅ |
| 블록별 0차 최적화 (Alg. 2) | `batch_grasp_optimizer.py: optimize/optimize_step` (바깥 5 × 변이 6, σ = 5·√(면적/2048)) | ✅ 변이는 배치 방향으로만 병렬이고 변이끼리는 순차 |
| 안정성 점수 J = FSWO (Eq. 1) | `compute_wrench_score_vectorized` (`wrench_alpha=10`), `batched_nnls` (PGD, lr 0.15, 15회 / warm 4회) | ⚠️ D2 |
| "Free lunch" warm start (Sec. 5.4) | `optimize_step`의 `init_sol=nnls_sol` | ✅ `03` §1.4 toy로 효과 재현 |
| 두 점 맞춤 DLS IK (Eq. 7–8) | `kinematics.batch_contact_ik` (`normal_beta=0.01`, regularization 1e-4) | ✅ Eq. 8의 Jacobian 변환도 같음. 재시작·조합은 D5 |
| 2단계 투영 미세 조정 (Sec. 5.5) | `pipeline/module/kinematics.batch_contact_adjustment` (5 × 3스텝, 3스텝마다 투영, 최종 4 mm) | ✅ |
| 후처리 (Sec. 5.6) | `postprocess.py` (빈 관절 무작위 최대 5회), `collision.py` (AABB + GJK, 점-링크 메쉬) | ⚠️ 안정성 재검사 없음(D4) |

## 2. 불일치와 논문에 적히지 않은 구현 요소

| # | 내용 | 영향 |
|---|---|---|
| D1 | **핵심 커널이 비공개 바이너리입니다.** README는 "CUDA C++ 소스는 다음 릴리스에 공개"라고 씁니다. Release에는 Python 3.8–3.10 × CUDA 12.1–12.6용 `.so`(Ubuntu)만 있습니다. `lygra/__init__.py`는 커널을 못 읽으면 `gem`, `lbvh`를 `None`으로 두고, CPU·MPS로 대신 실행할 경로는 없습니다. `.cuda()` 호출도 32곳입니다. | 논문의 핵심인 BVH 구성·질의(Alg. 1), 점 투영, GJK, 점-메쉬 충돌을 검증할 수 없습니다. NVIDIA GPU와 Linux가 아니면 실행할 수 없습니다. |
| D2 | **안정성 점수는 마찰 없는 FSWO만 있습니다.** 토크 가중 α_w = 10 m⁻¹(λ = 100 m⁻²), 점수 = 잔차 노름, 통과 < 0.15(`contact_optimization.py:22`)입니다. 토크 기준점은 pivot 접점입니다. NNLS는 고정 학습률 0.15 PGD로 15회 돕니다. 마찰(GSWO, Eq. 2)과 중력은 없습니다(이슈 #14에서 저자는 "b에 중력을 더하면 된다"고 답함). | 점수가 물체 크기에 따라 달라집니다. 같은 배치라도 30 cm에서는 8%만 통과합니다(`03` §1.2). PGD는 정확히는 통과인 배치의 9–29%를 떨어뜨립니다(`03` §1.3). 마찰 원뿔 기준보다 보수적이면서, 중력은 무시하므로 정반대 방향의 오차도 있습니다. |
| D3 | **손바닥 접점과 손가락 접점의 좌표계가 섞입니다.** 접촉 영역 점은 물체 좌표계(`contact_collection.py:91`이 변환 전 `object_points`에서 모음)입니다. 손바닥 접점은 손 좌표계(`object_placement.py:120`, `robot_contact`)입니다. `batch_grasp_optimizer.py:304, 338`에서 둘을 그대로 이어 붙여 점수를 냅니다. 최종 점수(`:405`)는 손바닥 접점을 빼고 다시 계산합니다. | 손바닥 배치(전체의 50%)의 최적화 목적함수가 올바른 값과 순위 상관 0.25입니다(`03` §1.5 toy). 잘못된 grasp가 통과하지는 않지만, 손바닥에 기대는 power grasp는 손끝만으로 균형이 맞아야 통과합니다. 코드 읽기로 확인했고 실행 검증은 하지 않았습니다. |
| D4 | **실현된 접점은 다시 채점하지 않습니다.** 안정성은 계획한 접점에서만 봅니다. IK가 4 mm 허용 안에서 맞추고 빈 손가락을 무작위로 채운 뒤에는 충돌만 검사합니다(`postprocess.py`). 논문 Sec. 5.6은 "충돌이나 안정성 기준에 걸리면 버린다"고 씁니다. | 실제 손 자세의 접점과 법선이 계획과 다를 수 있습니다. 시뮬레이터에서의 불안정(이슈 #6)과 이어질 수 있습니다. |
| D5 | **IK에 무작위 재시작과 사슬별 조합이 있습니다.** 1단계 IK는 무작위 초기값 10개로 12회 반복합니다(step 0.4). 그다음 접점마다 오차가 가장 작은 재시작을 골라, 그 사슬의 관절만 가져와 합칩니다(`kinematics.py: batch_contact_ik`). 1단계 통과 기준은 오차 제곱합/2 < 0.02²입니다. | 논문에 없는 장치이고, 독립 사슬이라는 가정에 기댑니다. 의존 그룹이 정적 링크로 잘 나뉘어야 맞습니다. |
| D6 | **Contact field를 실행 때마다 다시 만들고, 캐시는 고장 났습니다.** `ContactField.save/load`(`contact_field.py:310, 344`)는 `os`를 import하지 않았고 `path`, `data`가 정의되지 않아 호출하면 실패합니다. 표본 생성은 관절 배치 8만 회를 도는 Python 루프입니다. | Fig. 5의 "단계별 결과를 캐시해 재사용"은 공개판에서 쓸 수 없습니다. 이 준비 시간은 "forward 2–5초"에 들어가지 않습니다. |
| D7 | **접촉 영역과 투영은 이산 점입니다.** 물체 점군은 2,048개이고, 영역마다 500개를 다시 뽑습니다. Alg. 2의 `Project(p, D)`는 그 점들 중 가장 가까운 점(`torch.cdist`)입니다. | 큰 물체에서는 점 간격이 1 cm를 넘어 해상도가 떨어집니다(이슈 #15). |
| D8 | **"조율이 필요 없다"와 달리 손별 설정과 상수가 많습니다.** 손마다 canonical box, 링크별 허용·금지 법선, 충돌 예외 목록이 있습니다(`lygra/robot/*.py`). 상수로는 통과 0.15, α_w 10, σ 배수 5, IK λ와 step, 허용 오차 4 mm, β 1 cm, 질의 문턱 0.2, 패치 허용 1 cm·36°, 손바닥 배치 50%가 있습니다. | 에너지 균형 가중치는 없다는 주장은 맞습니다. 하지만 새 손을 붙일 때 설정이 필요합니다. 이슈 #5, #8, #10이 모두 새 손 모델 설정 문제입니다. |
| D9 | **접촉에 쓰는 링크가 좁습니다.** Allegro, DClaw는 손끝 링크와 손바닥만 씁니다. DClaw 손바닥은 허용 각도가 0이라 사실상 꺼져 있습니다(`has_aligned`가 cos 0 = 1을 넘을 수 없음). Shadow는 원위·중위 마디를 넣었지만 중위 마디 법선은 ±z와 +y를 막아 대부분 빠집니다(이슈 #12). LEAP 설정은 movable link 목록에 `base_link`가 있는데, LEAP URDF의 링크 이름과 맞는지는 자산을 받지 않아 확인하지 못했습니다. | Fig. 1의 "Diverse Contact ✓"는 손 전체 표면이 아니라 손끝 링크 표면 위의 다양한 점·법선을 뜻합니다. |
| D10 | **벤치마크·시간 측정 코드가 없습니다.** `demo.py`는 forward를 한 번 돌리고 해의 개수만 출력합니다. | Table 1과 Fig. 10을 저장소로 재현할 방법이 없습니다. |
| D11 | **작은 버그와 쓰지 않는 코드가 있습니다.** `contact_collection.py:79`는 `sys.exit`를 쓰는데 `sys`를 import하지 않습니다(탐색 실패 경로에서 NameError). `compute_wrench_score`는 `batched_nnls`가 받지 않는 `x_placeholder` 인자를 넘기는데(`:112`), 호출되지 않는 함수입니다. `concentration_score`도 쓰이지 않습니다. | 기본 경로에는 영향이 없습니다. 실패 처리가 깨져 있습니다. |
| D12 | **라이선스 표기가 둘입니다.** LICENSE와 README는 CC BY-NC 4.0이고 "소프트웨어와 생성 데이터"를 연구용으로 한정합니다. 소스 파일 머리말은 "All rights reserved"입니다. | 상업적 사용(생성한 grasp 데이터 포함)에는 별도 라이선스가 필요하다고 README가 명시합니다. |
| D13 | **mimic joint 지원(2026-04)은 실험 기능입니다.** README가 A100·RTX 4090에서 10–200 SPS라고 적습니다. torch 2.11에서는 GPU 메모리가 해제되지 않아 2.7로 낮추라고 안내합니다. | 커플링 관절이 있는 손(예: 사람형 손 다수)에서는 Table 1 속도를 기대하기 어렵습니다. |

## 3. 수치 재계산 (`analysis/table_check.py`, `fswo_toy.py`, `contact_field_toy.py`)

| # | 주장 (출처) | 재계산 | 판정 |
|---|---|---|---|
| C1 | Table 1 trimmed µ = 최솟값·최댓값 제외 평균 | LEAP 420.25, Shadow 558.78, DClaw 249.22는 맞음. **Allegro는 1149.42인데 1090.8** (Capsule + Apple을 뺀 값) | Allegro 행 ✗ |
| C2 | forward 한 번에 1,000–10,000개, 2–5초 (Sec. 1). 모든 설정 6초 안 (Table 1) | SPS × 6 < 1,000인 칸이 5개 (LEAP–Scissors 832, DClaw–Spoon 953·Cup 829·Scissors 757·Screwdriver 927) | ⚠️ 모든 경우는 아님 (SPS가 amortized라면 직접 비교 불가) |
| C3 | 기존 방법보다 수 자릿수 빠름 (Abstract, Fig. 1) | BODex 대비 SPS 6–33×, 시간 20–60×. DexGraspNet 대비 SPS > 100×, 시간 360–1000×. SpringGrasp 대비 SPS > 100×, 시간 2–20× | BODex 대비로는 한 자릿수 |
| C4 | TITAN X가 A100 위 기존 방법보다 20–100× (Sec. 6.3), "수백 배" (Fig. 10) | 626.1 SPS → BODex 대비 12.5–20.9×, DexGraspNet·SpringGrasp 대비 > 208× | 기준선에 따라 맞거나 틀림 |
| C5 | Fig. 10 단계 비율, 접촉·운동학 각 약 33% | 합 99.7–100.0%. 접촉 27.2–43.4%, 운동학 I+II 33.6–38.0%. A100 1578.0 ≈ Table 1의 1578.8 | ✅ |
| C6 | 메모리: box 3,000 × 256 × 16 B ≈ 12 MB, 100개 1.2 GB, 법선 2바이트 압축 8× (Sec. 4.4) | 12.29 MB + 0.29 MB, 1.26 GB, 8× | ✅ 산술 |
| C7 | 256방향이면 원뿔 반각 약 7° (Sec. 4.3) | arccos(1 − 2/256) = 7.17° | ✅ |
| V1 | "대부분 box는 50개 이하" (Sec. 4.4) | 코드 설정(80,000 × 6) toy: 78–82%가 50 초과, 42–53%가 256 초과 | ✗ 코드 설정에서는 성립 안 함 (`03` §2) |
| V2 | warm start로 내부 반복을 크게 줄인다 (Sec. 5.4) | warm 4회 과대추정 0.015–0.017 < cold 15회 0.031 | ✅ |
| V3 | FSWO가 안정성 기준 (Eq. 1) | 같은 배치를 4 → 30 cm로 키우면 통과 100% → 8%. PGD 15회는 정답 통과의 9–29%를 떨어뜨리고, 오통과는 0 | ⚠️ 크기 의존, 보수적 |

## 4. 구현 가능성 판단

### 4.1 공식 코드를 그대로 돌리는 경우

| 항목 | 요구 사항 | 비고 |
|---|---|---|
| OS | Ubuntu 22.04 / 24.04 (README) | 커널 바이너리가 Linux용 `.so` |
| GPU | NVIDIA Pascal–Ada Lovelace, CUDA 12 | RTX 50 시리즈는 이슈 #7에서 별도 대응. TITAN X에서도 626 SPS(Fig. 10) |
| Python, torch | 3.8–3.10. torch 2.4.1(py38) 또는 2.8.0(py39) conda 환경. 2.11은 메모리 누수 | 그 밖의 버전은 이슈로 요청 |
| 자산 | Release의 `assets.zip`(손·물체 44 MB)과 커널 zip(약 64 MB) | 저장소 자체는 6 MB |
| 시간 | 물체 하나에 몇 초. 준비 단계(contact field 구성)는 실행마다 다시 돎 | D6 |
| 라이선스 | CC BY-NC 4.0, 생성 데이터 포함 연구용 | 상업적 사용은 별도 라이선스 |

**판단:** NVIDIA GPU가 있는 Linux 머신이면 **그대로 돌릴 수 있습니다.** 요구 사양이 낮은 편이라(구형 GPU도 됨) 재현 장벽이 낮습니다. 다만 Table 1 숫자를 재려면 시간 측정 코드를 직접 붙여야 합니다(D10). 결과를 쓸 때는 시뮬레이터 검증을 따로 해야 합니다(D2, D4, 이슈 #6).

### 4.2 이 맥(Apple M4 Pro GPU, CUDA 없음)에서

| 단계 | 가능 여부 | 근거 |
|---|---|---|
| 점수식·contact field 밀도·표 수치 toy | ✅ 완료 | numpy·scipy, 30초 안 |
| 공식 코드 실행 | ❌ | CUDA 전용 바이너리, `.cuda()` 32곳, CPU·MPS 대체 경로 없음 |
| 순수 Python 부분(FK·IK·NNLS 점수)을 CPU torch나 MPS로 이식 | 조건부 | torch 설치가 필요합니다(설치 전 승인). 커널이 하던 BVH 질의, 투영, 충돌은 다시 구현해야 합니다 |
| 공식 grasp 결과로 MuJoCo 흔들기 검증 | 조건부 | MuJoCo는 CPU로 돌지만, grasp를 만들려면 GPU 머신이 먼저 필요합니다 |

### 4.3 아이디어만 다시 구현하는 경우

| 구성 요소 | 난이도 | 메모 |
|---|---|---|
| contact field 표본(FK로 손 표면 벡터 모으기) | 낮음 | 몇십 줄입니다. 표본 수는 box당 밀도로 정하면 됩니다(`03` §2: 8만 개는 과다) |
| BVH 구성과 (위치, 법선) 질의 | 중간 | 공개 BVH 라이브러리(Warp, Open3D 등)나 KD-tree에 법선 검사를 더하면 됩니다. GPU 처리량을 맞추는 것이 관건입니다 |
| 접점 0차 탐색 + NNLS 점수 | 낮음 | `fswo_toy.py`의 점수식 그대로입니다. 크기 정규화(α_w = 1/R)와 마찰 원뿔을 넣는 것을 권합니다 |
| 두 점 맞춤 DLS IK | 낮음–중간 | 미분 가능한 FK 라이브러리가 있으면 쉽습니다. 무작위 재시작과 사슬별 조합(D5)까지 넣어야 수율이 나옵니다 |
| 충돌 필터 | 중간 | 볼록 분해 + GJK(예: FCL 계열)나 MuJoCo 충돌로 대신할 수 있습니다 |
| 물리 검증(흔들기, 중력) | 중간 | 논문에 없는 단계지만, 데이터로 쓰려면 꼭 필요합니다 |

**요약:** 공식 코드는 NVIDIA Linux 머신에서 바로 쓸 수 있는 수준이고, 이 맥에서는 돌지 않습니다. 아이디어 자체는 단순해서 CPU나 MPS로 다시 구현할 수 있습니다. 하지만 논문의 속도는 비공개 CUDA 커널에서 나오므로, 재구현판은 기능 검증용이 될 것입니다. 어느 쪽이든 점수식의 크기 의존성(D2), 손바닥 좌표계(D3), 실현 접점 재채점(D4)을 고치고 시뮬레이터 검증을 붙이는 것이 실제로 쓸 데이터를 얻는 조건입니다.

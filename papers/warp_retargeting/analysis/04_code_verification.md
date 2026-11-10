# WARP 논문과 공개 코드 대조 (정적 읽기)

- **대상 코드:** `Euler-Rodrigues-Lab/WARP_retargeting`, 얕은 clone `repo/WARP_retargeting/` (커밋 `172b1dcde11d3942d3fad6d378695b5703631ded`, 2026-11-08 "Fix CI Python setup and canonical author identity"). 첫 공개 커밋은 `b98b97f` "Initial public WARP retargeting release" (2026-11-08)입니다.
- **라이선스:** 저장소 코드 MIT (LICENSE). 의존 패키지는 각자 라이선스를 따르고, **`geo_kin` 솔버는 별도 라이선스**입니다(README).
- **방법:** 읽기만 했습니다. 설치(`uv sync`), 실행, 테스트는 하지 않았습니다. 설치는 CLAUDE.md 규칙에 따라 목록을 먼저 보고 승인을 받아야 하므로 보류했습니다.
- **실행 기록:** 이 저장소의 `README.md`, `docs/`(SCOPE, WORKFLOWS, PROVENANCE, RUNNING, SAMPLE_VALIDATION), `src/warp_retargeting/`(config, variants, metrics, offline_validate), `experiments/warp_paper/configs/`를 읽었습니다.

## 1. 저장소가 담은 것과 빠진 것

| 구성 | 저장소 상태 | 근거 |
|---|---|---|
| WARP / c-SEW / SEW-Mimic 솔버 소스 | **없음.** `geo_kin` 휠을 따로 받아 등록해야 함 | `docs/PROVENANCE.md`, `docs/SCOPE.md` ("protected solver sources are absent") |
| 설정 프리셋 (`warp`, `c_sew`, `sew_mimic`, `warp_no_spring`, `warp_no_joint_limits`) | 있음 | `experiments/warp_paper/configs/variants.yaml` |
| MINK-EF / MINK-TE 변환 | 있음 (공개 폴백 `geo_kin_core`). **손(XHand) IK는 없음** ("hand outputs unsolved") | `docs/RUNNING.md`, `docs/WORKFLOWS.md` |
| 재생 (MuJoCo, HDF5) | 있음 | `replay_hdf5.py`, `fixtures/retargeted/*.hdf5` |
| 지표 | 일부만: EEF 위치·방향, 몸통 방향, 관절 속도·저크, 자기 충돌 | `metrics.py` (`compute_metrics`) |
| NNAD, PCA/RMS 해 변동, P95/P99, 팔꿈치 오차, Out@15, 관절 한계 비율, 여유 최솟값 | **없음.** 문서에 "global NNAD/PCAV pass는 재현하지 않는다"고 명시 | `docs/WORKFLOWS.md` 마지막 표 |
| 정책 학습 (HPT, flow matching, block-causal 마스크) | **없음.** src, tests, scripts, docs를 검색해 해당 항목을 찾지 못함 | 검색 결과 없음 |
| 정책 롤아웃 클라이언트 | 있음 (websocket, msgpack) | `policy_rollout.py` |
| DexMimicGen 파이프라인, 시뮬 데이터 생성 | 없음 | 논문 C.2 |
| 실물 로봇 코드 | 없음 (`external/rby1_teleop` submodule로 분리) | `docs/SCOPE.md` |
| 데이터 | 샘플 1개 (washing dishes, 420프레임, 30 Hz) + 참조 HDF5 5종 | `fixtures/` |
| BONES-SEED 514 / 7,623 부분집합 목록 | 없음 | — |

## 2. 주장별 판정 (C1–C23)

판정 기호: ✔ 일치, ◐ 조건부 또는 부분 일치, ✘ 불일치, ? 검증 불가. 근거는 `03_analysis.md`(재계산), `paper/paper.txt`(본문), 저장소 파일입니다.

| ID | 논문 주장 (위치) | 판정 | 근거·메모 |
|---|---|---|---|
| C1 | 실물 Fig. 7: 상자 회전 롤아웃 65 대 40, 세탁 재생 70 대 0 (§4.3) | ✔ | 그림 막대 라벨과 일치. 10회 시행 |
| C2 | 냉장고 과제 재생 90% (§4.3) | ? | 본문만 있음. Fig. 7에 없음, 시행 수 10으로 추정 |
| C3 | 서론 "정책 35% 우수" | ◐ | 세 과제 롤아웃 차이의 평균 %p와 일치(35.0). 정의가 적혀 있지 않음 (`03` §2) |
| C4 | 시뮬 정책 +12%p, 재생 동등 (Table 1) | ✔ (평균) / ◐ (유의성) | 평균 재계산 일치. coffee 과제 하나만 Holm 보정 후 유의 (`03` §1). 시드·롤아웃 수 없음 |
| C5 | "whole-body tracking error 99.34% 감소" (서론) | ◐ | 실제 값은 손바닥 **위치** 평균 오차(MINK-EF 대비, JL off). 용어가 부정확 |
| C6 | 손바닥 오차 150배 감소 (§4.1) | ◐ | 514 JL off에서 152배 재현. 7.6k 부분집합의 정상 프레임에서는 MINK-EF가 더 낮음 (`03` §3) |
| C7 | 방향 오차 머신 정밀도 (§4.1) | ◐ | JL off만 8.7e-6°. JL on은 3.26° |
| C8 | "exact palm matching by construction" (식 1, §3 Properties) | ◐ | 이론상 성립. 실험의 전체 시스템(JL on)에서 원시 평균 24 mm. 설명 없음 |
| C9 | 관절 한계·자기 충돌 최저 (§4.1) | ◐ | 514에서만 최저. 7.6k에서 SEW-M이 충돌·한계 모두 낮음 |
| C10 | NNAD, PCA, RMS 최저 "수 자릿수 차이" (§4.1) | ✘ (NNAD) / ◐ (PCA, RMS) | PCA, RMS는 닫힌 형식이라 정의상 0에 가까움. NNAD 차이는 약 20%, 팔 NNAD는 SEW-M이 더 낮음 |
| C11 | 30배 빠름 (§4.1) | ✔ | 192.1 / 6.78 = 28배 (514), 36배 (7.6k) |
| C12 | 프레임당 마이크로초 (§3) | ✘ | Table 3은 6.8 ms (밀리초). 저장소 fixture README도 기록 시간은 벤치마크가 아니라고 함 |
| C13 | "SEED 한 시간 대 하루" (§4.1) | ✘ / ? | 프레임 × 시간으로 계산하면 514에서 15분 대 7.3시간. 조건 미기재 |
| C14 | 기지 필터 파라미터 δ = 0.05 m, δ_θ = 0.1 rad, f_n ≈ 1.5 Hz, ζ = 1 (부록 B.5) | ✔ | `warp` 프리셋: `mobile_base_deadband 0.05`, `yaw 0.1`, `natural_freq 1.5`, `damping 1.0`. `config.py` 기본값과 일치 |
| C15 | "JL off" = 관절 한계 없음 (Table 3 열) | ◐ | 프리셋 `warp_no_joint_limits`는 관절 한계와 스프링 댐퍼 기지를 **함께** 끔. 논문 "JL off" 행은 기지 필터도 꺼진 설정으로 보임 (D5) |
| C16 | 시뮬 실험은 사람 데이터 (서론의 "human demonstrations" 전제) | ✘ | 원천은 DexMimicGen으로 만든 **GR1 로봇 시연** (부록 C.2). 사람 데이터는 실물에만 쓰임 |
| C17 | 실물 두 방법 정책이 "identical" (§1) | ? | 정책 구조와 하이퍼파라미터 비교 조건이 적혀 있지 않음. 공개 코드에 정책 없음 |
| C18 | 사람 영상 없이 Quest 데이터만으로 제로샷 전이 (§1) | ◐ | 정책은 AprilTag 물체 포즈와 로봇 상태를 입력. 이미지 없음(D.1). "제로샷"의 뜻이 좁음 |
| C19 | "first framework ... zero-shot whole-body mobile manipulation" (초록) | ? | 선행 연구 전수 조사 없음. HOMMI [3]와 비교하지 않음 |
| C20 | CoRL 2026 채택 (BibTeX, 프로젝트 페이지, 저장소 README) | ◐ | 세 곳에서 같은 표기. arXiv abs에는 댓글 없음. 학회 목록은 확인하지 못함 |
| C21 | 공식 코드 공개 (BibTeX/README의 링크) | ◐ | 논문과 프로젝트 페이지에는 코드 링크가 없음. 공개 저장소는 있으나 **WARP 솔버 없이 MINK 기준선과 재생만 포함** |
| C22 | JL on의 큰 손바닥 오차는 작업공간 밖 목표 때문 (§4.1) | ? | 분해 실험 없음. 같은 자리에 "몸통 이탈·관절 한계"가 함께 있어 다른 원인도 가능 |
| C23 | 부록 A.2: 어떤 MINK 가중치도 SEW와 세 지표를 동시에 맞추지 못함 | ? | 그림 10만 있음. MINK 스윕 코드와 수치 없음 |

## 3. 코드 대조에서 나온 차이 (D1–D8)

- **D1. MINK-TE 팔꿈치 가중치 불일치.** 논문 Table 2는 팔꿈치 스윙 비용 0.2, 저장소 `variants.yaml`의 `mink_te`는 `elbow_angle_cost: 0.5`입니다. 몸통 방향 가중치 0.5는 두 곳이 같습니다. 어느 값이 실험에 쓰였는지 확인해야 합니다.
- **D2. 솔버 부재.** WARP, c-SEW, SEW-M 결과는 `geo_kin` 휠이 있어야 새로 만들 수 있습니다. 공개 저장소로는 **논문 표 3, 4를 재생성할 수 없습니다.** 제공된 참조 HDF5 5종은 washing dishes 한 동작의 결과물이라 표의 514 집합과 맞지 않습니다.
- **D3. 지표 누락.** 공개 `metrics.py`의 지표는 9개 정도입니다. 논문 표의 P95, P99, 팔꿈치 오차, Out@15, 관절 한계 비율, 여유 최솟값, NNAD, PCA, RMS는 **공개 코드에 없습니다.** 저장소 문서가 이것을 명시합니다. 논문의 일관성 지표 전체를 독립적으로 확인할 방법이 없습니다.
- **D4. 정책 학습 코드 없음.** HPT 기반 흐름 정합, block-causal 마스크, 체크포인트, 학습 설정이 공개 저장소에 없습니다. 실물 정책 재현이 불가능합니다.
- **D5. "JL" 열의 의미.** 프리셋 `warp`는 `enable_joint_limits: true`와 `mobile_base_spring_damper: true`를 함께 켭니다. `warp_no_joint_limits`는 두 가지를 함께 끕니다. 따라서 논문 Table 3·4의 "off" 행(손바닥 0.0046 mm, 충돌 0.163)은 **관절 한계와 lazy base가 모두 빠진 설정**으로 보입니다. 논문은 "JL"을 정의하지 않고 본문에서 "auxiliary constraints 없이"라고만 씁니다. 이 구분이 헤드라인 수치를 크게 바꿉니다.
- **D6. 파이프라인 차이 (시뮬).** WARP 시연은 재생 시 SEW 솔버를 **온라인으로** 돌리고, MINK 시연은 **미리 계산된 관절 목표**를 저장합니다(논문 C.2). 두 방법의 데이터 생성 경로가 같지 않습니다.
- **D7. 손 리타게팅 차이.** 공개 MINK 폴백은 XHand IK가 없어 손 출력이 비어 있습니다. 논문의 MINK 기준선은 보호된 "analytical XHand IK"를 썼다고 저장소 문서가 밝힙니다(`docs/RUNNING.md`). 손 쪽 비교는 재현할 수 없습니다.
- **D8. 충돌 회피 꺼짐.** 모든 프리셋에서 `enable_collision_avoidance: false`입니다. 자기 충돌 지표는 사후 측정이고, IK는 충돌을 피하도록 풀지 않습니다. 논문의 "self-collision avoidance prior" 해석(§4.1)은 이 점을 고려해야 합니다. 충돌 비율 0.163(JL off)은 이 설정에서 나온 값입니다.

## 4. 구현 가능성 판정 (이 맥 기준)

**이 맥:** Apple M4 Pro, 통합 메모리 약 24 GiB, MPS 사용 가능, CUDA 없음, torch 미설치. 설치는 승인 전이므로 하지 않았습니다.

| 항목 | 판정 | 이유 |
|---|---|---|
| **공식 재현 (논문 표 전체)** | **불가** | WARP 솔버는 `geo_kin` 라이선스 휠이 필요합니다. 저자에게 요청하지 않는 한 확보할 수 없습니다. 정책 학습 코드, 데이터 처리, 시뮬 GPU, 실물이 없습니다. |
| 공개 부분 재현 (MINK 기준선, 재생, 공개 지표 4종) | **조건부** | `uv sync`와 `mujoco`, `h5py`, `geo_kin_core` 설치가 필요합니다(승인 필요). 설치 후 headless 재생과 MINK 변환은 CPU로 됩니다(시간은 수 분 규모로 예상, 미검증). |
| c-SEW 아이디어 재구현 (기하 부분) | **중간 난이도** | 식 (1)–(3), SP3, Stereo-SEW, SEW-Mimic은 수백 줄의 numpy입니다. Stereo-SEW와 ik-geo의 원 논문 [24], [25]를 읽어야 세부를 맞출 수 있습니다. 정확도 검증 기준은 손바닥 등식과 해의 유일성 두 가지로 충분합니다. |
| lazy base 필터 | **쉬움** | 2차 시스템과 데드밴드 50줄 이하입니다. 장난감으로 가능 (`03` §4.4). |
| 정책 학습 (HPT류, 11M) | **어려움** | 코드가 없어 처음부터 구현해야 합니다. 맥 MPS에서 작은 규모 학습은 가능할 수 있으나 미검증입니다. 시뮬 Table 1 규모는 GPU가 필요합니다. |
| 시뮬 DexMimicGen | **불가 (현재 조건)** | MuJoCo, robosuite 설치, GPU 권장. 설치 승인 전까지 보류. |
| 실물 RB-Y1, XArm, Quest, Vicon | **불가** | 하드웨어 없음. |

**판정 요약:** 논문의 공식 재현은 **불가**합니다. 공개 코드로는 MINK 기준선과 재생, 일부 지표까지만 확인할 수 있습니다. 아이디어 재구현은 기하 부분이 **중간 난이도**이고 CPU로 충분합니다. 정책 부분은 코드가 없어 **재구현 노력이 크고** GPU 필요성이 큽니다.

## 5. 권리

- **논문:** arXiv abs 페이지 라이선스는 **CC BY-NC-ND 4.0**입니다(CC BY 아님). ND는 변형 게시를 금지하고 NC는 상업 이용을 금지합니다. 따라서 논문 그림과 표는 **개인용 리포트와 PPT에만** 넣고, 공개 산출물에는 넣지 않습니다. 공개물은 직접 만든 그래프와 우리 문장만 씁니다.
- **PDF 1쪽 권리 표기:** arXiv 스탬프(`arXiv:2606.29940v2 [cs.RO] 19 Aug 2026`)만 있습니다. 저작권 문구, 라이선스 문구, `©` 표기는 전체 23쪽에서 찾지 못했습니다(pdftotext 검색). 권리 표기는 abs 페이지 기준으로 판단합니다.
- **공개 저장소 코드:** MIT. 저작권 표기는 저자 개인 계정명입니다. 재사용은 가능하지만, 이 분석에서는 코드를 복사하지 않았습니다.
- **`geo_kin`:** 별도 라이선스. 저장소에 포함되지 않습니다. **재배포 대상이 아닙니다.**
- **`rby1_teleop` (submodule, 공개):** 라이선스 파일이 없습니다. 기본 저작권이 적용되므로 복사하지 않았습니다.
- **MINK:** Apache-2.0 (`kevinzakka/mink`).
- **DexMimicGen:** NOASSERTION. 확인하지 않았습니다.
- **BONES-SEED 데이터:** 접근 조건을 확인하지 않았습니다.

## 6. 빠진 정보와 저자에게 물을 것

1. `geo_kin` 솔버를 연구용으로 받을 수 있는지, 또는 논문 표 재생성용 출력을 공유할 수 있는지.
2. D1: MINK-TE 팔꿈치 가중치 (0.2 또는 0.5).
3. D5: "JL off"가 관절 한계만인지, lazy base까지인지. 전체 WARP 시스템(JL on)의 손바닥·방향 오차.
4. 시뮬 Table 1의 롤아웃 수, 학습 시드, 교집합 학습 시연 수.
5. 실물 MINK 변형(EF 또는 TE), 정책 구조와 하이퍼파라미터 동일 여부.
6. D7: 공개되지 않은 XHand IK를 MINK 기준선에 쓴 방식.
7. 표 4의 정상 프레임 손바닥 오차에서 MINK-EF가 WARP보다 낮은 이유.

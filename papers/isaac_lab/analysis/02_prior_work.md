# 02. 선행연구와 관련 저장소

## 1. 계보

| 단계 | 연구 | 이 논문과의 관계 |
|---|---|---|
| 기반 | PhysX (NVIDIA), Isaac Gym (Makoviychuk 2021) | GPU 텐서를 PyTorch에 노출한 첫 end-to-end GPU RL 파이프라인. Isaac Lab의 직접 선행 (1절) |
| 설계 | Orbit (Mittal 2023) | 원래 프레임워크. Isaac Lab은 Orbit에서 시작했다고 밝힘 (부록 A, Acknowledgments) |
| 대상 | Isaac Sim, Omniverse RTX, OpenUSD | 장면 표현, 렌더링, 물리 파라미터를 가져옴 (2절) |
| 후속 | Newton (NVIDIA, Google DeepMind, Disney Research), Warp, MuJoCo Warp | 미분 가능한 GPU 물리 엔진. 7절의 통합 대상 |

## 2. 같은 문제를 다루는 다른 시뮬레이터 (논문이 인용한 것)

| 시뮬레이터 | 논문 인용 | 이 논문의 비교 |
|---|---|---|
| Isaac Gym | Makoviychuk 2021 | 설계 계보. 처리량 비교 없음 |
| ManiSkill3 | Tao 2025 | GPU 병렬 시뮬레이터로만 언급 (1절) |
| MuJoCo Playground | Zakka 2025 | GPU 병렬 시뮬레이터로만 언급 (1절) |
| MuJoCo | Todorov 2012 | CPU 기준선으로만 언급 (1절) |
| PyBullet | Coumans & Bai 2016–2023 | CPU 기준선으로만 언급 (1절) |
| Gazebo, Unity, Unreal, AirSim | 2.1절 | 형식과 엔진 한계를 논의 |

논문에는 이들과의 처리량 비교나 같은 과제 성공률 비교가 없다. 상대 우위의 근거는 "GPU 가속과 설계 편의"이다(`01` Q14).

## 4. GitHub와 프로젝트 페이지 (조사 날짜 2026-11-22, GitHub API 메타데이터)

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 기준 커밋 |
|---|---|---|---|---|---|
| [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | 공식 코드 (arXiv 댓글과 README가 가리킴) | `LICENSE` 파일은 BSD-3-Clause. `LICENSE-mimic`은 Apache-2.0. Isaac Sim 등 일부 워크플로는 별도 독점 라이선스 (API는 NOASSERTION 판정) | 8,697 | 2026-11-21 | 분석 대상 `v2.3.0` 태그 `3c6e67b` (2025-10-28). 최신 `main`은 v3.0.0 (Isaac Sim 6.2) |
| [isaac-sim.github.io/IsaacLab](https://isaac-sim.github.io/IsaacLab) | 프로젝트 페이지 (문서 사이트, 최신 main 문서로 넘어감) | 문서 | — | — | — |
| [newton-physics/newton](https://github.com/newton-physics/newton) | 7.1절 통합 대상 물리 엔진 | Apache-2.0 | 5,977 | 2026-11-21 | — |
| [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) | 5.1절 RL 라이브러리 | API 판정 불가 (NOASSERTION), LICENSE 파일은 미확인 | 3,151 | 2026-11-20 | — |
| [NVlabs/GraspDataGen](https://github.com/NVlabs/GraspDataGen) | 6.4절 Isaac Lab 기반 파지 샘플링 | API 라이선스 null (미확인) | 40 | 2025-10-03 | — |

- **v2.3.0 얕은 복제:** `isaac_lab/repo/`에 `git clone --depth 1 --branch v2.3.0`으로 받았다. LFS 파일은 받지 않았다(`GIT_LFS_SKIP_SMUDGE=1`). 복제 크기는 약 80 MB다. 정적 읽기만 했다.
- **main 트리 확인:** v3.0.0과 main의 트리 목록(GitHub API 재귀 조회)에서 `source/isaaclab_contrib/isaaclab_contrib/sensors/tacsl_sensor/`를 찾았다. v2.3.0에는 없다(`04` D5).
- **Isaac Lab-Arena:** v3.0.0 트리에서 찾지 못했다. 논문의 "곧 공개" 상태와 맞다.
- **태그 목록:** `v2.3.0`(2025-10-28), `v2.3.1`(2025-12-04), `v2.3.2`(2026-02-02), `v3.0.0`(2026-10-20). 논문 제출(2025-11-06)에 가장 가까운 릴리스는 v2.3.0이다.

## 5. 라이선스와 공개 범위

- **논문 라이선스:** arXiv abs의 "view license"는 CC BY 4.0이다(확인 날짜 2026-11-22). 그러나 원문 PDF 첫 쪽과 끝 쪽에 "© 2025 NVIDIA. All rights reserved."가 있다. 이 고지는 CC BY와 충돌한다. 따라서 **논문 그림은 공개본에 넣지 않는다.** 그림의 출처 표시만 하고, 공개본의 그림은 우리가 그린 그래프 두 장뿐이다.
- **본문 인용:** 수치와 표 값은 출처를 붙여 인용한다.
- **코드:** 공식 코드는 공개본에 넣지 않는다. 정적으로 읽은 내용만 `04`에 기록한다.

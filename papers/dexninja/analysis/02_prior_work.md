# 02. 선행연구와 관련 저장소

## 1. 선행연구 계보 (구성 요소별 출처)

| 구성 요소 | 이 논문이 기대는 것 | 이 논문의 몫 (논문 서술 기준) |
|---|---|---|
| 절단 시뮬레이션 | RoboNinja [8] (MPM 기반 다중 재료 절단과 적응 정책, RSS 2023), DiSECt [44] (미분 가능 절단 엔진, RSS 2021), Taccel [43], Genesis [72], Warp [80] | 변형체 절단 시뮬레이터와 관절 로봇 시뮬레이터를 "칼의 합성 렌치"로 잇는 국소-장면 등록 다리(부록 1.4) |
| real2sim2real 틀 | D-REX [15] (같은 1저자 그룹의 미분 가능 real-to-sim-to-real 파지 엔진, ICLR 2026), Scalable Real2Sim [73], TwinAligner [16] | 음식의 위상 변화 절단과 촉각을 포함 |
| 데이터 증강 | MimicGen [19] (시뮬 궤적 생성), DemoGen [20] (실세계 시연 확장), R2RGen [21], Real2Render2Real [23], GigaWorld-0 [22] | 자세·궤적만이 아니라 질량·기하·마찰(선택적으로 외형)을 바꾸는 "물리 조밀화" |
| 형상 사전과 에셋 | Gaussian splatting 계열 [25–33], Trellis [88] (영상-3D 생성), Robo-GS [31] (복원) | 범주 속성(모양·표면·색·숙성도) 조건 생성 + 2D/3D 판정 + 시뮬 수락 검사(부록 4) |
| 정책 구조 | ACT [83] (행동 청크 트랜스포머), 적응 청크 [85, 86] | 팔·손 분리 머리, 그룹별 행동 정규화, 단계 조건 가변 청크(식 8, 14) |
| 촉각 | ViTacFormer [64], 공간 고정 촉각 인식 [65], DiffTactile [66] | MPM 입자 기반 촉각 합성 + 프레임 일관 표현(부록 1.3) |
| 파지·손 | DexGraspNet 2.0 [87] (칼·손 파지 사전), 원격 조종 장갑·리타겟팅 [75–78] | 칼 사전 파지 초기화 |
| 평가 과제 | 음식 반 가르기 [47] (변형 접촉 인지 MPC, 힘 조절 조작) | 4종 범주 실로봇 25회씩 |

가장 가까운 대안은 (1) RoboNinja 계열(시뮬 안에서 절단 정책을 학습하되 칼을 손에 강체로 고정), (2) MimicGen·DemoGen 계열(시연 변환으로 자세만 증강), (3) 시뮬 없이 많은 실데이터를 모으는 방식입니다. 부록 표 2는 (1)과 비교하고, 표 1은 Real-only(3)와 Sim-only를 비교합니다. (2)와의 직접 비교, 즉 같은 데이터 양의 단순 자세 섭동 증강 대조는 논문에 없습니다.

## 2. 코드와 저장소 (GitHub 표)

조사일 2026-11-26. 별 수와 마지막 push는 GitHub API 값이고, 논문 폴더 안에 clone한 것은 없습니다.

| 저장소·자원 | 관계 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| **공식 코드** | 해당 없음 | - | - | - | 프로젝트 페이지 "Code Coming soon". GitHub 검색(`dexninja`, 저자명)에서 저장소를 찾지 못했습니다. `github.com/Dexninja`는 2021년에 만든 빈 조직(저장소 0)으로 이 논문과 무관해 보입니다 |
| [dex-ninja.github.io](https://dex-ninja.github.io/) | 프로젝트 페이지 | - | - | - | 영상, 요약 표, BibTeX(CoRL 2026), 에셋 링크 |
| [huggingface.co/datasets/DexNinja/DexNinja_Simulated_Assets](https://huggingface.co/datasets/DexNinja/DexNinja_Simulated_Assets) | 공개 시뮬레이션 에셋 | 데이터셋 카드에 라이선스 메타데이터 없음(API) | - | 2026-10-13 생성 | 목록 API 기준 에셋 1,500개(딸기·사과·바나나·레몬·감자 각 300), 에셋당 파일 6종(`gs.ply`, `mesh.obj`, `mesh.stl`, `mesh-256.vox`, `mesh_sdf_res128_r0.60.pkl`, `pcd.ply`), 합계 약 140 GB. 받지 않았고 크기는 API 메타데이터에서만 읽었습니다 |
| [real-stanford/roboninja](https://github.com/real-stanford/roboninja) | 기준선·시뮬레이터의 출발점 [8] | MIT | 43 | 2023-05-01 | 부록 표 2의 기준선은 저자가 공개 설명을 바탕으로 다시 구현했다고 적습니다 |
| [Genesis-Embodied-AI/Genesis](https://github.com/Genesis-Embodied-AI/Genesis) | 장면 수준 로봇 시뮬레이터 [72] | Apache-2.0 | 30,315 | 2026-11-25 | |
| [NVIDIA/warp](https://github.com/NVIDIA/warp) | 미분 가능 시뮬 커널 [80] | Apache-2.0 | 7,357 | 2026-11-26 | |
| [tonyzhaozh/act](https://github.com/tonyzhaozh/act) | 정책 골격 [83] | MIT | 2,355 | 2024-07-23 | |
| [NVlabs/curobo](https://github.com/NVlabs/curobo) | IK·충돌 검사 [82] | Apache-2.0 | 2,045 | 2026-11-21 | |
| [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) | 음식 포즈 추정 [79] | SPDX 식별 안 됨(NOASSERTION) | 3,717 | 2026-04-29 | |
| [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) | 영상-3D 에셋 생성 [88] | MIT | 14,401 | 2026-10-27 | |
| [PKU-EPIC/DexGraspNet2](https://github.com/PKU-EPIC/DexGraspNet2) | 칼 사전 파지 [87] | 라이선스 파일 없음 | 165 | 2025-01-23 | |
| [RoboVerseOrg/RoboVerse](https://github.com/RoboVerseOrg/RoboVerse) | 시뮬 플랫폼 [81] | Apache-2.0 | 1,882 | 2026-11-23 | |
| [Taccel-Simulator/Taccel](https://github.com/Taccel-Simulator/Taccel) | 촉각 시뮬레이터 [43] | MIT | 203 | 2025-10-13 | |

D-REX [15], DiSECt [44], Robo-GS [31]의 공식 저장소는 이번 조사에서 찾지 못했습니다(검색 한도 안에서 `gh search`로 확인, 없다는 뜻은 아님).

**에셋 메타데이터에서 읽은 것.** 에셋은 범주마다 `0000`부터 번호가 붙은 300개이고 레몬이 포함됩니다. 본문 표 4(깔때기)는 레몬을 항목에 두지 않고 4개 범주의 수락 1,200개를 적습니다. 바나나 `0254`에는 `mesh_sdf_res128_r0.60.pkl`이 목록에 없습니다(파일 수 1,799). 에셋당 평균 크기는 약 93 MB(합계 140.0 GB / 1,500)입니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **RoboNinja [8]:** 기준선 수치의 출처. 부록 표 2의 기준선은 저자 재구현이고 원래 설정(강체로 붙인 칼, 힘 센서)과 달라 원 논문 수치와 직접 비교할 수 없습니다(부록 5.1 본문이 직접 밝힘).
- **MPM 변형체 시뮬레이션 [11, 12]:** 서론이 "작은 물성 불일치가 정책을 망친다"는 근거로 인용하는 것은 MPM 교재와 MLS-MPM 논문입니다. 절단 정책의 민감도를 직접 측정한 결과는 이 논문에 없습니다.
- **D-REX [15]:** 같은 1저자 그룹의 선행 틀. DexNinja가 새로 더한 것은 위상 변화 절단, 촉각, 생성 에셋입니다(논문 서술).

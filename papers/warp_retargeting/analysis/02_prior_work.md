# WARP 선행연구와 관련 연구

분석일 2026-11-10. GitHub 상태는 같은 날 `gh api`로 확인했습니다. 논문 인용은 원문 참고문헌 번호([n])를 씁니다.

## 1. 계보: 무엇을 이어받고 무엇을 바꿨나

| 축 | 논문이 이어받은 것 | 논문이 바꾼 것 | 참고 |
|---|---|---|---|
| 기하 해법 | SEW 표현, 닫힌 형식 SEW-Mimic 해, Stereo-SEW(팔꿈치 반평면), SP3, ik-geo | 손바닥을 hard constraint로, 두 단계 (몸통 오프셋 → 팔꿈치 정렬), 무릎 앞 해 선택 | [8] Kong et al. 2026 (arXiv:2602.01632), [24] Elias & Wen 2024, [25] Elias & Wen 2025 |
| 최적화 리타게팅 | MINK의 가중 속도 IK (실험 비교 대상) | 가중합 대신 등식 제약, 자세 정규화 항 대신 해석해 | [9] Zakka, MINK (2024). 부록 A |
| 전신 리타게팅 | GMR이 대표하는 "추종 + 자세 가중" 전신 IK | 팔꿈치 위치를 기하량으로 직접 다룸 | [4] Araujo et al. 2025 (arXiv:2510.02252): 논문은 GMR을 MINK 기반으로 서술 |
| 보행·전신 제어 | HOMIE, CLONE, BEHAVIOR가 쓴 전신 제어 | 준정적·조작 중심 가정, 기지는 느린 필터 | [10] HOMIE, [21] CLONE, [20] BEHAVIOR Robot Suite |
| 사람 데이터 수집 | UMI/DexCap류의 "사람 데이터 직접 수집" 발상 | 엔드이펙터만 저장하지 않고 상체 운동학 트리 전체를 저장 | [13] UMI, [14] DexCap |
| 사람 데이터 기반 모바일 조작 | 가장 가까운 기존 연구 | 직접 비교 없음 | **[3] HOMMI (Xu et al. 2026, arXiv:2603.03243): 사람 시연으로 전신 모바일 조작 학습.** 같은 목표이지만 논문 실험에 없음 |

## 2. 직접 대조하는 것과 빼 놓은 것

- **논문 실험의 비교 대상:** MINK-EF, MINK-TE, SEW-M (표 3·4, Fig. 5). 정책 비교는 MINK 하나뿐입니다.
- **서론에서만 언급된 것:** GMR [4], HOMIE [10]. 표에 없습니다.
- **요청하신 기준선 중 논문에 없는 것:** PHC와 OmniH2O류는 논문 참고문헌에 없어서 **비교 대상이 아닙니다.** 이 분석에서 그 결과를 인용하지 않았습니다.
- **가장 가까운 연구인 HOMMI [3]:** 같은 해 같은 목표(사람 시연 → 전신 모바일 조작)입니다. 논문이 이것과 비교하지 않은 이유는 적혀 있지 않습니다.

## 3. 이 저장소의 관련 분석과의 관계

| 폴더 | 대상 (arXiv) | WARP와 맞닿는 지점 | 차이 |
|---|---|---|---|
| `gnr/` | GNR, Generative Neural Retargeting (2610.12440, Meta Reality Labs) | 리타게팅 해의 다중성이 정책 학습을 해친다는 같은 문제의식. GNR은 잔차를 **생성 모델**로 다중 해를 받아들이고, WARP는 해를 **유일하게** 만듦 | 손 리타게팅(DG-5F), 샘플링 MPC + flow matching, 공개 코드 없음 |
| `flashdexretarget/` | FlashDexRetarget (2610.01849) | 사람 시연을 로봇 손 궤적으로 옮기는 과제 | RL 정책으로 리타게팅 (손), 학습은 GPU 필요 |
| `chord/` | CHORD (2607.00033, NVIDIA, CoRL 2026) | 접촉 정보를 리타게팅·학습의 신호로 쓴다는 점. WARP는 접촉을 **기하로 제약하지 않음** (손바닥만 hard) | 손 조작 RL, 마찰원뿔 보상 |
| `cross_embodiment_wm/` | Scaling Cross-Embodiment WM (2511.01177, IROS 2026) | 서로 다른 몸체 사이의 간극을 표현 단계에서 흡수한다는 발상 (파티클 변위 대 SEW) | 파티클 세계모델, 손 전용, MPC |
| `dreamx/` | DreamX (2611.05377) | 사람 데이터를 로봇 명령으로 옮김. 기하 리타게팅 대신 영상 latent action을 씀 | 영상 diffusion 백본, 공개 코드 없음 |
| `motion_latent_action/` | Motion-Focused Latent Action (2606.18955, IROS 2026) | 사람 영상을 로봇 VLA 학습에 쓰는 법. 기하가 아니라 잠재 행동 | 사람 egovideo, 마스크 파이프라인 필요 |
| `dextouch_wm/` | DexTouch-WM (2609.20649, IROS 2026 워크숍) | WARP의 한계(이미지 없음, 접촉 정보 없음)를 촉각으로 보완하는 방향 | 촉각 세계모델, 워크숍 논문 |

**정리:** WARP의 학습 쪽 주장(일관된 행동 목표 → 좋은 정책)은 GNR의 "다중 해를 생성 모델로 받는다"와 정면으로 반대되는 설계입니다. 어느 쪽이 맞는지는 같은 과제에서 두 방식을 비교해야 알 수 있고, 두 논문 모두 그 비교를 하지 않았습니다. 이 비교 방향은 연구 활용 측면에서 따로 정리했습니다.

## 4. GitHub 확인 (2026-11-10, `gh api`)

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | 별 / 포크 | 마지막 push |
|---|---|---|---|---|---|
| `Euler-Rodrigues-Lab/WARP_retargeting` | **이 논문의 공개 릴리스.** 리타게팅 파이프라인, MINK 기준선, 지표 일부, 재생 도구. WARP(c-SEW) 솔버는 없음 | 저자 조직 (조직 소개: "Robotics Embodiment Retargeting using Geometry") | MIT | 1 / 2 | 2026-11-08 (커밋 `172b1dc`, 첫 커밋 `b98b97f` "Initial public WARP retargeting release") |
| `warp-retargeting/warp-retargeting.github.io` | 프로젝트 페이지 저장소 (Nerfies 템플릿) | 저자 조직 추정 | 없음 | — | 2026-10-08 |
| `Euler-Rodrigues-Lab/rby1_teleop` | RB-Y1 텔레옵, MuJoCo 모델, SEW 기하 리타게팅 래퍼. WARP 저장소의 submodule (`973ed0d`) | 저자 조직 | 없음 | 1 | 2026-10-27 |
| `Euler-Rodrigues-Lab/geo_kin_core` | "SEW 기하 텔레옵 리타게팅의 공개 코어" (타입, 솔버 인터페이스, 로봇 사양). 공개 폴백(MINK)이 여기 연결 | 저자 조직 | 없음 | — | 2026-10-27 |
| `Euler-Rodrigues-Lab/XRT_devices` | 입력 장치 어댑터 (SEED CSV 읽기 포함) | 저자 조직 | MIT | — | 2026-10-27 |
| `Euler-Rodrigues-Lab/SEW-Geometric-Teleop` | 논문이 말하는 원본 소스 체크포인트 (`4f0fa5e`) | — | — | — | **404, 공개 아님** |
| `kevinzakka/mink` | MINK (MuJoCo 기반 IK), 기준선의 엔진 | 원 저자 | Apache-2.0 | 1,604 | 2026-11-02 |
| `NVlabs/dexmimicgen` | DexMimicGen 환경 (시뮬 실험의 데이터 원천) | NVIDIA | NOASSERTION (확인 안 됨) | 298 | 2025-12-06 |
| `warp-retargeting` (검색어) | `NVIDIA/warp`(물리 엔진)와 이름 혼동 주의 | — | — | — | — |

- GitHub 검색 `WARP whole-body retargeting`으로 찾은 저장소는 위 `Euler-Rodrigues-Lab/WARP_retargeting` 하나뿐입니다. 논문 본문과 프로젝트 페이지에는 코드 링크가 **없습니다** (프로젝트 페이지 HTML의 링크를 전부 확인). 공개 저장소의 홈페이지 필드만 프로젝트 페이지를 가리킵니다.
- 공개 저장소 clone: `repo/WARP_retargeting` (얕은 clone, 커밋 `172b1dcde11d3942d3fad6d378695b5703631ded`, 저장소 `.gitignore`에 `repo/` 추가). 세부 대조는 `04_code_verification.md`.
- SEW-Mimic(`arXiv:2602.01632`)의 전용 저장소는 확인하지 못했습니다. 조직 저장소 목록에 해당 이름이 없습니다(`SEW-Geometric-Teleop` 404). GMR 저장소는 검색으로 찾지 못했습니다. 두 항목은 "확인 못 함"으로 둡니다.
- 저자 개인 계정 세 곳의 저장소 목록에서도 WARP 관련 저장소는 찾지 못했습니다(이름으로만 검색, 추가 확인 안 함).

## 5. 기준선 해석 메모

- **MINK-EF/TE**는 "가중 속도 IK + 기본 자세 정규화"입니다. 손 목표와 자세 목표의 가중치가 해를 바꾸므로 결과가 가중치 선택에 민감합니다. 부록 A.2 스윕이 이 점을 보입니다(그림으로만 제시).
- **SEW-M**은 SEW 해를 목표 없이 쓰는 기준입니다. 같은 저자군이므로 WARP의 기여를 분리하기 가장 좋은 대조입니다. 실제로 손바닥 정확도(JL off 178.979 mm)가 나쁘고, 몸통 이탈 0.000과 방향 정확도는 WARP와 같습니다. 즉 **손바닥 hard constraint와 몸통 오프셋이 기여하는 부분이 드러나는 비교**입니다.
- **공정성 문제:** MINK 변형이 소프트 가중치로 튜닝되지 않았습니다("튜닝 없음"이 WARP의 장점으로 제시됨). 기준선에 같은 수준의 튜닝을 했는지는 논문에 없습니다.

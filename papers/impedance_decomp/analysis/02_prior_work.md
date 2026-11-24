# 02. 선행연구와 관계

## 1. 계보

논문이 서는 자리는 세 갈래가 만나는 곳입니다. (a) 임피던스·작업 공간 제어, (b) 나사 이론에 기반한 관절 물체의 운동학, (c) 생성형 정책이 만드는 궤적 분포입니다.

| 갈래 | 인용 | 이 논문이 가져다 쓴 것 |
|---|---|---|
| 임피던스 제어 | Hogan [3], Khatib의 작업 공간 제어(부록 A.2에서 언급) | 식 1–2의 가상 질량·감쇠·강성과 τ = Jᵀ F_cmd, 관성으로 정규화한 게인(식 17–18) |
| 군 위의 PD 제어 | Bullo & Murray [19, 20] | body 프레임 이중 측지 오차 e_T와 어드조인트로 수송한 e_V (식 9) |
| 나사·열린 체인 운동학 | Lynch & Park의 *Modern Robotics* [4] | body Jacobian의 열 B_i(q), S(q) = span{B_i}, 곱의 지수(PoE) 공식(식 3, 24) |
| se(3) 위 메트릭 | F. C. Park 1995 [18] | PCA에 필요한 se(3) 내적 선택, 특성 길이 α (식 5, 29–34) |
| 관절 물체 추정 | ScrewNet [10], Jain 외 [11], ScrewSplat [13], Sturm 외 [5], Katz 외 [6], Hausman 외 [7], Martín-Martín & Brock [9], Buchanan 외 [14], 시연 기반 [8, 12] | 비교 대상이 아니라 "별도 인식·탐색 단계가 필요한 대안"으로만 인용 (§1, §2) |
| 분산 기반 강성 | Calinon 외 [15] | 위치 시연 분산 → 강성. 이 논문은 body twist 공간에서 분산이 작은 방향이 곧 구속 방향이라고 보고 같은 아이디어를 twist 공간으로 옮김 |
| 방향별 제어 | SwRL [16], ACP [17] | SwRL은 작업 공간 분해를 사전 지식으로 주고, ACP는 외력 방향의 강성을 낮추는 규칙을 쓴다고 대비 (§2) |
| 생성형 정책 | Diffusion Policy [21] | 하나의 관측에서 N개 샘플링이 가능한 정책이면 됨. 정책은 수정하지 않음 |
| 시뮬레이터 | Isaac Sim [22] | 시뮬 평가 환경 |

**새로운 부분으로 보이는 것(논문 서술 기준):** 정책 샘플의 산포에서 구속 부분공간을 읽어 6×6 게인으로 바꾸는 파이프라인 전체. 개별 부품(비중심 PCA, Λ̄-직교 사영, body 오차, 2차계 게인 규칙)은 기존 기법이며, 논문도 그렇게 인용합니다.

## 2. 가까운 대안과 구성 요소별 출처

| 목적 | 논문의 선택 | 가까운 대안 (인용된 것) | 차이 |
|---|---|---|---|
| 구속 방향을 얻는 법 | 정책 앙상블의 twist PCA | 관절 축을 영상·힘·상호작용·시연으로 추정 [5–14] | 별도 모듈과 기준 프레임 정렬이 필요 vs 정책 출력만 사용. 단 정책이 그 물체를 이미 조작할 수 있어야 함 |
| 방향별 강성 | 추정 부분공간의 Λ̄-직교 사영 | SwRL의 given 분해 [16], ACP의 힘 방향 규칙 [17], Calinon의 분산 → 강성 [15] | 분해를 추정하되 힘 신호는 제어 루프에 넣지 않음(부록 A.1, H.3) |
| 위치 vs 방향 문제 | 6×6 한 번 설정 | 하이브리드 힘-위치 제어(초록에서 "force와 position을 직교 부분공간으로 엄격히 분리"로 대비) | 임피던스로 연속적으로 낮춤 (r = 0.1) |
| 배치 의존 물체 | 전역 포락선 S_env 추정 | 순간 부분공간을 따라가는 방법(향후 과제로 언급) | 크랭크에서 개선이 작아짐(표 2) |

인용된 문헌 중 방향별 compliance를 직접 다루는 것은 ACP [17]뿐이고, 이 논문은 ACP를 실험으로 비교하지 않고 서술로만 구분합니다.

## 4. 공식 코드와 관련 저장소

모든 수치는 GitHub API로 2026-11-24에 확인했습니다. 저장소는 정적으로만 읽었고 실행·설치하지 않았습니다.

| 저장소 | 관계 | 라이선스 | 상태 (별/포크, 마지막 push, 커밋, 비고) |
|---|---|---|---|
| [erickun0125/constraint-decomposed-impedance-control](https://github.com/erickun0125/constraint-decomposed-impedance-control) | **공식 구현** (방법, Isaac Lab 시뮬, 평가·학습 스크립트) | MIT (Franka hand 메시는 Apache-2.0, `NOTICE`) | 5/0, 2026-10-05, 커밋 `d1f99f98db2c36f8673278ac0c60a871a89792f7` (2026-10-06 KST, 단일 커밋). 약 2.4 MB, 시뮬레이션만, 실로봇 코드 없음 |
| 같은 저장소 Release `v1.0.0` | 정책 체크포인트 5개 (revolute, cylindrical, planar, universal, multitask) | 저장소와 같음 | 2026-10-05, 각 약 355 MB (총 약 1.8 GB). 받지 않음 |
| [erickun0125/constraint-decomposed-impedance](https://github.com/erickun0125/constraint-decomposed-impedance) | 프로젝트 페이지 소스 (GitHub Pages) | 명시 없음 | 0, 2026-10-14, 약 34 MB (JS, 영상) |
| [프로젝트 페이지](https://erickun0125.github.io/constraint-decomposed-impedance/) | 논문, 코드, 영상, BibTeX 링크 | - | "CoRL 2026 Spotlight" 표기 |
| [erickun0125/SWIVL](https://github.com/erickun0125/SWIVL) | 같은 저자 계정의 관련 작업 (나사-wrench 기반 임피던스 가변 학습, 양팔 관절 물체) | MIT (README 배지, API는 null) | 0, 2026-02-08. 논문이 인용한 SwRL [16]과의 관계는 확인하지 않음 |
| [yifan-hou/adaptive_compliance_policy](https://github.com/yifan-hou/adaptive_compliance_policy) | ACP [17] 공식 구현 | MIT | 145. 이 분석은 읽지 않음 |
| [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | Diffusion Policy [21] | MIT | 4743, 2024-12-24. 공식 코드는 이 구현 대신 robomimic의 DiffusionPolicyUNet을 사용 |
| [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic) | 정책 학습 프레임워크 (공식 코드가 커밋 `0ca7ce7`을 고정) | MIT | 1609, 2026-08-09 |
| [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | 시뮬 프레임워크 (v2.3.2, 커밋 `3d42bff`, Isaac Sim 5.1.0) | NOASSERTION (API) | 8709, 2026-11-24. Linux/Windows + NVIDIA GPU 전제 (macOS 미지원) |

공식 저장소는 논문 제출(2026-10-12) 일주일 전인 2026-10-05 이후 갱신이 없고 커밋이 하나입니다. 저장소 README는 `con_dec_imp/`(방법, numpy·torch만)와 `con_dec_imp_sim/`(Isaac Lab 환경)을 분리해 두었고, `examples/data`에 시뮬에서 기록한 N=128 앙상블 4개(약 0.1 MB씩)를 함께 제공합니다. 이 데이터로 시뮬이나 정책 없이 Algorithm 1을 돌릴 수 있으며, 이 분석의 toy(`03` §8)가 그 데이터를 썼습니다.

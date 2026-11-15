# 02 선행연구와 관련 자료 (Flow Policy Gradients, arXiv:2602.02481)

## 1. 계보

| 세대 | 연구 | 무엇을 했나 | 이 논문과의 관계 |
|---|---|---|---|
| 기반 | PPO (Schulman 외, arXiv:1707.06347) | 클리핑된 정책 비율 목적 | 목적 형식의 원형(식 (1)–(2)). 논문 [2] |
| 기반 | Sutton 외 1999 | 정책 gradient 정리 | 논문 [1] |
| 가우시안 로봇 RL | Rudin 외 2022 (보행, IsaacLab 계열), Lee 외 2020 (ANYmal) | 가우시안 PPO로 보행 | 비교 기준. §4.5와 부록 C.1에서 IsaacLab 기본 rsl_rl 구현과 비교 |
| 가우시안 로봇 RL | BeyondMimic (Liao 외, Science Robotics 2026) | 가이드 확산으로 모션 트래킹 | G1 모션 트래킹의 보상·관측·종료 조건 출처(부록 B.2) |
| flow/diffusion 모방 | Diffusion Policy (Chi 외), π0 (Black 외 2024), π0.5 (2025) | 전문가 행동으로 flow/확산 정책 학습 | 베이스 정책 계열. 논문은 이들이 행동 레이블을 요구한다고 지적(§2) |
| flow 정책 RL (우도 기반) | DPPO (Ren 외, arXiv:2409.00588) | 확산 정책을 디노이즈 단계별 MDP로 PG 학습 | 조작 비교 기준(DPPO 고정·학습 노이즈). 우도가 샘플링 궤적에 묶임 |
| flow 정책 RL (우도 기반) | ReinFlow (Zhang 외, arXiv:2505.22094) | 학습 가능한 노이즈로 우도를 정확히 만듦 | 조작 비교 기준(부록 C.4) |
| flow 정책 RL (우도 기반) | NCDPO (arXiv:2505.10482), GenPO (arXiv:2505.18763) | 디노이즈 단계를 역전파하거나 가역 구조 사용 | 논문 §2에서 "구조 제약과 펼치기(unrolling)가 필요하다"고 비판 |
| 우도 없는 flow PG | **FPO** (McAllister 외, arXiv:2507.21053, 제목 "Flow Matching Policy Gradients") | CFM 손실 차이로 대리 비율을 만들어 PPO식 학습 | 직접 선행(논문 [13]). 합성 과제에서 검증 |
| 우도 없는 flow PG | **FPO++ (이 논문)** | 샘플별 비율 + ASPO + 로봇 과제 적용 | 본 분석 대상 |
| 음의 advantage 안정화 | SPO (Xie 외, arXiv:2401.16025) | 클리핑 대신 2차 패널티로 trust region | ASPO의 음수 advantage 쪽 식(식 (10)) 출처 |
| 오프라인 flow RL | FQL (Park 외, arXiv:2502.02538), Q-score matching, 어드조인트 매칭(Q-learning with adjoint matching, arXiv:2601.14234), DIME, EBFM 계열 | Q 학습과 생성 모델 결합, 역전파 불안정 해결 | 부록 A. 온라인 PG와 다른 문제 |
| 1단계 적분 | "When One Step Is Enough" (arXiv:2610.30735) | 흐름 정책의 Euler 단계 수 실험 | 본 분석 `toy_euler_steps.py`와 같은 질문(§3 참조) |
| 장기 맥락 | 확산 강제(Diffusion Forcing, arXiv:2407.01392) | 시퀀스 확산 | 논문 "미래 응용" 목록 |

**요약:** 이 논문의 기여는 새 우도 근사가 아니라, 우도 없이 flow 정책을 로봇 과제에서 안정적으로 학습시키는 **구현 수준의 변경 두 가지**(샘플별 비율, ASPO)와 그 검증입니다.

## 2. 구성 요소별 출처

| 요소 | 논문 위치 | 출처 | 메모 |
|---|---|---|---|
| CFM 손실 차이 비율 | 식 (3), §3.1 | FPO [13] | 논문 자체 기여 아님 |
| 샘플별 비율 | 식 (9), §3.2 | 이 논문 | FPO 식 (8)과 비교 |
| ASPO (PPO + SPO) | 식 (10)–(11), §3.2–3.3 | PPO [2] + SPO [33] 조합 | 조합은 이 논문. 양/음 advantage 분기가 핵심 |
| 시간 샘플링, 손실 클램프 | 부록 B.2.3 | 저장소 코드 `fpo.py`에서 확인 | 본문에는 없음(`04` D4–D5) |
| 0 샘플링(평가) | §4.4, 부록 B.2.2 | 이 논문 (v1 표 I 근거) | v2에서 표 삭제 |
| 조작 베이스 정책 | 부록 B.3 | ResFiT [44] 데이터 처리, DPPO 구현 [7] 변형 | 베이스 정책은 W&B 프로젝트 `flow-bc` (공개 여부 미확인) |
| 조작 과제 | 부록 B.3, 그림 A.2 | RoboMimic [24], DexMimicGen [43] | DexMimicGen는 NVIDIA 비상업 라이선스 (저장소 README 기준, `04` §5) |
| 보행 환경 | 부록 B.1 | IsaacLab [36], rsl_rl [47] | 보상 항목은 IsaacLab 기본값 |
| 모션 트래킹 | 부록 B.2 | BeyondMimic [4], LAFAN [42] | 트래킹 보상·관측·종료 |
| 휴머노이드 배포 | §4.2 | HumanoidVerse [40], IsaacGym [41] | T1 보행 |
| 시각 인코더 | 부록 B.3 | ViT [45], DPPO의 ViT 구현 | `src/vit.py`는 DPPO에서 일부 가져옴 |
| 가우시안 PPO 기준 | 부록 C.1 | rsl_rl 기본 설정 [47] | 논문이 튜닝한 값과 기본값을 함께 씀 |


## 4. GitHub와 코드 자료 (확인 2026-11-15, `gh repo view`)

| 자료 | 링크 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| **FPO++ 공식 저장소** | https://github.com/amazon-far/fpo-control | **없음** (LICENSE 파일 없음, GitHub `licenseInfo: null`) | 318 | 2026-08-12 | 클론 커밋 `8f460e4` 정적 읽기만. 하위 구성요소 라이선스는 README에 있음 |
| 프로젝트 페이지 | https://hongsukchoi.github.io/fpo-control | - | - | - | 영상과 링크 |
| FPO 원 저장소 | 검색으로 찾지 못함 | - | - | - | 미확인 |
| IsaacLab | https://github.com/isaac-sim/IsaacLab | BSD-3-Clause (README 기준, GitHub 자동 판정은 "Other") | - | - | 저장소가 submodule로 참조 |
| HumanoidVerse | https://github.com/LeCAR-Lab/HumanoidVerse | MIT | 474 | 2025-06-12 | T1 보행 기반 |
| BeyondMimic 계열 whole_body_tracking | https://github.com/HybridRobotics/whole_body_tracking | MIT | 2,443 | 2026-10-16 | G1 모션 트래킹 기반 |
| DPPO | https://github.com/irom-princeton/dppo | MIT | 875 | 2025-02-04 | 조작 비교 기준(본문 4.3은 FPO 저장소의 수정 구현, C.4는 원본 코드) |
| ReinFlow | https://github.com/ReinFlow/ReinFlow | MIT | 379 | 2026-04-24 | 조작 비교 기준(부록 C.4에서 저자 공개 코드 사용) |
| 비공식 FPO 구현 | https://github.com/SestoAle/Flow-Matching-Policy-Gradient | 라이선스 없음 | 0 | 2026-05-03 | 비공식. 참고만 하고 코드는 쓰지 않음 |

**라이선스 주의:** 공식 저장소에는 최상위 LICENSE가 없습니다. README는 하위 구성요소(IsaacLab BSD-3, rsl_rl BSD-3, LeRobot Apache-2.0 vendored, robosuite MIT vendored, DPPO MIT 일부, DexMimicGen NVIDIA 비상업)를 나열하지만, 저장소 전체 재배포 조건은 명시하지 않습니다. 따라서 저장소 코드를 복사해 공개하지 않고, 읽은 내용을 요약하는 수준으로만 썼습니다(`04` §5).

## 5. 문헌 확인 메모

- 논문 본문의 인용 번호는 v2 기준입니다. v1에서 인용 번호가 바뀌었을 수 있습니다.
- 인용한 논문 중 FPO(2507.21053), DPPO(2409.00588), ReinFlow(2505.22094), SPO(2401.16025)는 arXiv 초록 페이지로 제목과 라이선스만 확인했습니다. 본문은 읽지 않았습니다. 라이선스는 DPPO와 ReinFlow가 CC BY 4.0, SPO가 non-exclusive입니다.
- 나머지 인용은 이 논문의 참고문헌 목록에서 가져왔습니다. 원문은 확인하지 않았습니다.

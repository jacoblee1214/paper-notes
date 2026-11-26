# 02. 선행연구와 가까운 대안

## 1. 계보

| 흐름 | 대표 | RoFacto와의 관계 |
|---|---|---|
| 행동 조건 로봇 영상 세계모델 | IRASim [3], Ctrl-World [4], DreamDojo [5], WorldGym [7] | 행동 벡터를 조건층(AdaLN, 교차 주의, 프레임별 행동)에 넣는다. RoFacto는 이 경로를 비교 대상(AdaLN 상태 벡터, Ctrl-World pose)으로 삼는다 |
| 잠재 행동 | AdaWorld [23], CLAM [25], UniVLA [26], 잠재 행동 세계모델 [27] | 행동 표현을 학습한다. RoFacto는 반대로 기하를 명시적으로 렌더링한다(§2) |
| 시각 조건(렌더링) | **VAP** [8] (로그된 로봇 상태를 영상 프롬프트로 렌더링), **DWM** [44] (정적 장면 렌더링 + 손 메시 궤적, 잔차 동역학 + 인페인팅) | RoFacto의 직접 조상. VAP과는 "무엇을 렌더링하나"(로그 상태 대 명목 궤적)가, DWM과는 "손 대신 로봇, 정책 가능 입력"이 다르다 |
| 영상 제어 확산 | ControlNet [28], ControlVideo [29], 궤적·광류 조건 [38–43] | 일반적 시각 조건 메커니즘 |
| 백본 | Wan 2.1 [47], VideoX-Fun Fun-InP [45], SVD [58], Cosmos | RoFacto는 Wan2.1-Fun-V1.1-14B-InP에 채널 연결 + LoRA |

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 논문 문장 |
|---|---|---|
| 정적 장면 + 잔차 동역학 + 인페인팅 백본, 마스크 전체 1 | DWM [44], VideoX-Fun [45] | "Following the residual-dynamics formulation of DWM"(§3.5) |
| 채널 연결 조건 | 일반적 inpainting/Control 경로. 패치 임베딩 확장 | 부록 B |
| 로그 상태 렌더링 | VAP [8] | §1, §3.2 |
| 4 스텝 증류 추론 | LightX2V [68] | 부록 B |
| 명목 궤적(컨트롤러 재생) | 이 논문의 새 구성. Isaac Lab [57]의 Panda 컨트롤러 재생, RoboCasa 충돌 없는 shadow rollout | §4.1, 부록 A |
| 깊이 | FoundationStereo [64](DROID), 시뮬레이터 렌더(RoboCasa) | 부록 A |
| 사람 시연 → 로봇 | AnyTeleop [61], PyRoki [62], MiniMax-Remover [69], DexYCB [56] | 부록 F |
| 과제 결과 판정 | GPT-5.4 한 번 호출, 12프레임 | 부록 E |

**새로운 것**: 명목 궤적(식 1)이라는 중간 신호와, 이를 URDF 메시 + EEF 깊이로 렌더링해 정적 문맥과 한 인터페이스로 묶는 설계. 자체 모델 구조(채널 연결 + LoRA)는 새롭지 않다.

## 4. GitHub와 코드 자료 (확인 2026-11-26)

| 자료 | 링크 | 라이선스 | 별 / 마지막 push | 비고 |
|---|---|---|---|---|
| **공식 코드 RoFacto** | https://github.com/bjkim95/rofacto | Apache-2.0 | 별 32, 커밋 6, 마지막 push 2026-11-10, 최신 `f9504d0b839ca2eb8936407eb23f7ae9d9ba4078` | `repo/` 얕은 clone, 정적 읽기만. Wan2.1 구현 + 추가 Cosmos3 백엔드 |
| 프로젝트 페이지 | https://bjkim95.github.io/rofacto/ | - | - | 초록은 v1 문구. 정성 영상 위주 |
| 체크포인트(Hugging Face) | `byungjun-kim/RoFacto-Wan2.1-14B`, `RoFacto-Cosmos3-Super` | Apache-2.0(Wan 어댑터), OpenMDW-1.1(Cosmos3 어댑터) | - | 받지 않음. 저장소 `docs/checkpoints.md` 기준: Wan RGB(DROID) 8,000 업데이트, Wan 깊이(DROID + RoboCasa) 8,000, Cosmos3-Super(batch 32) 5,000 |
| 베이스 Wan2.1 | https://github.com/Wan-Video/Wan2.1 | Apache-2.0 | 별 17,494 / 2026-03-05 | |
| VideoX-Fun | https://github.com/aigc-apps/VideoX-Fun | Apache-2.0 | 별 2,345 / 2026-11-25 | Fun-InP 경로 |
| LightX2V | https://github.com/ModelTC/lightx2v | Apache-2.0 | 별 3,027 / 2026-11-26 | 4 스텝 증류 |
| Ctrl-World | https://github.com/Robert-gyj/Ctrl-World | MIT | 별 587 / 2026-04-08 | 기준선 계열 |
| DiT4DiT | https://github.com/Mondo-Robotics/DiT4DiT | MIT | 별 480 / 2026-06-30 | RoboCasa-GR1 롤아웃 정책 |
| robocasa-gr1-tabletop-tasks | https://github.com/robocasa/robocasa-gr1-tabletop-tasks | NOASSERTION | 별 167 / 2025-08-09 | |
| PyRoki | https://github.com/chungmin99/pyroki | MIT | 별 1,826 / 2026-04-01 | |
| dex-retargeting | https://github.com/dexsuite/dex-retargeting | MIT | 별 1,322 / 2025-08-21 | |
| cuRobo | https://github.com/NVlabs/curobo | Apache-2.0 | 별 2,045 / 2026-11-21 | 궤적 편집 |
| HRDexDB | https://github.com/snuvclab/HRDexDB | 미표기 | 별 47 / 2026-11-20 | 데이터셋 사이트 |
| SPIDER | https://github.com/facebookresearch/spider | NOASSERTION | 별 614 / 2026-11-21 | 저장소가 retarget에 사용 |

공식 저장소가 공개하는 범위와 공개하지 않는 범위는 `04_code_verification.md`에 있다(평가 코드, AdaLN·SVD 기준선 체크포인트, 과제 결과 판정 코드는 저장소에서 확인되지 않음).

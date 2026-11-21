# 02. 선행연구, 공식 저장소, 관련 분석

## 1. 계보: 사람 영상에서 로봇 학습까지

이 논문이 놓인 흐름은 네 갈래입니다. 각 갈래의 대표 연구와 EgoEngine이 가져온 부분을 적었습니다 (논문 §2 및 참고문헌 번호).

| 갈래 | 대표 연구 | EgoEngine과의 관계 |
|---|---|---|
| 사람 영상을 로봇 학습에 쓰기 (공동 학습, 사전 학습) | EgoMimic [10], MimicPlay [15], EgoBridge [13], EMMA [14], ImMimic [12] | 논문은 이들이 여전히 **로봇 시연을 목표 도메인으로 요구한다**고 봅니다 (§2). EgoEngine은 로봇 시연을 쓰지 않습니다. |
| 사람 영상의 로봇 외형 치환 (시각 간극) | EgoMimic 마스크 [10], Phantom [17] (CoRL 2025), Masquerade [34] (ICRA 2026), VACE [42] (WAN2.1 [62]) | 같은 inpainting-rendering-blending 흐름을 쓰되, 디지털 트윈으로 접촉과 가림을 맞춘다는 점이 다릅니다 (§2, §4.2). 비교 기준선으로 쓰입니다. |
| 사람 손 → 로봇 손 리타겟팅, 물리 실행 (행동 간극) | MINK 역기구학 [49], SPIDER [32] (2026), H2S2R [30] (2025), SEW [19], OpenTV [20] | 리타게팅은 MINK, MPC 기준선은 SPIDER, RL 기준선은 H2S2R입니다. EgoEngine은 이들을 섞은 모드 전환을 제안합니다 (§2 Real2Sim2Real). |
| 사람 데이터 대규모 수집과 확장 | EgoDex [7], EgoVerse [5], EgoScale [8], Ego4D [6], Aria [9] | EgoVerse와 EgoDex는 부록 A.4의 디지털 트윈 확장성 시험에 쓰였습니다. EgoScale은 같은 확장 방향의 다른 접근입니다. |

**가장 가까운 대안 세 개:**
1. **Phantom [17]:** 로봇을 영상에 합성하고 그 영상으로 바로 학습합니다. 시각 충실도 측면에서 EgoEngine의 기준선입니다. EgoEngine은 같은 시각 흐름에 물리 궤적을 더합니다.
2. **H2S2R [30]:** 사람 한 번의 시연으로 sim-to-real RL을 합니다. 표 2에서 RL 기준선이며, 성공률은 같고 비용은 더 큽니다.
3. **Zero-shot 사람 영상 학습 [11] ZeroMimic, [16] EgoZero:** 논문은 이 둘이 더 단순한 조작이나 낮은 자유도 그리퍼에 머문다고 비교합니다 (§2 마지막 단락).

## 2. 공식 저장소와 프로젝트 페이지 (GitHub 표)

확인 날짜 2026-11-22. 저장소 메타데이터는 `gh repo view`로 가져왔습니다. 소스 코드는 얕은 복제(`--depth 1`)로 `repo/`에 받아 **정적으로만** 읽었습니다. 실행, 설치, import는 하지 않았습니다.

| 저장소 | 역할 | 라이선스 | 별 | 마지막 push | 확인한 커밋 | 상태 |
|---|---|---|---|---|---|---|
| [GaTech-RL2/EgoEngine-CoRL2026](https://github.com/GaTech-RL2/EgoEngine-CoRL2026) | 공식 코드 (행동 분기, 시각 합성, HDF5 내보내기) | 저장소 파일 `LICENSE`에 따라 **혼합**: `spider/`와 `action/`은 CC BY-NC 4.0, `visual/`은 MIT. GitHub API는 "Other" | 18 | 2026-10-16 | `b1363c2` (2026-10-16) | `repo/` 정적 읽기 완료 |
| [Randle-Github/EgoEngine-MPC](https://github.com/Randle-Github/EgoEngine-MPC) | 행동 분기 상류 (SPIDER 기반) | SPIDER (CC BY-NC 4.0, 저장소 `THIRD_PARTY_NOTICES.md` 기준) | 미확인 | 미확인 | `602d6ca` | 주소만 확인 (`UPSTREAMS.json`에서 옮김, 접속 확인 안 함) | 미확인 | `602d6ca` | 주소만 확인 (`UPSTREAMS.json`에서 옮김, 접속 확인 안 함) |
| [GaTech-RL2/robosuite](https://github.com/GaTech-RL2/robosuite) (branch `yc`) | 시각 분기 상류 (RB-Y1 렌더링, 합성기) | MIT (robosuite) | 미확인 | 미확인 | `8396578` | 주소만 확인 (`UPSTREAMS.json`에서 옮김, 접속 확인 안 함) |
| [egoengine.github.io](https://egoengine.github.io) | 프로젝트 페이지 (CoRL 2026 표기, 논문 그림 1 재사용) | 해당 없음 | 해당 없음 | 해당 없음 | 본문 추출 | 읽음 |

**코드 저장소 요약 (정적 읽기 기준):**
- 실행 경로 두 개: 시각 분기는 Python 3.10 + OpenGL (CUDA 불필요), 행동 분기는 Python 3.12 + Linux + NVIDIA CUDA (README와 `docs/REPRODUCIBILITY.md`).
- 공개 입력은 **사전 처리된** 6개 사례입니다 (Aria `mustard_1..3`, TACO 3개). FoundationPose, FoundationStereo, SAM2 등 인식 단계와 배경 제거 모델은 릴리스에 없습니다 (README "FoundationPose and early perception/reconstruction are outside this release").
- 자세한 대조는 `04_code_verification.md`에 있습니다.

## 4. 구성 요소별 출처 (논문 본문과 부록 기준)

| 구성 요소 | 출처 | 논문 번호 |
|---|---|---|
| 깊이 추정 | FoundationStereo | [46] |
| 마스크 | SAM2 | [47] |
| 물체 6D 자세 추적 | FoundationPose | [48] |
| 역기구학 리타게팅 | MINK | [49] |
| 정책 강화학습 | PPO [50], RSL-RL 구현 [51] | [50, 51] |
| 사람 제거 inpainting | Inpaint-Anything v2 | [58] |
| 정책 구조 | HPT | [59] |
| 평가 데이터 | TACO | [60] |
| 로봇 | RB-Y1, XHand | 부록 A |
| 시각 특징 비교 | ResNet18, VGG16, DINOv2 | [63, 64, 65] |
| 매끄러움 지표 | SPARC | [67] |
| 행동 기준선 | Spider(MPC) [32], H2S2R(RL) [30] | §4.3 |
| 시각 기준선 | EgoMimic [10], VACE [42], Phantom [17] | §4.2 |
| 확장 시험 데이터 | EgoDex [7], EgoVerse [5] | 부록 A.4 |
| 모드 전환 의미 | MCTS 계열 [57] LEAGUE (Cheng & Xu, 2023). 논문은 heuristic이며 완전 MCTS가 아니라고 적음 | §3.2.2 |

## 5. 참고 목록 점검

- 논문 본문 인용 중 [1]–[67]이 목록에 모두 있습니다 (텍스트 추출 기준).
- [7]의 본문 연도와 목록 연도가 다릅니다 ("2026" 대 arXiv 2505.11709). 인용 표기 오류입니다. 참고만 합니다.
- [17] Phantom은 CoRL 2025, [34] Masquerade는 ICRA 2026으로 표시되어 있습니다.

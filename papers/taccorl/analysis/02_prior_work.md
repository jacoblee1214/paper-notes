# 02. 선행연구와 관련 저장소

번호는 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| TacCoRL의 요소 | 출처 | 관계 |
|---|---|---|
| 기반 정책 π0.5 (flow 행동 전문가) | π0 [3], π0.5 [4], RT-2 [1], OpenVLA [2] | 구조를 그대로 쓰고 촉각 토큰 경로만 더합니다(§3.2) |
| 촉각을 VLA에 붙이는 방식 | Touch begins where vision ends [7], 촉각 인지 모듈 [15, 17, 38], 추론 시 촉각 유도 [18], 촉각-언어-행동 [12–14] | 저자가 "증강" 계열로 분류합니다(§2). 접촉 게이트는 [17]의 contact-aware fusion을 가져왔다고 씁니다(식 4 앞) |
| 촉각 이력 창 | OpenTouch [53]의 접촉 이력 분석 | 이력 길이 L = 10의 동기(§3.2) |
| 시뮬 촉각, 스프링-댐퍼 모델 | VT-Refine [8], 촉각 시뮬레이터 [20, 21] | 식 10과 정렬 절차가 같은 계열입니다(부록 B.2) |
| 시뮬 후학습으로 촉각 정책 개선 | VT-Refine [8] (양손 조립, 시각-촉각, 시뮬 미세조정) | 가장 가까운 선행연구입니다. 차이는 TacCoRL이 대형 VLA(π0.5)를 쓰고 공동학습 + 실데이터 앵커를 둔다는 점입니다 |
| 시뮬-실 공동학습 | Sim-and-human co-training [49], 대규모 행동 모델 [50], 공동학습 [23, 24] | 식 7의 혼합 손실 |
| 합성 시연 확장 | MimicGen [54] | 텔레옵 20개를 200개로(§3.3) |
| VLA 온라인 RL | RL Token [39], RLinf-VLA [40], π0.6* [41], 오프라인 어드밴티지 [42], πRL [63] | 후학습 구현은 RLinf와 πRL의 Flow-SDE PPO(부록 D) |
| 실데이터 앵커 | BC + RL 결합 [56], TD3+BC [57], 시뮬-실 정규화 [28] | 식 8의 β 항 |
| 컨트롤러 정렬 | Tune to learn [52], SIMPLER [51], MIT 임피던스 [59, 60], SPSA [61] | 부록 B.1 |
| 촉각 센서 | FlexiTac [58], GelSight [5], 3D-ViTac [6] | 하드웨어 |

**가까운 대안과의 차이.** VT-Refine [8]은 시각-촉각 정책을 시뮬에서 미세조정하고 실로봇으로 옮기는 같은 문제 설정입니다. TacCoRL이 새로 내세우는 부분은 (1) 사전학습된 VLA에 촉각 토큰을 게이트로 끼우는 인터페이스, (2) 공동학습 초기화 + 실데이터 앵커 RL의 조합입니다. 두 방법을 같은 과제에서 비교한 실험은 논문에 없습니다.

## 2. 코드와 저장소 (GitHub 표)

2026-11-23 확인. 논문 PDF와 프로젝트 페이지에 코드 링크가 없고, 저자 계정(Tac-CoRL)에는 프로젝트 페이지 저장소 하나뿐입니다. **공식 코드는 없습니다.**

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| 공식 코드 | 없음 | — | — | — | 논문, 프로젝트 페이지, GitHub 검색(저자 계정, "TacCoRL", "tac-corl")에서 찾지 못했습니다 |
| 프로젝트 페이지 https://tac-corl.github.io/ | 소스는 [Tac-CoRL/Tac-CoRL.github.io](https://github.com/Tac-CoRL/Tac-CoRL.github.io) (HTML, CSS, JS, 영상) | 없음(GitHub API license null) | 0 | 2026-10-07 (커밋 b339ff88) | 코드가 아닌 웹사이트. 복제하지 않음 |
| [RLinf/RLinf](https://github.com/RLinf/RLinf) | 후학습 구현의 기반(부록 D [62, 63]) | Apache-2.0 | 5,695 | 2026-11-21 | 논문의 π0.5 + 촉각 + 게이트 구현은 포함되어 있지 않다고 봅니다(확인 안 함). 참고용 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0.5 기반 모델 구현·가중치 | Apache-2.0 | 14,743 | 2026-08-24 | 기반 모델 |
| [NVlabs/mimicgen](https://github.com/NVlabs/mimicgen) | 시뮬 시연 확장 [54] | NOASSERTION(GitHub 판별 불가) | 674 | 2026-10-21 | 라이선스 파일 직접 확인 필요 |
| [FlexiTac/FlexiTac](https://github.com/FlexiTac/FlexiTac) | 촉각 센서 [58] | GitHub 판별 없음 | 190 | 2026-11-18 | 하드웨어 설계·펌웨어 |
| [binghao-huang/FlexiTac-IsaacSim-Simulation](https://github.com/binghao-huang/FlexiTac-IsaacSim-Simulation) | FlexiTac의 Isaac Sim 시뮬(관련, 논문이 쓴 시뮬 플랫폼과 같은지는 확인 못 함) | GitHub 판별 없음 | 59 | 2026-04-09 | |
| [NVlabs/vt-refine](https://github.com/NVlabs/vt-refine) | 가장 가까운 선행 [8] | Other | 134 | 2025-10-18 | 코드 이름만 확인, 내용은 보지 않음 |
| [OpenTouch-MIT/opentouch](https://github.com/OpenTouch-MIT/opentouch) | 촉각 이력 분석 [53] | 미확인 | 84 | 미확인 | 이름 검색 결과, 논문 저장소로 확정 못 함 |

사용한 시뮬 엔진은 논문에 명시되어 있지 않습니다(본문과 부록에서 MuJoCo, Isaac 등 이름을 찾지 못했습니다). 병렬 환경 128개, SDF 거리 질의, 촉각 시뮬이라는 서술은 GPU 시뮬을 시사하지만 확정할 수 없습니다.

## 4. 참고: 주장이 기대는 선행 결과

- VT-Refine [8]이 이미 "시각-촉각 정책의 시뮬 미세조정 → 실제 전이"를 양손 조립에서 보였으므로, TacCoRL의 새로움은 VLA 백본과 공동학습·앵커 조합에 있습니다.
- 앵커 항은 오프라인 RL의 BC 정규화(TD3+BC [57])와 같은 형태입니다. 이 형태가 안정성에 주는 효과는 선행 연구와 일관되지만, β의 최적 구간이 과제에 따라 얼마나 좁은지는 이 논문의 과제 하나(Assembly #2)만 보여 줍니다(`03` V9).

# 02. 선행연구와 관련 저장소

TDC의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공개 코드가 있는지를 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.txt` v2의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| TDC의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 과제 프레임으로 축마다 힘/위치를 나눔 | Task Frame Formalism [38] (Bruyninckx & De Schutter 1996) | 3.1절이 이진 분할을 세 모드로 확장하고, 프레임 원점을 엔드이펙터에 고정해 방향만 예측합니다 |
| 같은 분해를 정책이 예측 | Force Policy [34] (Fang 외, RSS 2026) | 2.1절이 "동시 연구"로 인정합니다. 과제 프레임과 축별 제어 모드 + 하이브리드 힘·위치 제어를 예측하고, 실세계 시연에서 프레임을 추출합니다. TDC와 차이는 (1) 저강성 모드 추가, (2) 시뮬 데이터만 사용, (3) 정답 프레임을 시뮬에서 바로 얻는다는 점(2.1절) |
| 위치만 내는 정책 + 등방성 컴플라이언스 | [25] (Yang 2025, 같은 연구실 IROS), [26] UniDoorManip, [27] AutoMate | TDC가 "적응형 컴플라이언스를 희생한다"고 비판하는 계열이며 L/M/H-A 기준선이 이를 대표합니다 |
| 힘·강성을 직접 예측 | ForceMimic [11], ACP [9] (Hou 2025), ForceVLA2 [28], ForceVLA [12], CR-DAgger [32] | 실데이터 필요. ForceVLA, ACP는 기준선(Table 4)이고 "-Real"과 "-Sim" 두 변형을 비교합니다 |
| 포즈 오프셋으로 힘을 암묵 표현 | RDP [10], FoAR [30], FACTR [31], [12], 토크 인식 VLA 설계 [29] (Zhang 외, CoRL 2025) | 2.1절. [29]는 3.3절에서 "새 가중치를 작은 크기로 초기화"하는 방식의 출처로도 인용됩니다 |
| 동역학 무작위화 강화학습 | [19] Watch less feel more, FORGE [20], [21], [37] | 상태 기반 정책에 머무른다고 비판합니다(1절). 시각 + 동역학 이중 무작위화의 어려움은 논문이 주장하고 실험은 하지 않습니다 |
| 실세계 적응·시스템 식별 | [13] 어드미턴스 잔차, [14] SGFT, [15] TRANSIC, [16–18], [35], [36] | 실데이터가 필요하다는 점이 대비 축입니다 |
| 사전학습 VLA | E2VLA [22] (같은 연구실, ICRA 2026), π0 [23], π0.5 [24] | 기반 정책은 E2VLA, 비교 기준선은 π0. 행동 공간 확장의 부작용을 한계 2에서 인정합니다 |
| 시뮬 시연 자동 생성 + 실물 스캔 | Isaac Sim [39], Isaac Lab [40], SAPIEN [41], Genesis [42], ArticuBot [5] | 시연은 규칙 기반 특권 전문가로 만들고(3.2절) 실물을 스캔해 씁니다 |
| 6D 회전 표현 | Zhou 외 2019 [43] | 포즈 출력 6D |

**가까운 대안과의 차이.** (1) Force Policy [34]가 가장 가깝지만 이번 분석에서 읽지 않았고, 두 논문이 서로의 수치로 비교한 실험은 TDC에 없습니다(비교는 서술뿐). (2) 같은 연구실의 [25]는 "위치 정책 + 등방성 컴플라이언스 + 시뮬 학습" 계열이고 TDC는 여기에 방향 정보를 추가합니다. (3) ACP [9]는 위치와 강성을 함께 예측하는 확산 정책이며, TDC의 방향/크기 분해에서 크기까지 예측하는 쪽입니다. TDC는 그 크기가 시뮬에서 전이되지 않는다고 주장하고 ACP-Sim으로 실험합니다(Table 1, SR-ND 5%).

## 2. 코드와 저장소 (GitHub 표)

2026-11-24에 `gh search repos`(검색어 "TDC directional compliance", "Transferable Directional Compliance", "Direction Matters sim-to-real")와 저자 계정 저장소 목록, 프로젝트 페이지 링크를 확인했습니다. **TDC 공식 코드는 찾지 못했습니다.** 프로젝트 페이지(v2 시점)에도 코드 링크가 없습니다.

| 저장소 | 관계 | 라이선스 | 별(★) · 마지막 push | 상태 |
|---|---|---|---|---|
| (공식 코드 없음) | TDC 구현 | — | — | 미확인. 그래서 `04`는 코드 대조 대신 주장 검증(C 항목) |
| [Yifei-Y/project-pages](https://github.com/Yifei-Y/project-pages) | 제1저자 계정의 프로젝트 페이지 모음(`TDC/` 정적 페이지 소스, 영상·BibTeX) | 없음(API license null). 페이지 하단에 템플릿 CC BY-SA 4.0 표기 | 0 · 2026-10-06 | 정적 페이지만. 코드 없음 |
| [hhcaz/e2vla](https://github.com/hhcaz/e2vla) | 기반 정책 E2VLA [22]의 공개 저장소로 보이는 것. 소유자 계정이 "hhcaz", 소속 ZJU로만 표기되어 있고 TDC 저자 계정과 같은지는 확인하지 못했습니다. README가 "Project under refactoring"이라고 합니다 | MIT | 29 · 2025-10-18, 커밋 78b2f52 | 파일 두 개(`models/action_expert.py`, `configs.py`)를 GitHub API로 읽기만 했습니다(복제 안 함, 실행 안 함). `04` §3 |
| [ft-robotic/ForceVLA](https://github.com/ft-robotic/ForceVLA) | 기준선 ForceVLA [12]의 공식 저장소로 알려진 것 | Apache-2.0 | 121 · 2026-01-27 | 이번에는 읽지 않음 |
| [yifan-hou/adaptive_compliance_policy](https://github.com/yifan-hou/adaptive_compliance_policy) | 기준선 ACP [9] 공식 | MIT | 145 · 2026-04-18 | 이번에는 읽지 않음 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 기준선 π0 [23] | Apache-2.0 | 14,743 · 2026-08-24 | — |
| [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab), [isaac-sim/IsaacSim](https://github.com/isaac-sim/IsaacSim) | 시뮬레이터 [39, 40] | API가 NOASSERTION(여러 라이선스 혼합, 파일 미확인) | 8,709 · 2026-11-23 / 4,528 · 2026-11-16 | — |
| [Genesis-Embodied-AI/genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world) | 시뮬레이터 [42] | Apache-2.0 | 30,302 · 2026-11-23 | — |

논문은 v1에서 시연 생성에 Isaac Sim과 Isaac Lab을 썼다고 적었고(v1 부록 K "IsaacLab [21]"), v2는 Isaac Sim, Isaac Lab, SAPIEN, Genesis 네 시뮬레이터를 모두 인용합니다(3.2절). 어느 과제에 어느 시뮬레이터를 썼는지는 v2에 없습니다.


## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **방향은 기하가 정한다** — Task Frame Formalism [38]의 전제입니다. 다만 TFF는 힘 제어 축에서 힘의 크기를 접촉 모델 없이 설정할 수 있다는 고전적 가정 위에 서 있습니다. TDC가 새로 가정하는 부분은 "시뮬에서 정답 프레임을 만들 수 있고 정책이 이미지로 이를 예측한다"입니다.
- **관리 제어기의 힘 축 수렴** — 힘 제어 축에서 K = 0이면 M p̈ + D ṗ = F_ext − F_cmd이므로 정상 상태에서 F_ext = F_cmd입니다(식 2, 3). 이 성질은 환경 강성과 무관하지만 과도 응답은 접근 속도와 환경 강성에 의존합니다(`03` §5 T1).

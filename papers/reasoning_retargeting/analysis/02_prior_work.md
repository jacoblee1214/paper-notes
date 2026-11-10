## 02. 선행연구와 관련 저장소

### 1. 계보

- **운동학 재targeting.** GMR(사람 몸 pose를 맞추는 빠른 IK), PHC, PyRoki, NMR, ReActor. 이들은 몸 자세와 말단 위치를 맞추며 물체 접촉은 직접 다루지 않는다(HOI-Retarget 논문 §II-A).
- **상호작용 인식 운동학.** OmniRetarget(상호작용 메시 변형, HOI-Retarget 논문의 가장 가까운 비교 대상). HOI-Retarget(arXiv:2609.34674, CC BY 4.0)은 객체 좌표계의 접촉 목표를 직접 최소화하는 윈도우 궤적 최적화를 제안했다. 블로그는 이 방법을 **초기 운동학 기준**으로 쓴다(블로그 §3.1).
- **물리 기반 재targeting.** DynaRetarget(SBTO), SPIDER(arXiv:2511.09484, 샘플링 기반 물리 최적화). 블로그는 SPIDER를 G1 + Dex3에 맞춰 다시 돌린 것을 기준선으로 쓴다(블로그 §3.1, "HOI-Retarget + SPIDER").
- **전신 제어기.** SONIC(arXiv:2511.07820). 블로그는 다리·허리 제어에 쓰고, 에이전트는 관절 출력을 보강하거나 대체할 수 있다.
- **LLM 에이전트 + 시뮬레이터 피드백.** 블로그 자체가 주요 출처다. 모델은 GPT-6 Astra(OpenAI 공개 발표 글만 인용), 실행 환경은 Codex(Apache-2.0). 이 조합을 쓴 심사 논문은 아직 없다.
- 블로그는 공개 글 두 개를 관련 연구로 인정한다(Xiao Hu의 DexGPT 데모와 코드, Haoyang Weng의 Steerable Physical Interaction Competence 글).

블로그 §1이 나누는 세 축은 (a) 사람 자세 보존(GMR류), (b) 물리 샘플링(SPIDER류), (c) 의미 보존 + 에이전트 루프(이 블로그)다. (c)는 기존 방법을 대체한다기보다 (a)를 입력으로 받아 (b)가 실패하는 접촉 전략을 고치는 구조다.

### 2. 구성 요소별 출처와 라이선스

| 구성 요소 | 출처 | 라이선스 (확인 방법) |
|---|---|---|
| 초기 운동학 기준 | HOI-Retarget, arXiv:2609.34674 | CC BY 4.0 (arXiv abs 페이지). 공식 코드 leggedrobotics/hoi-retarget |
| 물리 기준선 | SPIDER, arXiv:2511.09484 | 논문 CC BY 4.0. 코드 README 배지 CC BY-NC 4.0 (GitHub API는 NOASSERTION) |
| 전신 제어기 | SONIC, arXiv:2511.07820 | CC BY-SA 4.0 (arXiv abs). 코드는 NVlabs/GR00T-WholeBodyControl (API NOASSERTION) |
| 에이전트 프레임워크 | OpenAI Codex | Apache-2.0 (openai/codex, GitHub API) |
| 에이전트 모델 | GPT-6 Astra (OpenAI 발표 글) | 비공개 모델. 라이선스 해당 없음 |
| 시뮬레이터 | MuJoCo (Todorov et al., IROS 2012) | 공식 사이트 (라이선스 확인 생략) |
| G1 모델 | unitreerobotics/unitree_mujoco | BSD-3-Clause (GitHub API) |
| 사람 동작 데이터 | HUMOTO, OMOMO | 블로그가 인용. 원자료 공지(`repo/site/architecture-trace/DATA-NOTICE.txt`)는 상위 모션 데이터를 CC BY-NC-SA 4.0으로 표기한다 |
| 블로그 원자료 | narayananpp/semantic-humanoid-retargeting-blog | 라이선스 파일 없음 (GitHub API 404). 3절과 README 참조 |

### 3. GitHub 저장소 표

| 저장소 | 역할 | 라이선스 | 최근 push | 이 폴더에서 한 일 |
|---|---|---|---|---|
| [narayananpp/semantic-humanoid-retargeting-blog](https://github.com/narayananpp/semantic-humanoid-retargeting-blog) | 블로그 소스 + 과제별 원자료·트레이스·export 도구 | 없음 (LICENSE 없음) | 2026-11-06 (커밋 `84cbe2253`) | `repo/`에 shallow clone, 정적 읽기와 수치 재계산 (04 참조) |
| [facebookresearch/spider](https://github.com/facebookresearch/spider) | SPIDER 공식 구현 (기준선의 원 방법) | README: CC BY-NC 4.0 | 2026-09-22 | 정적 읽기만. GitHub 표시 NOASSERTION |
| [leggedrobotics/hoi-retarget](https://github.com/leggedrobotics/hoi-retarget) | HOI-Retarget 공식 구현과 데이터 | NOASSERTION (논문 CC BY 4.0) | 2026-09-30 | 정적 확인만. 클론하지 않음 |
| [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) | SONIC 제어기 코드 (블로그 SONIC 링크는 프로젝트 페이지 nvlabs.github.io/GEAR-SONIC) | NOASSERTION | 2026-11-03 | 정적 확인만 |
| [YanjieZe/GMR](https://github.com/YanjieZe/GMR) | GMR 운동학 재targeting | MIT | 2026-10-22 | 블로그 kinematic 비교 대상. 정적 확인만 |
| [openai/codex](https://github.com/openai/codex) | 블로그가 쓴 에이전트 환경 | Apache-2.0 | 2026-11-10 | 정적 확인만. 블로그 모델(GPT-6 Astra)과 무관한 소프트웨어 |
| [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) | G1 MuJoCo 모델 | BSD-3-Clause | 2026-09-07 | 정적 확인만 |
| [Hu-xiao-max/dexgpt](https://github.com/Hu-xiao-max/dexgpt) | 블로그가 관련 공개 작업으로 인용 | 없음 | 2026-09-11 | 정적 확인만 (손 재targeting, 이 블로그와 직접 관계 없음) |

`gh search repos "GEAR-SONIC"`에서는 해당 저장소가 나오지 않았다. 블로그 링크는 프로젝트 페이지이며 코드 저장소는 GR00T-WholeBodyControl이다.

### 4. 블로그 원자료 저장소의 내용 (repo/)

- `site/assets/analysis/*.json`: 과제별 추적 오차, LLM 판정, 토큰 사용량, 최선값 곡선, 매칭 시간 곡선, SPIDER 재실행 결과.
- `site/experience-story.json`: 경험 실험 10과제의 without/with 곡선과 과제별 지표.
- `site/trace-explorer/`: 에이전트 트레이스(도구 호출 내용은 가려짐, 메시지와 시도 기록은 공개).
- `tools/*.py`, `build.py`: 원자료를 웹 페이지 데이터로 바꾸는 export와 그림 스크립트.
- 없는 것: 에이전트 harness, 프롬프트 원문, 생성된 제어 코드, 심판 프롬프트 원문(해시만 있음), `object_path_v1` 지표 구현.

### 5. 이 저장소의 다른 논문과의 관계

| 폴더 | 논문 | 이 블로그와 맞닿는 지점 |
|---|---|---|
| `gnr/` | Generative Neural Retargeting (arXiv:2610.12440) | 샘플링 MPC가 푼 궤적에 생성 모델로 잔차를 더해 보정한다. "기준 궤적 + 시뮬레이터 피드백" 구조를 쓰지만 학습 기반이다 |
| `flashdexretarget/` | FlashDexRetarget (arXiv:2610.01849) | 사람 손-물체 시연을 로봇 손 궤적으로 옮긴다. 접촉·물체 목표 문제를 다루지만 손 중심이다 |
| `chord/` | CHORD (arXiv:2607.00033) | 접촉 법선에서 만든 wrench 보상을 RL에 넣는다. 블로그의 접촉 목표 정의와 대비되는 학습 기반 접근이다 |
| `cross_embodiment_wm/` | Scaling Cross-Embodiment World Models (arXiv:2511.01177) | 여러 몸체를 한 세계 모델로 다룬다. 블로그의 "로봇이 달라도 의미를 보존"과 같은 관심사다 |
| `dreamx/` | DreamX (arXiv:2611.05377) | 영상 생성 백본의 latent action으로 사람 시연을 로봇 명령으로 옮긴다. 블로그의 텍스트 명령(A1)과 대비된다 |
| `motion_latent_action/` | Motion-Focused Latent Action (arXiv:2606.18955) | 사람 egocentric 영상에서 교차 embodiment 학습을 한다. 블로그와 직접 비교 대상은 아니다 |
| `warp_retargeting/` | WARP (arXiv:2606.29940) | 이 폴더에서는 읽지 않았다. 필요하면 arXiv 논문만 참조한다 |

### 6. 선행연구 관점의 해석

- 블로그가 말하는 "에이전트가 물체 궤적을 목표로 삼는다"는 점은 HOI-Retarget의 접촉 목표 정의와 같은 방향이다. 블로그는 HOI-Retarget을 인용하지만, 두 논문의 지표(접촉 위치 gap 0.5 cm, 객체 추적 ADD 12–43 cm)가 달라 직접 비교하지 않는다.
- SPIDER는 블로그의 객체 추적 오차를 원래 쓰지 않는다. 기준선은 블로그가 정의한 지표로 다시 측정한 값이다.
- 블로그의 경험 효과 주장은 에이전트 메모리·요약 재사용 계열과 비교할 수 있지만, 블로그는 그쪽 문헌을 인용하지 않았다.

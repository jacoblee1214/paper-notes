# Paper Notes

로봇 학습(VLA, 제어, 물리 기반 학습) 논문을 깊게 읽고, 공개된 수치를 재계산하거나 코드와 대조하며 정리한 노트입니다.

| 날짜 | 글 | 요약 |
|---|---|---|
| 2026-10-29 | [손 전체 접점의 실시간 힘 조절 읽기: 촉각 없이 잡는다는 것, 그리고 시험이 실제로 잰 것](papers/reactive_grasp/) | 물체 메쉬와 관절각만으로 손 전체 링크의 접점을 구하고 매 주기 접촉력 QP를 다시 푸는 파지 제어기를, 그림 수치를 재계산하고 QP 구조를 toy로 뜯어 보며 같은 연구실 공개 코드와 대조해 읽었습니다. |
| 2026-10-29 | [ThinkWorld 읽기: 이득의 대부분은 '추론'이 아니라 미래 예측 보조 손실에서 나왔다](papers/thinkworld/) | 손목 움직임 토큰으로 미래를 상상하고 그 일치로 VLA를 강화하는 ThinkWorld를, 표를 재계산하고 공개 리더보드와 대조하며 읽었습니다. |
| 2026-10-29 | [RoboSimStudio 읽기: 쓸 만한 Real2Sim 도구, 그런데 스펀지는 손에 붙어 있었다](papers/robosimstudio/) | 사람과 코딩 에이전트가 함께 시뮬레이션 물성을 고치는 Real2Sim 도구 RoboSimStudio를, 공식 저장소와 대조하고 데모 수치를 다시 계산하며 읽었습니다. |
| 2026-10-29 | [PaceVLA 읽기: 빨라진 VLA의 우위는 속도가 아니라 덜 실패하는 데 있다](papers/pacevla/) | VLA가 국면마다 행동을 합쳐 시연보다 빠르게 움직이게 하는 PaceVLA를, 표를 재계산하고 지표와 추적 toy로 다시 따져 보며 읽었습니다. |
| 2026-10-29 | [GPT-as-Tactile-Policy 읽고 직접 돌려 보기: 촉각을 이해하는가, 증류된 규칙을 따르는가](papers/gpt6_astra_tactile/) | GPT-6-Astra를 촉각 정책으로 쓴 논문을 표 재계산으로 검토하고, 같은 구조를 재구현해 UniVTAC 폐루프에서 2×2 실험과 반사실 촉각 실험으로 메커니즘을 떼어 봤습니다. |
| 2026-10-29 | [Dust 읽기: 역전파 없이 Transformer를 학습한다, 그런데 무엇과 비교했나](papers/dust/) | 활성값을 토큰마다 흔들어 forward만으로 gradient를 추정하는 Dust를, 부록 표 재계산, 연산량 환산, 공식 코드 대조, numpy toy로 읽었습니다. |
| 2026-10-29 | [AW-0 읽기: 코드로 계획하는 World Action Model, 그런데 이득은 어디서 오는가](papers/agentic_wam/) | 코드·미래 영상·행동을 한 번에 denoise하는 AW-0를 표 재계산, RoboTwin 공식 리더보드 대조, 공개 코드 대조로 읽었습니다. |
| 2026-10-28 | [ZeroDex 읽기: 시연 없이 dexterous 손을 움직이는 다시점 VLM 그라운딩, 숫자는 어디서 나왔나](papers/zerodex/) | VLM의 2D 키포인트를 여러 보정 시점의 합의로 3D에 올려 dexterous 손을 zero-shot으로 움직이는 ZeroDex를, 표를 재계산하고 핵심 식을 toy로 구현하며 v1과 대조해 읽었습니다. |
| 2026-10-28 | [RAG-Anything 읽기: 표를 검산하고 코드를 열어 보니](papers/rag_anything/) | 그림·표·수식까지 지식 그래프에 넣는 멀티모달 문서 RAG인 RAG-Anything을, 표를 다시 계산하고 공개 코드와 대조하며 읽었습니다. |
| 2026-10-28 | [Hydra-0 읽기: 숫자는 맞는데, 무엇과 비교했나](papers/hydra0/) | 로봇 행동을 픽셀 궤적으로 표현하는 NVIDIA의 world model Hydra-0을, 공개된 수치를 모두 재계산하며 읽었습니다. |
| 2026-10-28 | [GraspSense 읽기: 힘 지도로 고르고, 같은 지도로 판정하면](papers/graspsense/) | 물체 표면별 허용 하중 지도로 dexterous 파지를 고르고 손가락 강성을 정하는 GraspSense를, 표를 재계산하고 저자 저장소와 대조하며 읽었습니다. |
| 2026-10-28 | [FlashDexRetarget 읽기: 시연 수백 개를 정책 하나로, 그런데 무엇이 성능을 만들었나](papers/flashdexretarget/) | 사람 손-물체 시연 모음을 RL 정책 하나로 로봇 궤적으로 바꾸는 FlashDexRetarget을, 표를 재계산하고 공개 코드와 대조하며 읽었습니다. |
| 2026-10-28 | [Do as I Do 읽기: SAM 3D 추적은 사실 EMA 필터다](papers/do_as_i_do/) | 일상 영상에서 다지 로봇손 데이터를 만드는 Do as I Do를 코드까지 대조해 읽고, 핵심 추적 기법의 정체를 유도했습니다. |
| 2026-10-28 | [DexScrew 읽기: 촉각은 평균이 아니라 꼬리를 바꾼다](papers/dexscrew/) | 불완전한 시뮬레이션으로 너트 조이기와 드라이버 돌리기를 배우는 DexScrew를, 표 수치 재계산과 코드 대조까지 해서 읽었습니다. |
| 2026-10-28 | [DexPolicy 읽기: 잡음을 줄이면 나아진다, 그런데 무엇과 비교했나](papers/dexpolicy/) | PPO의 행동 잡음 σ를 학습 step에 따라 줄이는 DexPolicy를, 표를 재계산하고 공개 코드·기반 환경과 대조하며 toy로 다시 따져 읽었습니다. |
| 2026-10-28 | [DDPG 다시 읽기: 표적망은 확실하고, 나머지는 구현마다 다르다](papers/ddpg/) | 연속 제어 off-policy RL의 출발점인 DDPG를 Table 1 재계산, 주요 구현 4종 대조, numpy toy로 다시 읽었습니다. |
| 2026-10-28 | [Contact-Grounded Policy 읽기: 촉각을 예측해서 제어기 목표로 바꾸면 무엇이 달라지나](papers/contact_grounded_policy/) | 다지 손 visuotactile 정책 CGP를, 표를 재계산하고 핵심 사상을 물리식과 3-DoF 손가락 toy로 다시 유도하며 읽었습니다. |
| 2026-10-28 | [Chronos Policy 읽기: 시간을 알려 주면 정책이 강해진다, 그런데 어디까지가 시간 덕인가](papers/chronospolicy/) | 관측 캡처 시각과 행동 실행 시각을 RoPE로 넣는 Chronos Policy를, 논문 표와 프로젝트 페이지 원값을 대조하고 시드별 성공 횟수까지 역산하며 읽었습니다. |
| 2026-10-28 | [AlphaNIST 읽기: 이진 보상 하나로 조립 보드 20개 과제, 그런데 98.2%는 어디서 시작한 성공률인가](papers/alphanist/) | 이진 성공 보상과 PPO 정책 하나로 NIST 조립 보드 20개 과제를 푸는 AlphaNIST를, 그림을 픽셀로 판독하고 SGS·OmniReset 원본과 대조하며 읽었습니다. |

각 폴더에는 블로그 글(`index.md`), 통합 리포트 PDF, 분석 문서, 검증 스크립트가 있습니다. 논문 그림은 원 논문 라이선스가 허용하는 경우에만 실었습니다.

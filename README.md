# Paper Notes

로봇 학습(VLA, 제어, 물리 기반 학습) 논문을 깊게 읽고, 공개된 수치를 재계산하거나 코드와 대조하며 정리한 노트입니다.

| 날짜 | 글 | 요약 |
|---|---|---|
| 2026-10-28 | [ZeroDex 읽기: 시연 없이 dexterous 손을 움직이는 다시점 VLM 그라운딩, 숫자는 어디서 나왔나](papers/zerodex/) | VLM의 2D 키포인트를 여러 보정 시점의 합의로 3D에 올려 dexterous 손을 zero-shot으로 움직이는 ZeroDex를, 표를 재계산하고 핵심 식을 toy로 구현하며 v1과 대조해 읽었습니다. |
| 2026-10-28 | [RAG-Anything 읽기: 표를 검산하고 코드를 열어 보니](papers/rag_anything/) | 그림·표·수식까지 지식 그래프에 넣는 멀티모달 문서 RAG인 RAG-Anything을, 표를 다시 계산하고 공개 코드와 대조하며 읽었습니다. |
| 2026-10-28 | [Hydra-0 읽기: 숫자는 맞는데, 무엇과 비교했나](papers/hydra0/) | 로봇 행동을 픽셀 궤적으로 표현하는 NVIDIA의 world model Hydra-0을, 공개된 수치를 모두 재계산하며 읽었습니다. |
| 2026-10-28 | [GraspSense 읽기: 힘 지도로 고르고, 같은 지도로 판정하면](papers/graspsense/) | 물체 표면별 허용 하중 지도로 dexterous 파지를 고르고 손가락 강성을 정하는 GraspSense를, 표를 재계산하고 저자 저장소와 대조하며 읽었습니다. |
| 2026-10-28 | [FlashDexRetarget 읽기: 시연 수백 개를 정책 하나로, 그런데 무엇이 성능을 만들었나](papers/flashdexretarget/) | 사람 손-물체 시연 모음을 RL 정책 하나로 로봇 궤적으로 바꾸는 FlashDexRetarget을, 표를 재계산하고 공개 코드와 대조하며 읽었습니다. |
| 2026-10-28 | [Do as I Do 읽기: SAM 3D 추적은 사실 EMA 필터다](papers/do_as_i_do/) | 일상 영상에서 다지 로봇손 데이터를 만드는 Do as I Do를 코드까지 대조해 읽고, 핵심 추적 기법의 정체를 유도했습니다. |
| 2026-10-28 | [DexScrew 읽기: 촉각은 평균이 아니라 꼬리를 바꾼다](papers/dexscrew/) | 불완전한 시뮬레이션으로 너트 조이기와 드라이버 돌리기를 배우는 DexScrew를, 표 수치 재계산과 코드 대조까지 해서 읽었습니다. |
| 2026-10-28 | [DDPG 다시 읽기: 표적망은 확실하고, 나머지는 구현마다 다르다](papers/ddpg/) | 연속 제어 off-policy RL의 출발점인 DDPG를 Table 1 재계산, 주요 구현 4종 대조, numpy toy로 다시 읽었습니다. |
| 2026-10-28 | [Contact-Grounded Policy 읽기: 촉각을 예측해서 제어기 목표로 바꾸면 무엇이 달라지나](papers/contact_grounded_policy/) | 다지 손 visuotactile 정책 CGP를, 표를 재계산하고 핵심 사상을 물리식과 3-DoF 손가락 toy로 다시 유도하며 읽었습니다. |

각 폴더에는 블로그 글(`index.md`), 통합 리포트 PDF, 분석 문서, 검증 스크립트가 있습니다. 논문 그림은 원 논문 라이선스가 허용하는 경우에만 실었습니다.

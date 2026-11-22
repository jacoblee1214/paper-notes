# 02. 선행연구와 자료 (Ego2Robot)

## 1. 계보: 이 논문이 서 있는 자리

Ego2Robot의 새로움은 방법보다 **규모와 평가 틀**입니다. 손 → 그리퍼 리타게팅과 로봇 합성 자체는 선행(Phantom, EgoMimic)이 보였고, 이 논문은 "그것을 VLA 사전학습 규모(18,561 h, 15개 형태)로 키우면 OOD 일반화에 도움이 되는가"를 묻습니다(§1, §2). 논문이 직접 언급하는 갈래는 다음과 같습니다.

| 갈래 | 대표 | 이 논문과의 관계 (논문 §2 기준) |
|---|---|---|
| 사람 영상 → 로봇 형식 데이터(리타게팅 + 렌더링) | Phantom [14], 사람 팔을 로봇 팔로 바꾸는 계열 [30], EgoMimic [15](마스킹) | 직접 선행. "소규모 또는 과제 하나에만 연구됨"이라고 선을 긋고 18,561 h로 확장 |
| 사람/자기 중심 영상과 로봇 데이터 공동학습 | EgoMimic, ViTRA [29] 등 [26–29] | 공동학습 비율(1:3, 3:1, 1:1)을 변수로 둠. 행동은 로봇 렌더링으로 같은 공간에 맞춤 |
| 로봇 ↔ 로봇 시각 증강 | RoviAug [33], Mirage [34], OXE-AugE [35] | 인페인팅·이미지 편집으로 로봇 사이 격차를 메움. Ego2Robot은 더 큰 사람 → 로봇 격차를 다룬다고 주장 |
| 사람 영상으로 시각 사전학습·보상·모션 사전·점 추적 | [19–25] | 로봇 데이터 없이는 임베디먼트 격차를 메우지 못한다고 정리 |
| 대규모 로봇 데이터 | Open X-Embodiment [6], DROID [7], AgiBot World [8] | 비교 기준(DROID + AgiBot + InternData ≈ 6,565 h) |
| 분리형 일반화 벤치마크 | Colosseum [43], LIBERO-Plus [44], LIBERO-PRO [45] | 변형 축을 분리한다는 아이디어는 선행. 이 논문은 양팔·교차 형태를 지원하는 RoboTwin 2.0 확장으로 차별화 |
| 양팔·형태 벤치마크 | RoboTwin 2.0 [16], EBench [46] | 평가의 바탕. EBench는 "Preprint coming soon"으로 인용됨 |

## 2. 구성 요소별 출처 (공식 코드 README와 논문 부록 기준)

| 단계 | 사용한 것 | 출처 | 라이선스 (GitHub 확인 2026-11-22) |
|---|---|---|---|
| 손 자세(주석 없는 영상) | WiLoR + AnyHand 백엔드, Dyn-HaMR 시간 최적화 | 논문 [47–49], 코드 `models/` | WiLoR: NOASSERTION, AnyHand: CC0-1.0, Dyn-HaMR: MIT |
| 팔 분할 | SAM 3 (텍스트 프롬프트 "person") | [51] | NOASSERTION (SAM 라이선스) |
| 손 제거 | ProPainter | [52] | NOASSERTION |
| 깊이·카메라 자세 | Depth Anything 3 | 코드 `da3_wrapper.py`, 부록 A.4 | Apache-2.0 |
| IK | MuJoCo + mink(quadprog) | [53, 54] | mink: Apache-2.0 |
| 로봇 모델 | MuJoCo Menagerie 등 16종 XML | 코드 README | 업스트림 라이선스 유지(다양) |
| 하위 과제 분할·L3 | Qwen3.5 (DashScope API) | [50], 코드 `subtask_processor.py` | 원격 API 호출 |
| 원천 영상 | ANT(자체), EgoDex, ViTRA, EgoVerse | [12, 13, 29] | EgoDex: NOASSERTION, ViTRA: MIT, EgoVerse: MIT |
| 평가 | RoboTwin 2.0 포크 | [16] | MIT |
| 정책 | Qwen3.5-4B + DiT, π0.5 비교 | 부록 C, F.1 | 학습 코드 미공개 |

## 4. GitHub와 자료 표 (확인 2026-11-22, `gh api`)

| 자료 | 링크 | 라이선스 | 별 / 마지막 push | 비고 |
|---|---|---|---|---|
| **공식 코드 + 프로젝트 페이지** | https://github.com/www-Ye/ego2robot_blog | **없음**(`LICENSE` 파일 없음, GitHub license null). `evaluation/robotwin/LICENSE`는 RoboTwin의 MIT | 12 / 2026-11-01 | 커밋 1개 `244a4ee` (2026-11-02, "Add Ego2Robot synthesis and RoboTwin evaluation code"). 파일 1,492개, clone 112 MB(.git), 로봇 모델 360 MB. `repo/` 얕은 clone, 정적 읽기만 |
| 프로젝트 페이지 | https://www-ye.github.io/ego2robot_blog/ | - | - | 같은 repo의 `index.html`. 표 1·2와 같은 수치, 영상 12.6 MB |
| RoboTwin 2.0 | https://github.com/RoboTwin-Platform/RoboTwin | MIT | 3,074 / 2026-09-24 | 평가의 업스트림 |
| EBench | https://github.com/InternRobotics/EBench | 미확인 | - | 논문이 "Preprint coming soon"으로 인용. 가중치·데이터 받지 않음 |
| Qwen-RobotManip | https://github.com/QwenLM/Qwen-RobotManip | 없음 | 189 / 2026-06-29 | 후속 기술 보고서(arXiv:2606.17846)의 저장소. 이 논문의 데이터를 사전학습에 사용 |
| SAM 3 | https://github.com/facebookresearch/sam3 | NOASSERTION | 12,373 / 2026-11-17 | 파이프라인 필수 |
| ProPainter | https://github.com/sczhou/ProPainter | NOASSERTION | 7,092 / 2026-10-27 | 파이프라인 필수 |
| WiLoR | https://github.com/rolpotamias/WiLoR | NOASSERTION | 746 / 2026-04-07 | 원시 영상 경로 |
| Dyn-HaMR | https://github.com/ZhengdiYu/Dyn-HaMR | MIT | 331 / 2026-03-12 | 별도 Python 환경 |
| AnyHand | https://github.com/chen-si-cs/AnyHand | CC0-1.0 | 95 / 2026-11-03 | 원시 영상 경로 |
| Depth Anything 3 | https://github.com/ByteDance-Seed/Depth-Anything-3 | Apache-2.0 | 6,723 / 2026-07-27 | 모든 `base_*` 단계 |
| mink | https://github.com/kevinzakka/mink | Apache-2.0 | 1,619 / 2026-11-19 | IK |
| EgoVerse | https://github.com/GaTech-RL2/EgoVerse | MIT | 586 / 2026-11-22 | 원천 데이터 954 h |
| ViTRA | https://github.com/microsoft/VITRA | MIT | 519 / 2026-06-12 | 원천 데이터 249 h |
| EgoDex | https://github.com/apple-aiml-research/ml-egodex | NOASSERTION | 418 / 2026-09-11 | 원천 데이터 732 h |
| 합성 데이터 18,561 h, 가중치 | 저장소·페이지에서 링크를 찾지 못함 | - | - | 공개 여부 미확인. 다운로드하지 않았고 크기도 알 수 없음 |
| 동명 다른 저장소 | `msunbot/ego2robot`(공장 에피소드, 2025), `BAp3r/Ego2Robot`, `shaohua-pan/ego2robot` 등 | - | - | 이 논문과 무관(공식 README의 clone 주소는 `www-Ye/ego2robot_blog`). 검토하지 않음 |

## 5. 관련 질문 (이 분석에서 남긴 것)

- 18,561 h의 정의: "처리 뒤" 시간인지 원본 영상 시간(약 1,940 h × 형태 수)을 품질 선별로 줄인 시간인지, 그리고 형태별 시간이 같은지(`03` V11).
- 소스별 샘플링 가중치(그림 7)의 값: 그림에 비율만 있고 수치가 없음.
- 합성 데이터와 베이스 탐색 결과의 공개 계획.

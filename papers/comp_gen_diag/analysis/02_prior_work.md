# 02. 선행연구와 관련 저장소

이 문서는 이 논문의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드가 있는지를 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References, v2 기준)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| 이 논문의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 체계적 일반화 문제 설정 | Lake & Baroni 2018 [12], Yagcioglu 외 2024 [13] | 언어·시퀀스 모델에서 알려진 원시 요소의 새 조합 실패를 보인 계열. 이 논문은 이를 로봇 지시 공간으로 옮김 |
| 조합 일반화 벤치마크 | VIMA-Bench [11], ClevrSkills [14], RoboHiMan [15] | 보류 조합을 평가하는 벤치마크. 이 논문은 "무엇을 학습에 넣어야 하는가"라는 데이터 쪽 질문을 묻는다고 선을 그음(§2) |
| 데이터 수집 비용 줄이기 | MimicGen [16] (객체 중심 하위 과제 변환), Gao 외 2024 [17] (환경 요인을 조합 변화로 효율 수집) | [17]을 "동기 면에서 가장 가까운" 연구로 인용. 이 논문은 환경 요인이 아니라 순차 지시 인자를 다룸 |
| 직교 배열, 조합 테스트 | Kacker 외 1991 [18] (Taguchi 직교 배열), Hedayat 외 [19], Kuhn 외 [20] | 전수 곱 집합 없이 저차 상호작용을 덮는 실험 설계. 직교 16과제 설계(그림 4)의 근거. 조합 테스트의 경험칙은 소프트웨어 결함이 소수 인자의 상호작용에서 생긴다는 것이고, 이 논문의 "쌍 커버리지"와 같은 구조 |
| 계층·모듈러 제어 | Options/옵션 크리틱 [21], HRL [22], TAMP [23], SayCan [1], π0.5 [2], Hi Robot [24], Gemini Robotics [3] | 단계 지시자 `σ(z)`를 쓰는 "단계별 모듈러 정책"(§4.1)의 원형. v1은 π0.5를 이 계열로 적었고 v2는 SayCan과 Gemini Robotics만 적음 |
| 정책 구조 | Transformer [8], Diffusion Policy [10], 흐름 매칭(π0.5 [2] 등), FAST [9], Octo [4], OpenVLA [7], RT-2 [6], Interleave-VLA [5] | 일반 정책 계열. 이 논문은 이 중 어느 것도 쓰지 않고 작은 흐름 매칭 정책을 처음부터 학습 |
| 영상 인코더와 융합 | DINOv2 [26], BLIP-2의 Q-Former [27] | 동결 DINOv2 + 지시 토큰을 질의로 쓰는 Q-Former형 교차 주의 (부록 B.2) |
| 과제와 시뮬레이터 | robomimic [25] (§5.1), Isaac Lab [28] (사람 시연, 부록 C) | 공개 코드는 robosuite 기반 맞춤 환경(`robot_envs`)을 씀(`04` D12) |

**가까운 대안과의 차이.** 이 논문은 새 알고리즘이 아니라 **진단 연구**입니다. 정책은 표준 구성이고, 기여는 (1) 일반화 격차의 세 항 분해, (2) 직교 설계 기반 학습 집합이 전수보다 효율적일 수 있다는 실험, (3) 사전학습이 희소할 때 부족한 것이 "조향"이라는 해석, (4) 의존 지시에서 의존 쌍을 덮어야 한다는 관찰입니다. 가장 가까운 선행은 [17]이지만 요인의 종류가 다릅니다. 이 논문이 인용하지 않은 영역(언어 모델의 조합 일반화 이론, 분포 이동 하의 학습 이론 등)과의 정밀 비교는 하지 않았습니다.

## 2. 코드와 저장소 (GitHub 표)

조회일 2026-11-26. 별 수, 최종 push는 `gh api`, 라이선스는 API의 `license`와 저장소 안 LICENSE 파일 유무로 확인했습니다.

| 저장소 | 관계 | 라이선스 | 별 / 포크 | 최종 push | 대조 기준 | 상태 |
|---|---|---|---|---|---|---|
| [YixiaoWang7/Diagnosing_Compositional_Generalization_Robot](https://github.com/YixiaoWang7/Diagnosing_Compositional_Generalization_Robot) | 공식 코드 (RoboPolicy 라이브러리 + `cg` 응용) | 라이선스 파일과 패키지 메타데이터 없음 (API license: null) | 8 / 2 | 2026-10-13 | `e29c44474495b341d19636bbf90f3cf0fcb1d2d2` (2026-10-13) | 정적 읽기만 (`repo/`) |
| [YixiaoWang7/Diagnosing_Compositional_Generalization_Robot_Page](https://github.com/YixiaoWang7/Diagnosing_Compositional_Generalization_Robot_Page) | 프로젝트 페이지 소스. 집계 CSV와 JSON 포함 | 라이선스 파일 없음 | 0 / 0 | 2026-11-11 | main (집계 파일만 받음, 약 3 MB) | 수치 재계산에 사용 |
| [YixiaoWang7/robot_envs](https://github.com/YixiaoWang7/robot_envs) | 시뮬레이터. 공식 README가 `c7e06a0784853d74db57ebf78d7a66fa40c2df99`(2026-05-24)를 고정 | 라이선스 파일 없음 (API license: null) | 1 / 0 | 2026-10-05 | 받지 않음 | 정적 확인 없음 |
| Hugging Face 데이터셋 `yixiiiiii/cg_l4_1s` (PP), `cg_l5` (PPP), `cg_l4_2s` (2S-PP) | 시연 데이터 | 카드에 라이선스 표기 없음 | 다운로드 640–1,173회 | 2026-10-19 갱신 | 받지 않음 | 메타데이터만 (아래) |
| [NVlabs/mimicgen](https://github.com/NVlabs/mimicgen), [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic) | 선행 도구 (공식 코드의 환경 일부가 mimicgen 포크를 포함) | NOASSERTION / MIT | 676 / 1,610 | 2026-10-21 / 2026-08-09 | - | 참고 |

데이터셋 크기(HF API 메타데이터의 `usedStorage`)는 PP 4.7 GB, PPP 20.7 GB, 2S-PP 17.2 GB로 합계 약 42.6 GB입니다(받지 않음). 파일 목록에서 shard(hdf5) 수는 PP 1,600개(16과제), PPP 3,840개(64과제), 2S-PP 2,880개(144과제)입니다. shard당 시연 5개로 가정하면 과제당 500, 300, 100개로 논문 부록 B.1의 시연 풀(8,000 / 19,200 / 14,400)과 같습니다(`03` K3). 사전학습 체크포인트는 공개 안내에 "아직 없음"이라고 적혀 있습니다.

프로젝트 페이지: <https://yixiaowang7.github.io/Diagnosing_Compositional_Generalization_Robot_Page/> (CoRL 2026 스포트라이트 표기, 3분 영상, Q1–Q3 인터랙티브 보기 포함).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- 쌍 커버리지가 OOD를 설명한다는 주장은 조합 테스트의 경험칙(소수 인자 상호작용이 결함 대부분을 일으킨다)과 구조가 같습니다. 다만 테스트에서는 결함 탐지 여부를 직접 확인하고, 학습에서는 일반화가 일어나는지를 함수 근사기를 거쳐 간접적으로 보므로, 같은 경험칙이 학습에도 성립한다는 보장은 이 논문이 실험으로 부분적으로만 보입니다(`03` V6).
- 직교 배열의 최소 크기는 강도 2에서 q²행입니다. 인자 수가 늘어도 이 최소는 q²로 같고 전수 곱은 qⁿ으로 늘어나므로, 인자 수가 늘수록 "필요한 비율"은 급격히 작아집니다(3인자 1/4, 4인자 1/16, 5인자 1/64). 반면 상호작용이 3차까지 필요하면 최소가 q³이 되어 4인자에서야 1/4입니다(`03` K1). 논문도 "더 긴 과제는 고차 상호작용 커버리지가 필요할 수 있다"고 §5.2에 적었습니다.

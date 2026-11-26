# 02. 선행연구와 계보

## 1. 계보

RoboFollow는 "성공률이 높아도 언어를 쓰는지는 알 수 없다"는 문제를 **벤치마크 설계**로 다룹니다. 가까운 계보는 다섯 갈래입니다(참고문헌 번호는 논문 기준).

1. **언어 조건 로봇 정책과 VLA.** RT-2 [1], OpenVLA [2], π0 [3], π0.5 [11], GR00T N1 [10]. 이어서 이산 행동 토큰에서 flow matching 머리로, 시각 CoT를 얹은 구조로 발전했고(§2.1), 이 논문은 그 구조의 평가 대상으로만 쓰입니다. WAM 쪽은 GR-1 [4], Motus [5], FAST-WAM [48]입니다.
2. **기존 벤치마크의 언어 의존성 점검.** LIBERO [41]는 기준 벤치마크이고, LIBERO-PRO [8]와 LIBERO-Plus [9]는 정책이 언어 의미 대신 시각 패턴을 쓴다는 근거를 보였습니다. 논문은 이를 "시험 시점의 섭동"으로 보고, RoboFollow는 "학습 시점에 과제 모호성을 만든다"는 점이 다르다고 위치를 정합니다(§2.2). "Seeing to act, prompting to specify" [7]는 시각과 언어의 베이즈 분해로 같은 문제를 학습 쪽에서 다룹니다.
3. **언어 사용을 강화하려는 방법.** LangForce [49](잠재 행동 질의로 텍스트-행동 상관을 강화), 안정적 언어 유도(Stable language guidance [50], 논문은 CFG로 부름), QA 공동학습(망각 방지). 표 2는 이 셋과 더 강한 VLM(Qwen3-VL)을 Scene 2에서 시험합니다.
4. **일반화 축으로 나눈 평가 틀.** CALVIN [34], VIMA [35], VLABench [36], RoboTwin 2.0 [37]이 장기 상호작용, 다중모드 추론, 의미 일반화, 도메인 무작위화를 다룹니다. π0.7 [42]은 더 다양한 데이터와 풍부한 문맥으로 지시 따르기가 좋아진다고 보고하고, 논문은 훈련과 평가 설정이 달라 이 진단을 따로 둔다고 적습니다(§2.2). 불가능한 지시의 거부와 수정을 다루는 IVA [43]는 "가능한 과제 사이의 선택"을 묻는 RoboFollow와 보완 관계입니다.
5. **장면 엔트로피라는 아이디어.** 조건부 엔트로피 H(T|S)로 "시각만으로 과제가 정해지는 정도"를 재는 정의는 이 논문이 도입한 것으로 쓰여 있습니다(식 (1)). 비슷한 양을 이미 쓴 선행 문헌은 논문에 인용이 없고, 이 분석에서 찾지 않았습니다(미확인).

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| 시뮬레이터, 로봇(ALOHA AgileX 양팔), 카메라 3대, 기본 자산 | RoboTwin 2.0 [37] (MIT, 저장소 `LICENSE.RoboTwin`) | 저장소는 RoboTwin 체크아웃 안에 설치하는 구조이고 기준 커밋 `6dde571…`을 고정합니다(`04` D12). |
| 전문가 시연 생성 | RoboTwin의 계획기(cuRobo)와 자작 장면 코드(`scene_sources/`, `tasks/`) | 시연 수집에 cuRobo가 필요합니다. 장면마다 과제 정의를 새로 짰습니다. |
| 평가 프로토콜 L0–L3 | 이 논문 | LIBERO-PRO의 섭동 분류와 목적이 가깝지만 L1–L3의 정의가 다릅니다. |
| 단계별 부분 점수 | 이 논문 | 이전 벤치마크의 진행도 점수와 유사한 형태입니다. 같은 이름의 선행 정의는 확인하지 않았습니다(미확인). |
| π0.5 학습 | OpenPI(Apache-2.0)와 XPolicyLab 통합 | 저장소에 π0.5 어댑터와 LeRobot 변환기만 포함. 나머지 8개 모델은 없음. |
| 완화 기법 | LangForce [49], Stable language guidance [50], Qwen3-VL, GR00T 확산 머리 | 구현 코드와 설정은 저장소에 없음(`04` D13). |
| 학습 VQA 데이터 | ShareGPT4V-COCO(QA1), 자체 Scene 2 QA(QA2) | 자체 QA 데이터는 공개 목록에서 확인하지 못했습니다. |

## 4. GitHub와 자료 (확인 2026-11-26)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드 (RoboFollow)** | https://github.com/AutoLab-SAI-SJTU/RoboFollow | GitHub API MIT (파일은 `LICENSE.RoboTwin`뿐) | 별 8, 열린 이슈 2, 마지막 푸시 2026-10-17, 커밋 1개 `f642e3d19ddc0a3b0fb5795eef2cd17a12f286d8` | `repo/` 얕은 clone, 정적 읽기만. 약 32 MB(자산 `.glb` 포함) |
| 프로젝트 페이지 | https://mrc-crm.github.io/RoboFollow/ | - | - | README와 `docs/site-config.js`가 가리키는 주소. 2026-11-26 기준 확인하지 못했습니다. 페이지 원본은 저장소 `docs/`에 있습니다. |
| 데이터셋 | https://huggingface.co/datasets/AutoLab-SJTU/robofollow-data | 카드에 라이선스 표기 없음 | 다운로드 3,886, 수정 2026-09-23 | 파일 15,158개, 합계 약 40.4 GB(API 메타데이터). 받지 않음 |
| RoboTwin | https://github.com/RoboTwin-Platform/RoboTwin | MIT | 별 3,081 / 2026-09-24 | 시뮬레이터 본체. 기준 커밋 `6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755` |
| XPolicyLab | https://github.com/XPolicyLab/XPolicyLab | Apache-2.0 | 별 614 / 2026-11-26 | π0.5 학습 통합 |
| OpenPI | https://github.com/Physical-Intelligence/openpi | Apache-2.0 | 별 14,772 / 2026-08-24 | π0.5 학습기 |
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | MIT | 별 2,573 / 2025-03-15 | 비교 벤치마크 |
| LIBERO-PRO | https://github.com/Zxy-MLlab/LIBERO-PRO | MIT | 별 367 / 2026-07-17 | 논문 [8] |
| LIBERO-Plus | https://github.com/sylvestf/LIBERO-plus | 표기 없음 | 별 504 / 2026-01-21 | 논문 [9] |
| Motus | https://github.com/thu-ml/Motus | Apache-2.0 | 별 1,321 / 2026-10-09 | 평가 WAM |
| FastWAM | https://github.com/yuantianyuan01/FastWAM | GitHub API NOASSERTION | 별 1,676 / 2026-08-20 | 평가 WAM. 저장소 이슈에서 저자는 RoboTwin 체크포인트가 아니라 Wan2.2 + ActionDiT 초기화로 미세조정했다고 답했습니다. |

저장소의 이슈 두 개는 다음과 같습니다(2026-11-26 확인).
- #1(2026-10-06): FAST-WAM을 어떤 체크포인트에서 시작했는지 묻는 질문. 답변: Wan2.2 + ActionDiT 초기화.
- #2(2026-10-14): Scene 2 L3에서 "left bowl/right bowl" 참조와 채점 대상이 어긋난다는 보고. 답변(2026-10-17): 이전 공개본에 구버전의 Scene 2 평가 코드가 들어 있었고 수정했다. 이어서 논문의 평가 결과가 정확한지 묻는 질문에 "Yes". 자세한 대조는 `04` D8.

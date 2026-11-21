# 02. 선행연구와 관련 저장소

번호 [n]은 논문 참고문헌 번호입니다(`paper/paper.txt`의 References). 이 문서의 각 논문은 원문 인용만 옮겼고 별도로 열어 검증하지 않았습니다. 공식 코드(`repo/`)는 정적으로 읽었습니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| SCOUT의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 테스트 시점 적응을 그래디언트로 | TTT [4] (Sun 2020), TENT [5], EATA-류 [6], CTTA [7], MAE-TTT [8] | SCOUT는 "엔트로피·자기지도 목표" 대신 **행동 결과 예측 오차**로 적응합니다. 원문은 이 목표가 보상이나 자기지도 시각 목표보다 정보가 많다고 주장합니다(§3) |
| 메타학습 (내부 루프 미분) | MAML [9], FOMAML [10], CAVIA [16] | 외부 손실을 내부 갱신을 통과시켜 미분하는 구조는 MAML 계열입니다. CAVIA는 같은 "작은 문맥 벡터를 경사로 적응"하지만 과제 목표(보상·행동 라벨)를 씁니다 |
| 메타 강화학습과 맥락 추론 | RL² [11], Learning to RL [12], PEARL [13], VariBAD [14], Fakoor 2020 [15] | PEARL은 맥락 잠재의 사후를 샘플링합니다. 원문 B.2는 이를 지도 학습 GMM으로 바꾼 PEARL 계열 기준선을 만들었다고 밝힙니다 |
| 자기지도 적응 (배포 중) | PAD / Hansen 2020 [18] | 같은 자기지도 계열이지만 정책의 특징 추출기를 갱신합니다. 원문은 물리 변화가 "지각 변화"로 흡수된다고 비판합니다. SCOUT는 **명시적 신념 잠재**를 갱신합니다 |
| 동시 연구 | WorldAgen [19] (AAAI 2026) | 전진 예측 오차를 쓰지만 정탐색 뒤 LoRA를 미세조정합니다. 원문 §3에서 대비됩니다 |
| 맥락 기반 온라인 시스템 식별 | Yu 2017 [23], Peng 2018 [24], Andrychowicz 2020 [25], Lee 2020 [26], RMA [27], Dynamics as Prompts [28] | 이력에서 숨은 파라미터를 추정합니다. 원문은 "이 중 다수는 개선 신호를 명시하지 않는다"고 씀. RMA는 정책 안에서 인코딩합니다 |
| 상호작용 인지 (interactive perception) | Bohg 2017 [20], Katz 2014 [21], Weng 2024 [22] | 행동으로 지각을 얻는 고전 틀 |
| 이력 기반 행동 생성 | Zhao 2023 ACT [29], FlowBotHD [30], HAVE(Li 2025) [31] | HAVE는 **생성기 + 검증기**를 씁니다. SCOUT 표 2의 주요 비교 대상입니다. PNDiT [31]는 생성기 기준선입니다 |
| 세계 모델 (잠재 동역학) | Ha & Schmidhuber 2018 [32], PlaNet [33], Dreamer [34], Visual Foresight [35,36], DayDreamer [37], Simulation Distillation [38], TD-MPC [39] | SCOUT는 세계 모델을 **계획이 아니라 정책 신념 갱신**에 씁니다. 원문 §3은 "결합이 암묵적으로 남아 있었다"고 주장합니다 |
| 객체 상태 추정 (비교 벤치마크) | ISE [40] (Ko 2026), AVDC [44] (Ko 2024), BC, CQL [42] | ISE 과제 집합과 평가 설정을 그대로 씀. AVDC는 원문 [43]과 [44]가 같은 문헌입니다(참고문헌 중복) |
| 문 벤치마크 | SAPIEN [41], HAVE [31] | 모호한 문 (같은 기하, 다른 개방 방향)을 씀. PartNet-Mobility 기반 |
| 점 추적 (흐름 추정) | DELTA [45] | 실로봇 흐름 추정 (표 6)에 씀 |

**차별점 요약:** 원문은 "정책이 들고 있는 신념 잠재를 세계 모델의 예측 오차로 갱신하고, 그 갱신이 행동을 바로 바꾸도록 메타학습한다"는 점을 내세웁니다. 가장 가까운 대안은 이력 조건 GMM(같은 헤드, 기울기 없음)과 PEARL 계열(확률 맥락, 보상 없는 지도 학습)입니다.

## 2. 코드와 저장소 (GitHub 표)

| 저장소 | 관계 | 라이선스 | 대조 기준 | 상태 |
|---|---|---|---|---|
| [liy1shu/SCOUT](https://github.com/liy1shu/SCOUT) | 공식 구현 (CoRL 2026 저장소, 설명문 "Official repo") | 저장소 LICENSE 파일 없음 (API `license: null`). 하위 `third_party/metaworld`는 MIT | HEAD `cdb079c` (2026-10-18). API `pushed_at` 2026-11-19, 생성 2026-09-21, 별 7개 (2026-11-21 조회) | `repo/`에 얕은 복제. 정적 읽기만. 실행과 설치 안 함 (`04`) |
| [liy1shu.github.io/SCOUT](https://liy1shu.github.io/SCOUT/) | 프로젝트 페이지 | 확인 안 함 | 초록에서 링크 | 링크만 확인 |
| 체크포인트 (Google Drive, README 링크) | 여섯 과제 가중치 (`pickbar.pt` 등) | 확인 안 함 | 저장소 `ckpts/`에는 `.gitkeep`만 있음 | 받지 않음 (용량과 승인 필요) |
| 문 자산 (Google Drive, README 링크) | 생성된 모호한 문 자산 | 확인 안 함 | README에 PartNet-Mobility 재생성 명령 있음 | 받지 않음 |
| PartNet-Mobility | 문 원본 자산 | 등록 필요 (원자료 라이선스, 확인 안 함) | `generate_door_assets.py` | 받지 않음 |
| Meta-World v3.0.0 포크 (저장소에 포함) | ISE 과제 시뮬 환경 | MIT (`third_party/metaworld/LICENSE`, 2019 Meta-World Team) | 저장소 내부 | 정적 읽기만 |
| HAVE 코드 (Li 2025) | 문 시뮬을 "trimmed port" (README) | 확인 안 함 | 저장소 README 문장 | 확인 안 함 (범위 밖) |
| Ko 2026 ISE 공식 코드 | ISE 기준선 구현 출처 | 확인 안 함 | 논문 인용 | 확인 안 함 (범위 밖) |

- 저장소에는 실로봇 코드, 기준선 코드 (BC, CQL, AVDC, ISE, PEARL 계열, 이력 조건 GMM, PNDiT), 표 6·7과 D.2의 분석 스크립트가 없습니다. `grep -rniE 'pearl|cql|avdc|pndit' scout scripts tests`는 0건입니다.
- 저장소의 세부 대조는 `04_code_verification.md`에 있습니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과와 참고문헌 점검

- 내부 루프의 안정성 논증은 자기 인코더 E의 LayerNorm 설계와 흐름 재구성 손실에 기댑니다(A.3, A.4). 선행 붕괴 문제는 원문이 자체 실험으로 보여 줍니다(그림 6).
- 참고문헌 점검:
  - [43]과 [44]는 같은 문헌(Ko et al., ICLR 2024)이 두 번 실렸습니다.
  - [31]은 학회와 연도가 없습니다("In 9th Annual Conference on Robot Learning").
  - [19]는 본문에서 "WorldAgen", 참고문헌에서 "Worldagen"입니다.
  - [38]의 Simulation Distillation은 arXiv:2603.15759(2026)로 적혀 있습니다. 확인하지 않았습니다.
- 부록 B.5는 손실 식을 "Lrecon (Eq. 1)"으로 가리키지만, 식 1은 B.2의 PoG 사후 식입니다. 재구성 손실에는 식 번호가 없습니다.

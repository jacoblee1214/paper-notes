# PHUMA 선행연구와 관련 분석 (02)

## 1. 계보: 무엇을 이어받고 무엇을 바꿨나

**데이터 출처 계보**

- 모캡: AMASS(13), LaFAN1(12), LocoMuJoCo(21), GRAB, EgoBody 등. 정확하지만 작고 걷기 중심입니다(§2.1).
- 영상 복원: Humanoid-X(1)는 영상에서 SMPL을 복원(VIBE 계열 [16])하고 리타게팅하는 방식입니다. PHUMA는 이 흐름을 이어 받되, 떠 있음·관통·관절 위반을 걸러 냅니다.
- 주요 영상 데이터: Motion-X Video(33.0K 클립), IDEA400, HAA500, AIST, HuMMan(표 8).

**리타게팅 계보**

- IK(PINK [67], 스케일링 IK), GMR [69]: 사람 관절 위치를 로봇 뼈 길이에 맞춰 선형 스케일링.
- SINK [9, 17, 70–72]: 사람 SMPL 형상을 로봇 T자세에 맞춘 뒤 자세를 맞춤. PhySINK의 출발점.
- 물리 항 추가 연구 [73–78]: 접촉 제약을 넣지만 "바닥이 가장 낮은 발"이라는 가정을 씀. 영상 노이즈에서 깨진다고 본문이 지적(§2.2).
- 샘플링 기반 동역학 리타게팅 SPIDER [75]: PhySINK를 초기값으로 쓰는 실험(부록 B.3, 표 11)에서 비교.
- 신경 리타게팅 [79, 80]: 학습 쌍을 위해 기존 리타게팅이 필요하다고 지적.

**추적 정책 계보**

- MaskedMimic [8](시뮬 평가 프레임워크, 전신 성공률), BeyondMimic [18](실물 G1 배치), KungfuBot [86](교차 검증용), ProtoMotions(저장소 README의 학습 파이프라인 안내).

**PHUMA의 차별점(논문 자체 주장)**

1. 물리 지표를 사람 모션에 먼저 적용해 클립을 거름(모델 기반 추적 정책으로 거르지 않음, §2.1).
2. PhySINK로 관절 한계, 접지, 미끄럼을 손실에 넣음(§3.2).

## 2. 직접 대조하는 것과 빼 놓은 것

| 대조 대상 | 논문에서 어떻게 쓰는가 | 비고 |
|---|---|---|
| LaFAN1 (2.4 h) | 표 4 데이터셋 비교, 리타게팅 완료본 그대로 사용 | 리타게팅 방법이 PHUMA와 다름 |
| AMASS (20.9 h) | SINK로 리타게팅한 뒤 비교 (표 3, 4), PhySINK 적용본도 비교(D.4) | 같은 리타게팅 방법(SINK)에서 비교 |
| Humanoid-X (231.4 h) | 완료본을 그대로 씀 (§4.4) | 리타게팅과 큐레이션이 Humanoid-X 쪽 방법이라 PHUMA와 혼합됨 |
| IK, GMR, SINK | 리타게팅 비교 (표 1, 3, 9, 25) | |
| SPIDER | 초기값 실험 (부록 B.3) | 샘플링 최적화는 재현하지 않음 |
| KungfuBot | 다른 알고리즘에서 데이터 효과 확인 (부록 D.3) | Isaac Gym |
| MuJoCo | 교차 시뮬 평가 (표 24) | |

**빼 놓은 것**

- 물리 항을 넣은 다른 리타게팅 연구 [73–78](Kinodynamic [76], SPARK [77], ReActor [78] 등)과의 정량 비교는 없습니다.
- OmniRetarget [74] 같은 상호작용 보존 리타게팅과의 비교도 없습니다(PHUMA는 평지 보행 범위).
- 데이터 큐레이션을 다른 방식(추적 정책으로 거르기, [11])과 직접 비교하지 않습니다. 관련 연구 절에서 "정책으로 거르면 OOD 동작을 버린다"고만 주장합니다(§2.1).

## 3. 이 저장소의 관련 분석과의 관계

| 폴더 | 대상 (arXiv) | PHUMA와 맞닿는 지점 | 차이 |
|---|---|---|---|
| `warp_retargeting/` | WARP, 전신 리타게팅 (2606.29940, CoRL 2026 표기) | 리타게팅 품질 지표가 정책 성능의 근거로 쓰이는 구조. 두 논문 모두 "리타게팅 품질 → 정책"을 주장 | WARP는 사람 상체 시연과 로봇 시연(DexMimicGen, GR1)을 씀. PHUMA는 전신 보행과 사람 모캡·영상 |
| `ultra_humanoid/` | ULTRA, 휴머노이드 loco-manipulation (2603.03279) | 관통·접촉 지표와 증강 데이터 효과를 같은 표에서 비교하는 문제. ULTRA도 OmniRetarget 대비 데이터 차이를 섞음 | 상호작용(물체) 중심. PHUMA는 평지 보행 |
| `reasoning_retargeting/` | Retargeting Through Reasoning (블로그, 2026-11-06, 논문 아님) | 리타게팅 품질을 MuJoCo 피드백으로 재는 점 | 블로그, 물체 상호작용, LLM 에이전트 루프. 공개 코드는 확인하지 않음 |
| `gnr/` | GNR, 생성형 신경 리타게팅 (2610.12440) | 리타게팅 해의 다중성과 정책 학습. PHUMA는 해를 하나로 두고(SINK 계열), GNR은 생성 모델로 받음 | 손 리타게팅(DG-5F), 샘플링 MPC + flow matching |
| `cross_embodiment_wm/` | Scaling Cross-Embodiment WM (2511.01177, IROS 2026) | 서로 다른 몸체 사이 간극 흡수 | 손 전용, 파티클 세계모델 |
| `motion_latent_action/` | Motion-Focused Latent Action (2606.18955, IROS 2026) | 사람 영상을 로봇 학습에 쓰는 방법. PHUMA는 영상을 기하로 옮기고, 이 연구는 잠재 행동으로 옮김 | 사람 egovideo, 마스크 파이프라인 |

**정리.** PHUMA를 이 저장소의 다른 리타게팅 분석과 비교하면, 공통된 질문은 "리타게팅 품질 지표를 정책 성능과 어떻게 연결할 것인가"입니다. WARP, ULTRA, GNR은 같은 질문에 다른 답을 냈고, 각 논문의 평가 설계 차이가 결과 해석을 좌우합니다. PHUMA만의 특징은 보행 범위와 데이터 양 대 품질 실험입니다.

## 4. GitHub, Hugging Face 확인 (2026-11-12)

| 항목 | 확인 결과 | 확인 방법 |
|---|---|---|
| arXiv 논문 | 2510.26236, v3 2026-10-12 제출, v1 2025-10-30 제출, 라이선스 CC BY 4.0 | arXiv abs 페이지 |
| 학회 | CoRL 2026 Spotlight (프로젝트 페이지, GitHub README, 원문 머리글 일치) | 프로젝트 페이지 HTML, PDF 1쪽 |
| 프로젝트 페이지 | https://davian-robotics.github.io/PHUMA/ (`paper/project_page.html` 저장) | curl |
| 코드 저장소 | github.com/davian-robotics/PHUMA, 라이선스 Apache-2.0, 스타 253, 마지막 push 2026-09-30, 기본 브랜치 main | GitHub API (`/repos/...`) |
| 분석에 쓴 코드 | 얕은 복제 HEAD `2faf15b` (2026-09-30), `phuma/repo/` (gitignore) | `git clone --depth 1` |
| 저장소 파일 | `src/curation` 5개 + 공통 유틸, `src/retarget` 3개(`motion_adaptation.py`, `_multiprocess.py`, `shape_adaptation.py`), `asset/` 로봇 모델, `data/split/*.txt`(분할 목록), 정책 학습 코드 없음 | 파일 목록 |
| HF 데이터셋 | DAVIAN-Robotics/PHUMA, `gated: false`, `private: false`, 카드 `license: apache-2.0` | HF API (`/api/datasets/...`) |
| HF 데이터 파일 | `data.zip` 3,416,370,206 바이트 (**3.42 GB**), README, .gitattributes, .gitignore | HF API 목록, `resolve/main/data.zip` HEAD 요청 (본문 다운로드 안 함) |
| 분할 개수 | train 67,953, test 7,557, unseen_video 504 (합계 76,014) | `data/split/*.txt` 줄 수 |
| 공개 범위 | HF에는 G1, H1-2 리타게팅 결과. LAFAN1·LocoMuJoCo 제외(라이선스). 학습·테스트 원본 사람 포즈 비공개(라이선스). Unseen Video의 SMPL 파일은 "곧 공개" | HF README, GitHub FAQ |
| 외부 파이프라인 | ProtoMotions(NVlabs)에 PHUMA 데이터 지원 추가(2025.12 소식). 이 분석에서는 확인하지 않음 | GitHub README 소식란 |

**판단.** 코드는 Apache-2.0이라 분석과 재사용에 문제가 없습니다. 데이터는 Apache-2.0을 표시하지만, 원천(AMASS, Motion-X, IDEA400 등)과 제외 사유를 보면 원천 약관이 함께 적용될 수 있습니다. 이 분석에서는 원천 약관을 확인하지 않았으므로 **데이터 사용 전에 원천별 약관을 따로 확인해야 합니다.** 게시본에는 데이터를 넣지 않았습니다.

## 5. 기준선 해석 메모

- **데이터셋 비교는 세 가지가 섞입니다**: 데이터 양(2.4 h, 20.9 h, 231.4 h, 73.0 h), 리타게팅 방법(LaFAN1과 Humanoid-X는 원 제공자 방식, AMASS는 SINK, PHUMA는 PhySINK), 큐레이션 여부(PHUMA만 있음). 표 4만으로 "PHUMA 품질이 높다"는 결론을 분리할 수 없습니다. 같은 리타게팅(SINK)으로 큐레이션만 바꾼 표 2가 이 문제를 일부 풉니다.
- **Humanoid-X 기준선이 가장 약합니다.** 저자 본인이 "품질 문제로 낮다"고 씁니다. 그러나 Humanoid-X 쪽 리타게팅 방식과 학습 예산(반복 수)은 논문에 명시되지 않았습니다.
- **같은 학습 예산인지 확인되지 않습니다.** 데이터가 3배 큰 Humanoid-X가 더 많은 학습 샘플을 받았는지 여부가 보고되지 않습니다.
- **평가 데이터의 리타게팅도 PhySINK입니다.** Unseen Video 입력도 PhySINK로 처리했습니다(부록 D.1). 부록 D.4가 GMR로 바꾼 평가 입력으로 재확인합니다.

# 02. 선행연구와 코드 저장소

## 1. 계보와 가까운 대안

MESA가 서 있는 자리는 "LIBERO류 언어 조건 저수준 제어 벤치마크의 포화와 일반화 분리 부족"입니다(§1, §2). 논문이 직접 비교하는 계열은 다음과 같습니다.

| 계열 | 대표 | MESA가 보는 한계 (논문 §2, 표 1) | 이 분석의 보충 |
|---|---|---|---|
| 평생·다과제 언어 조건 | LIBERO(130과제), CALVIN, MetaWorld | 포화(SOTA 98% 이상, 논문 [18] 인용), OOD 후속(LIBERO 변형 [39, 40])은 학습 데이터를 물려받아 모든 RFM이 거의 0 | 이 포화 서술은 논문이 인용한 [18]의 수치에 기댑니다 |
| 광범위 가정 환경 | RoboCasa, RoboCasa365, BEHAVIOR-1K | 이동 조작·장기 과제라 RFM 사전학습 분포 밖이고 통제된 OOD 분할 없음 | RoboCasa는 MESA의 객체 자산과 MimicGen 기반 확장을 공유합니다 |
| 실-시뮬 쌍둥이 | SIMPLER, MolmoSpaces, REALM, RoboLab, DROID 기반 | DROID(크고 지저분함) 학습이 필요, 분포 밖 설정을 학습에서 제외했음을 보장하기 어려움 | MESA는 "DROID-like Franka" 환경을 따로 만들어 이 문제를 피합니다 |
| 이중 팔 | PerAct2, Robomimic, RoboTwin 2.0 | 다과제 언어 따르기와 일반화를 재기엔 다양성 부족 | BiMESA는 팔 배정, 핸드오프 방향, 고정물 위치를 OOD 축으로 둡니다 |
| 키프레임 기반 | RLBench, COLOSSEUM, VLABench | 저수준 제어 정책(RFM)과 입출력이 다름 | 해당 없음 |

표 1(논문)의 "Num. Tasks 1021"은 단일 팔 학습 806 + 단일 팔 OOD 평가 70 + 양팔 145의 합입니다(`03` V9에서 계산 확인). 표 1의 MESA 행 "Obj./Insts 153 / 2509"는 RoboCasa 행과 같은 값이라 MESA가 RoboCasa 자산 전체를 쓰는 것으로 읽히지만, 실제 과제에서 쓰이는 범주 수는 논문에 없습니다(미확인).

### 구성 요소별 출처

| 구성 요소 | 출처 | MESA에서 바뀐 점 |
|---|---|---|
| BDDL 과제 정의 | BEHAVIOR 계열 BDDL, LIBERO가 채택 | 과제마다 train/eval 변형 JSON(과제당 평가 50개), `distractor_goal_specs` 추가(`04` D3) |
| MimicLabs 인프라 | Saxena 외 [55] (robosuite, MimicGen, LIBERO, RoboCasa 통합) | 하위 과제 색인, 이어붙이기 생성기, 궤적 시작 최적화 |
| MimicGen | Mandlekar 외 [27] | 이어붙이기(D.3), roll/pitch/yaw 제거와 시작점 최적화(D.4) |
| 시뮬레이터 | MuJoCo, robosuite | Franka Panda + Robotiq 2F-85(Menagerie), YAM 양팔 |
| 데이터 수집 | Meta Quest, oculus_reader | 원천 시연 1,125개(단일 팔), 양팔 2,152개 |
| 기준선 학습 | openpi(JAX)의 PG-Bin, PG-FM, π0, π0-FAST, π0.5, Isaac-GR00T | 논문은 기본 설정 사용 |
| 실세계 비교 | RoboArena(Atreya 외 [51]), Jain 외 [22]를 따름 | Elo 대 성공률 순위만 비교(표 5) |
| LLM 사용 | 물체-용기 호환성 라벨(GPT-5.1), 요 대칭 라벨(LLM, D.4) | 저장소에는 정적 결과(카테고리 목록)만 있고 LLM 호출 코드 없음(`04` D12) |

## 2. GitHub와 공개 자료 표 (2026-11-23 `gh api` 확인)

| 항목 | 관계 | 라이선스 | 별 | 마지막 push | 커밋 | 상태 |
|---|---|---|---|---|---|---|
| [pairlab/mesa-env](https://github.com/pairlab/mesa-env) | **공식 코드**(MESA-Gen, MESA-Bench, 평가 서버) | MIT | 1 | 2026-11-12 | `e4a82e2`(2026-11-05) 얕은 복제 | `repo/`에 정적 읽기만 |
| [pairlab/openpi-mesa](https://github.com/pairlab/openpi-mesa) | openpi 포크(정책 서버용으로 추정) | Apache-2.0 | 0 | 2026-10-15 | 받지 않음 | 확인 안 함 |
| [pairlab/mesa-GR00T](https://github.com/pairlab/mesa-GR00T) | GR00T 추론 전용 포크 | GitHub 표기 없음(NOASSERTION), 가중치는 NVIDIA 비상업 라이선스(문서) | 0 | 2026-10-31 | 받지 않음 | 확인 안 함 |
| [프로젝트 페이지](https://pairlab.github.io/MESA) | 프로젝트 사이트, 문서 링크 포함 | 사이트 하단 템플릿 표기 CC BY-SA 4.0(논문과 무관) | - | - | - | HTML 저장(`paper/project.html`) |
| [OpenReview](https://openreview.net/forum?id=Br2rXixvyN) | CoRL 2026 채택 포럼 | - | - | - | - | 링크만 |
| Hugging Face `albertwilcox/mesa` 컬렉션 | 데이터와 체크포인트 15개 | 데이터 CC BY 4.0, 체크포인트 π는 Apache-2.0, GR00T는 비상업(other) | - | 2026-10-03 갱신 | - | **받지 않음**, 메타데이터 크기만 확인 |

Hugging Face 항목 크기(메타데이터 조회, 받지 않음): mesa-70-lerobot 11.7 GB, mesa-70-hdf5 473 GB, mesa-all-train-lerobot 131.6 GB, bimesa-57-lerobot 10.3 GB, mesa-assets 13.6 GB, mesa-init-states 0.17 GB, 체크포인트 mesa-pi05 12.4 GB, mesa-gr00t-n1.6 6.6 GB.

관련 공개 저장소(이 분석이 대조에 쓰지는 않음): Physical-Intelligence/openpi(Apache-2.0, 별 14.7k), NVIDIA/Isaac-GR00T(Apache-2.0, 8.4k), NVlabs/mimicgen(라이선스 NOASSERTION, 674), robocasa/robocasa(NOASSERTION, 1.9k), Lifelong-Robot-Learning/LIBERO(MIT, 2.6k, 마지막 push 2025-03), robo-arena/roboarena(117).

## 4. MESA가 열어 두는 질문 (문헌 위치)

- **언어 따르기 지표의 타당성**: L = S/(S+D)의 D가 "과제 난이도 비대칭"에 취약한지(`03` §3 C6).
- **합성 데이터로 만든 벤치마크의 순위 신뢰성**: MimicGen 데이터로 학습한 정책의 순위가 사람 시연 학습 정책과 같은가. 논문도 검증하지 않았다고 인정합니다(§6).
- **보조 과제 다양성 공동학습의 무이득**: 고정 스텝 예산에서 ID 과제가 덜 학습되는 혼란이 있습니다(`03` C1).

# 02. 선행연구와 구성 요소의 출처

번호 [n]은 AnyDexRT v2의 참고문헌 번호입니다.

## 1. 계보: 사람 손 동작을 로봇 손으로 옮기는 운동학 리타게팅

| 계열 | 대표 | 방식 | AnyDexRT가 지적하는 한계 (1절, 2.1절) |
|---|---|---|---|
| 최적화 (과제 벡터) | DexPilot [18], DexMV [20], AnyTeleop [10], Open-TeleVision [21], ACE [22], Bunny-VisionPro [28], Wuji 리타게팅 [19] | 사람과 로봇의 지문·관절 벡터 쌍 {(v^H_i, v^R_i)}을 수작업으로 정하고 Σ α_i‖v^H_i − v^R_i‖² 최소화로 로봇 관절을 구함. 관절 한계·평활·시간 정규화로 안정화 | 좌표계, 벡터 원점, 스케일, 목적 가중치를 보정해야 함. 손마다 목적·제약을 따로 설계 |
| 최적화 모방 신경망 | Robotic Telekinesis [11], HumDex [23], Li 외 [24] | 위 최적화를 신경망이 모방해 속도를 높임 | 학습 목표가 보정된 최적화 결과라 같은 가정과 조정을 물려받음 |
| 대응 학습 | 조화 사상 [34], Xin 외 [17](핵심 목적 분석), **GeoRT** [12] | 사람과 로봇의 도달 가능 지문 공간을 맞춰 신경망 정책으로 상각 | GeoRT는 고정 신경 FK 모델, 손목 좌표계의 운동 보존, 양방향 Chamfer를 쓰는데 이들이 훈련 샘플·보정 오차·로봇 여분 영역에 민감해질 수 있음 |
| 전용 하드웨어 | 글러브·외골격 [25, 26], DexOP [2], DexUMI [27] | 손 형태를 맞춘 장치로 직접 대응 | 형태에 묶여 다른 크기·비율의 손으로 일반화가 어려움 |
| 운동감각 교시 | KineDex [40] | 로봇 손을 사람이 직접 움직여 수집 | 가림이 생기고 사람 손과 크기가 많이 다르면 어려움 |

AnyDexRT는 "대응 학습" 계열에 속하고 GeoRT를 직접 겨냥한 수정판입니다. 기준선은 최적화 [10]과 GeoRT [12] 둘이며, 다른 신경망 방식(Robotic Telekinesis, HumDex)과는 비교하지 않았습니다.

## 2. 구성 요소별 출처

| 구성 요소 | 논문 위치 | 출처·선행 | 새로운 점 |
|---|---|---|---|
| Chamfer 거리 | 3.2절 | Barrow 외 1977 [46], Fan 외 2017 [47], GeoRT [12]가 리타게팅에 사용 | 리타게팅에서는 GeoRT가 먼저 씀 |
| 부분(한 방향) Chamfer | 식 1 | Cycle4Completion [48] (점군 보완) | 리타게팅의 "여분 영역" 문제에 적용 |
| 쌍별 거리 보존 | 식 2 | 일반적인 등거리 정규화 | GeoRT에도 비슷한 기하 보존 항이 있는지는 이 문서에서 확인하지 않음 |
| 지역 좌표계 운동 보존 | 식 3 | GeoRT의 전역(손목) 좌표 운동 보존 [12]의 지역 좌표 변형 | 새 변형. 동기는 글러브 보정 오차 |
| 소수 샷 앵커 정렬 | 식 4, 부록 D | 쌍을 이룬 지도 신호 몇 개로 비지도 대응을 고정하는 일반 기법 | 굽힘·옆 회전 앵커를 손가락 시너지 관계(λ, [43] Rijpkema–Girard 1991)로 설계 |
| 손가락 결합·시너지 가정 | 3.1절 | Santello 외 1998 [44], Duits 외 2015 [45] | 이 가정으로 f_s를 일대일로 둠 |
| 접촉 분류기 | 식 5, 3.4절 | 핀치 정밀도 문제는 [17]에서 논의 | 지문 두 점을 입력으로 하는 이진 분류기와 템플릿 보정 |
| NNS 역변환 | 부록 B | 흔한 방식 | 사전 생성한 지문-관절 쌍 탐색 |

## 4. 공개 저장소와 자료

2026-11-25에 GitHub API와 프로젝트 페이지로 확인했습니다. 별과 포크는 그날의 값입니다.

| 저장소·자료 | 역할 | 라이선스 | 별 / 포크 | 마지막 push | 커밋 | 이 분석에서 |
|---|---|---|---|---|---|---|
| [chenxi-wang/anydexrt](https://github.com/chenxi-wang/anydexrt) | **공식 구현** (데이터 수집, 학습, 라이브 시각화) | **없음** (API license: null, 루트에 LICENSE 파일 없음). `hands/` 아래 손별 URDF·메시에는 각자 LICENSE가 있음 | 10 / 1 | 2026-11-14 | `627003b` (커밋 3개: 11-06 first commit, 11-14 앵커 수집 예시 추가, 11-14 update) | 얕은 clone 후 정적 읽기만 (`repo/`, gitignore) |
| [프로젝트 페이지](https://chenxi-wang.github.io/projects/anydexrt) | 논문·코드 링크, 방법 요약, 시뮬·실로봇 영상 | 하단에 CC BY-SA 4.0 링크가 있으나 템플릿(Nerfies) 표기로 보이며 논문 그림 재사용 허가로 읽지 않음 | — | — | — | HTML 읽기 |
| [chenxi-wang/manus_core_python_sdk](https://github.com/chenxi-wang/manus_core_python_sdk) | 글러브 연동에 README가 요구하는 MANUS Core 파이썬 인터페이스 | 없음 | 0 / 0 | 2026-07-13 | — | 메타데이터만 |
| [qqsq12321/anydexRT](https://github.com/qqsq12321/anydexRT) | 제3자 재구현. README에 "공식 코드 미공개(Code: Coming Soon)"라고 적힌 시기(2026-11-06)에 논문 본문만으로 만든 것(v1 제목 "Calibration-Free" 사용) | 없음 | 0 / 0 | 2026-11-06 | — | 메타데이터와 README 앞부분만. 공식 코드와 무관 |
| [facebookresearch/GeoRT](https://github.com/facebookresearch/GeoRT) | 기준선 GeoRT [12]의 공식 구현 | CC BY-NC (비상업) | 241 / 40 | 2025-09-02 | — | 메타데이터만 |
| [dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting) | 기준선 [10]의 최적화 리타게팅 구현(AnyTeleop에서 출발) | MIT | 1,322 / 155 | 2025-08-21 | — | 메타데이터만 (논문이 이 구현을 썼는지는 본문에 명시되지 않음) |
| [wuji-technology/wuji-retargeting](https://github.com/wuji-technology/wuji-retargeting) | Wuji 손 전용 리타게팅 [19] | MIT | 80 / 27 | 2026-08-17 (보관됨) | — | 메타데이터만 |

공식 저장소의 구성: `anydexrt/`(수집·데이터셋·학습·추론 모듈), `configs/`(손 9종: Ability, Allegro, Encos, Inspire, LEAP, Shadow, Sharpa, Wuji, XHand), `hands/`(URDF와 메시), `scripts/`, `docs/DATA_COLLECTION.md`. 논문에 나오는 7개 손에 Encos와 Sharpa가 더 있습니다. 학습된 체크포인트, 사람 글러브 데이터, 평가 코드(GMC·LMC, 보정 민감도, 시드 안정성), 실로봇 제어 코드는 저장소에 없습니다(`04` §1).

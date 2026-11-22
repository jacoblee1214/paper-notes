# 02. 선행연구와 관련 자료: ADEPT (arXiv:2608.19182)

참고문헌 번호는 논문 v2의 번호입니다. 논문이 직접 서술한 계보와, 표에서 확인되는 구성 요소 출처를 구분해서 적었습니다.

## 1. 계보

### 1.1 sim-to-real 숙련 조작 RL

- **대규모 PPO + 도메인 무작위화:** OpenAI의 Rubik's Cube[1], 손 안 조작[15], DeXtreme[2](Isaac Gym, ADR)이 출발점입니다. ADEPT의 ADR 곡선과 비대칭 액터-크리틱, 상태 기반 교사 → 시각 학생 구조는 이 계열을 따릅니다.
- **손 안 재방향:** Visual Dexterity[4], Chen 외[16], Qi 외[17, 18]. 표 1의 VisDex 152개 물체가 이 계열의 평가 물체입니다.
- **초기 상태·커리큘럼:** OmniReset[20](다양한 리셋 상태), ManipGen[19](지역 정책 + 모션 플래너), DemoStart[22](시연으로 시드).
- **물체 중심 영샷:** SimToolReal[21]은 물체 중심 정책으로 영샷 도구 사용을 하지만 배포 시 추정된 물체·목표 자세에 조건화합니다(논문 §2). ADEPT는 학생이 포즈 추정기 없이 RGB와 촉각을 바로 씁니다.
- **팔-손 시각 정책의 직접 선행:** DextrAH-G[3], DextrAH-RGB[14]. 둘 다 기하 패브릭을 저수준 제어기로 쓰고(손은 5차원 PCA 파지 부분공간 + 6차원 손바닥 자세), 교사-학생 증류를 쓰며, 보조 위치 손실(ADEPT의 8 모서리 키포인트 손실의 원형)을 씁니다.

### 1.2 RL 사전학습과 전이

- **행동 사전분포·망각 억제:** URLB[6], Cal-QL[7], 온라인 파인튜닝이 오프라인 데이터를 유지할 필요 없음[23], 적응적 BC 정규화[24], EWC[8], RL's Razor[10](온라인 RL이 덜 잊는다는 주장). ADEPT의 "로컬 갱신으로 사전 능력을 보존한다"는 서사는 [10]에 기댑니다.
- **숙련 사전학습:** DexGen(DexterityGen)[25]은 RL이 만든 손 안 전이로 확산 제어기를 학습해 텔레옵 명령을 다듬습니다. 동시기 Play2Perfect[26]는 놀이 사전학습으로 정밀 조립을 합니다. 물체 자세는 외부 추정, 목표 자세는 외부 공급을 가정합니다(논문 §2).
- **ADEPT의 위치:** 정책 수준 일반성(reposing 사전학습)을 RL로 만들고, 사후학습에서 관측·보상 변경을 처리하는 "구조화된 적응"(액터 증류, 비평가 워밍업, 보수적 PPO)을 제안합니다.

### 1.3 기하 패브릭

- 리만 운동 정책(RMP)[27], RMPflow[28], 최적화 패브릭[29], 기하 패브릭[13](ICRA 2024). 패브릭은 안정성이 보장된 2차 행동 사전분포와 충돌·관절 한계 회피를 줍니다.
- DextrAH 계열은 팔 전체를 6차원 손바닥 목표 + 5차원 PCA 손으로 몰았고, ADEPT는 **관절 공간 전체**(23/29차원)를 구동합니다(논문 §2, §3.5). 이 선택으로 손가락 협응(finger gaiting)을 열지만 탐색 문제가 어려워집니다(논문 자체 인정).

### 1.4 촉각 sim-to-real

- HydroShear[30](수탄성 전단 시뮬레이션, 평행 턱 영샷), TacMap[31](기하 일관 관통 깊이 지도), SaTA[32](손끝 위치로 손가락별 특징을 FiLM 조건화). ADEPT는 TacMap 깊이 + 이진 접촉 지도와 SaTA식 앵커링을 결합합니다(§3.4, 부록 H).

### 1.5 집단 기반 학습

- PBT[33], DexPBT[9](손-팔 시스템에 PBT). ADEPT는 사전학습에 분산 PBT(16 정책)를 씁니다(부록 C).

## 2. 구성 요소별 출처

| 구성 요소 | 논문 위치 | 출처 | 새로움 (논문 주장) |
|---|---|---|---|
| 비대칭 액터-크리틱 PPO, ADR | §3.1, 부록 A | DeXtreme[2], PPO[11] | 새롭지 않음 |
| 일반 reposing 과제, 16 기본 형상, 키포인트 보상 | §3.2, 부록 A.1 | DexPBT[9]의 reposing 과제 | 사전학습 대상으로 재사용 |
| PBT 탐색 | 부록 C | PBT[33], DexPBT[9] | 새롭지 않음 |
| BC 액터 증류 | §3.3 (1) | 정책 증류, DAgger 변형 | 관측 공간 변경에 사용 |
| 비평가 워밍업(액터 동결) | §3.3 (2) | 가치 사전 보정 (Cal-QL[7]류의 가치 정렬) | 하류 보상에 맞춘 새 비평가 학습 |
| 보수적 PPO(LR 1e-5, 클립 0.05) | §3.3 (3) | RL's Razor[10], EWC[8] | 학습률이 핵심이라는 절제 |
| 교사-학생 DAgger | §3.4 | DAgger[34], DextrAH-RGB[14] | 2단계 커리큘럼(대리 과제 → 삽입) |
| 8 모서리 키포인트 보조 손실 | §3.4, 식 32 | DextrAH-RGB[14]의 물체 위치 손실 | 방향 지도로 확장 |
| 관절 공간 기하 패브릭 | §3.5, 부록 B | 기하 패브릭[13], DextrAH[3, 14] | PCA 부분공간 대신 전체 관절 공간 |
| 촉각 표현 | §3.4, 부록 H | TacMap[31], SaTA[32] | 두 방법의 결합 |
| FMB 과제와 비교 기준 | §4 | FMB[12] | 평행 턱 → 다지 손 |

## 4. GitHub와 코드 자료 (확인 2026-11-22)

| 자료 | 링크 | 라이선스 | 별 / 마지막 push | 비고 |
|---|---|---|---|---|
| **공식 코드** | 없음 | - | - | 프로젝트 페이지 `Code` 버튼은 "Coming soon"(비활성). 저자 계정 `JayjunLee`와 조직 `adept-dexterity`에 코드 저장소 없음 |
| 프로젝트 페이지 | https://adept-dexterity.github.io (저장소 `adept-dexterity/adept-dexterity.github.io`) | 표기 없음 | 별 0 / 2026-10-05 (`d9fa561`) | HTML, 영상, 의사코드, BibTeX. 메타데이터와 HTML만 열람 |
| DextrAH (G/RGB 코드, ADEPT의 직접 기반) | https://github.com/NVlabs/DEXTRAH | NVIDIA License(비상업·NVIDIA 프로세서 한정) | 183 / 2025-09-09 (`ebc08ed`) | Isaac Lab v2.2.1, 기하 패브릭 별도 설치. 메타데이터와 README만 열람 |
| FABRICS (기하 패브릭 구현) | https://github.com/NVlabs/FABRICS | NVIDIA License(비상업·NVIDIA 프로세서 한정) | 97 / 2025-09-09 (`a7e55a6`) | ADEPT의 Cspace 패브릭 확장은 공개분에 없을 가능성이 큼(미확인) |
| Isaac Lab | https://github.com/isaac-sim/IsaacLab | NOASSERTION(API) | 8,700 / 2026-11-22 (최신 릴리스 v3.0.0) | 시뮬레이션 기반 |
| SimToolReal | https://github.com/tylerlum/simtoolreal | MIT | 437 / 2026-09-21 (`313d5ae`) | 논문 [21] |
| Play2Perfect | https://github.com/kushal2000/play2perfect | MIT | 166 / 2026-11-10 (`f67ef45`) | 논문 [26] |
| FMB (벤치마크) | https://github.com/rail-berkeley/fmb | 표기 없음 | 92 / 2024-05-28 | 논문 [12], 평가 기준 |
| rl_games (PPO 구현, DextrAH 계열이 사용) | https://github.com/Denys88/rl_games | MIT | 1,409 / 2026-11-21 | ADEPT 사용 여부는 미확인(DextrAH가 사용) |
| HydroShear | https://github.com/MMintLab/hydroshear | 표기 없음 | 56 / 2026-07-06 | 논문 [30] |
| TacMap, SaTA, OmniReset | 공식 저장소를 못 찾음 | - | - | GitHub 검색에서 확인 안 됨 |

- 어떤 저장소도 clone하지 않았습니다(공식 코드가 없고, 나머지는 메타데이터와 README만 봤습니다). 가중치와 데이터는 받지 않았습니다.
- DextrAH와 FABRICS의 NVIDIA License는 비상업 연구·평가 목적, NVIDIA 프로세서(GPU 등) 위 사용을 조건으로 합니다(LICENSE 본문). 맥에서 실행하는 것은 허용 범위 밖일 수 있어 정적 열람만 합니다.

## 5. 가까운 대안과 동시기 연구

- **놀이 사전학습 + 정밀 조립 (Play2Perfect[26])**: 사전학습 철학이 같지만 외부 포즈 추정과 목표 자세를 쓰는 상태 기반 교사를 배포합니다. ADEPT는 포즈 추정 없는 RGB 학생이 목표입니다.
- **다양한 리셋 (OmniReset[20])**: 사전학습 없이 초기 상태 분포 확장으로 숙련성을 이끕니다. ADEPT와 상보적입니다.
- **확산 제어기 (DexGen[25])**: 텔레옵 명령 보정. RL로 만든 전이로 학습한다는 점은 비슷하나 사후학습 RL이 없습니다.
- **오프라인 → 온라인 RL (Cal-QL[7] 등)**: 보수적 가치 보정으로 같은 문제(사전학습 가치 오정렬)를 오프라인 데이터 위에서 다룹니다. ADEPT는 온-정책 PPO 위에서 비평가를 새로 학습하는 쪽입니다.

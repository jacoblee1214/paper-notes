# 02. 선행연구와 관련 자료: TeleDexter (arXiv:2607.11481)

이 문서의 계보와 구성요소 출처는 논문 본문과 참고문헌([n])에서 가져온 것입니다. 각 선행 논문을 이번에 다시 읽지는 않았고, 요약은 참고문헌의 제목과 본문 서술, 그리고 이 저장소에서 이미 분석한 논문의 결과에 근거합니다. 확인하지 않은 부분은 "미확인"으로 적었습니다.

## 1. 계보

### 1.1 운동학 리타게팅 텔레옵

사람 손 관절(또는 키포인트)을 로봇 손 관절로 최적화로 옮기는 계열입니다(1절, 2절).
- 비전 기반: DexPilot [6], AnyTeleop [5] (dex-retargeting의 벡터 정렬 손실; TeleDexter의 DexRT 기준선과 1단계 리타게팅이 이것), Bunny-VisionPro [7], Open-TeleVision [8], ACE [18].
- 착용형: DOGlove [9], DexOP [10], DexExo [11], DexCap [19](MoCap).
- 신경 매핑: GeoRT [12](기하 목적 함수로 학습한 리타게팅 네트워크; TeleDexter의 GeoRT 기준선).

한계는 접촉력과 물체 관성을 모델링하지 않는다는 점이고, TeleDexter는 이를 "물체를 떨어뜨리는 원인"으로 지목합니다(1절). 이 계열의 같은 연도 후속은 DexFLEX 등에서 보입니다(§3).

### 1.2 학습한 동역학 사전분포로 텔레옵 보정

- **DexterityGen (DexGen)** [13]: 시뮬레이션의 AnyGrasp-to-AnyGrasp RL 정책 롤아웃으로 생성 모델(확산)을 학습해 거친 텔레옵 명령을 접촉이 풍부한 손 행동으로 바꿉니다. TeleDexter가 가장 가까운 대안으로 지목하고(2절) 기준선으로 쓰지만, 공식 구현이 없어 재구현했습니다(부록 D). 차이는 (a) TeleDexter는 생성 모델 없이 RL 정책이 직접 저수준 제어기, (b) 훈련 목표가 합성 파지 전이가 아니라 사람 손-물체 참조 모션입니다(1절).

### 1.3 시뮬레이션 RL 손 안 조작

- 과제별 정책: OpenAI Dactyl [24], Rubik's cube [25], Dextreme [26], HORA [27], Visual Dexterity [28], AnyRotate [29], DexNDM [30], simple-to-complex 재배향 [31], 펜 돌리기 [32].
- TeleDexter는 이들의 한계를 "과제별 보상 설계"로 요약하고(2절), 행동 공간 잔차 매개변수화는 HORA, 외력 무작위화는 Visual Dexterity, 커리큘럼·밀집 보상 모양은 ManipTrans [23]에서 가져옵니다(부록 C.1, C.2, C.3, C.7).

### 1.4 사람 손-물체 참조로 RL 가이드

- ManipTrans [23] (같은 연구실, 잔차 학습), PQS [33] (준물리 시뮬레이터 전이), 물체 중심 사람 모션 [34], DexTrack [35] (신경 추적 제어).
- 이 계열은 궤적당 하나의 정책이나 프레임별 추적이라 손 안 재배향·finger gaiting이 드물다고 서술하고(2절), TeleDexter는 **연속 부분목표 + 희소/밀집 하이브리드 보상**을 차별점으로 둡니다. 이 차별이 실제로 효과가 있는지는 시뮬 표 4에서만 비교됩니다(`04` C4).

### 1.5 도구 사용과 도구 조작 정책

- SimToolReal [42] (물체 중심 정책, 추론 시 참조 동작 필요): TeleDexter의 기준선(표 1의 † 카테고리별, ‡ 전체). 텔레옵 방법이 아니라 "학습된 도구 조작의 참고선"으로 쓴다고 명시합니다(4.1절).

### 1.6 자율 정책 학습

Diffusion Policy [43] (Conv-UNet, 부록 E)로 텔레옵 시연 50개를 모방합니다. DINOv2 ViT-S/14 인코더 두 개는 고정입니다.

## 2. 코드와 저장소 (2026-11-24 확인)

| 저장소 | 관계 | 라이선스 | 별 · 마지막 push | 상태 |
|---|---|---|---|---|
| (공식 코드 없음) | TeleDexter 구현 | — | — | 논문 abs, 프로젝트 페이지, GitHub 검색(`teledexter`, `bigai-dex`)에서 코드 링크·저장소를 찾지 못함 |
| [bigai-dex/bigai-dex.github.io](https://github.com/bigai-dex/bigai-dex.github.io) | 프로젝트 페이지 소스(정적, 영상 포함 약 217 MB) | 없음 | 0 · 2026-07-16 | 코드 아님. 복제하지 않음 |
| [ice4133/TeleDexter](https://github.com/ice4133/TeleDexter) | 이름만 같은 별개 프로젝트(WUJI Hand 운동학 리타게팅 코드, 2026-08-20 생성) | 없음 | 0 · 2026-08-21 | 이 논문과 관계 미확인. 읽지 않음 |
| [dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting) | 기준선 DexRT와 1단계 리타게팅의 구현 [5][6] | MIT | 1,321 · 2025-08-21 | 이번엔 읽지 않음 |
| [facebookresearch/GeoRT](https://github.com/facebookresearch/GeoRT) | 기준선 GeoRT 공식 구현 [12] | NOASSERTION(라이선스 파일 확인 필요) | 241 · 2025-09-02 | 이번엔 읽지 않음 |
| [tylerlum/simtoolreal](https://github.com/tylerlum/simtoolreal) | 기준선 SimToolReal 공식 구현 [42] | MIT | 439 · 2026-09-21 | 이번엔 읽지 않음 |
| [ManipTrans/ManipTrans](https://github.com/ManipTrans/ManipTrans) | 커리큘럼·밀집 보상 출처 [23] (같은 연구실) | GPL-3.0 | 350 · 2025-10-10 | 이번엔 읽지 않음 |
| [jayeshs999/sapg](https://github.com/jayeshs999/sapg) | 최적화기 SAPG [38] | MIT | 88 · 2024-09-17 | 이번엔 읽지 않음 |
| [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | 자율 정책 구조 [43] | MIT | 4,743 · 2024-12-24 | 이번엔 읽지 않음 |
| [NVIDIAGameWorks/kaolin](https://github.com/NVIDIAGameWorks/kaolin) | 리타게팅 SDF [39] | Apache-2.0 | 5,199 · 2026-11-21 | 이번엔 읽지 않음 |
| [NVlabs/curobo](https://github.com/NVlabs/curobo) | 평활 에너지 [40] | Apache-2.0 | 2,039 · 2026-11-21 | 이번엔 읽지 않음 |
| [leap-hand/LEAP_Hand_API](https://github.com/leap-hand/LEAP_Hand_API) | LeapHand 하드웨어 API [41] | NOASSERTION | 179 · 2025-10-20 | 이번엔 읽지 않음 |
| DexterityGen 공식 저장소 | DexGen [13] | — | — | 찾지 못함. 부록 D도 "No official implementation is available"이라고 적음 |

복제한 저장소는 없습니다(`repo/`는 gitignore 대상으로만 둠). 공식 코드가 공개되는지는 프로젝트 페이지와 GitHub 검색으로 다시 확인해야 합니다.


## 4. 구성요소별 출처

| 구성요소 | 출처 | 비고 |
|---|---|---|
| 연속 부분목표 + 희소/밀집 하이브리드 보상 | 본 논문 | 점수 보상의 지수 커널과 가중치는 ManipTrans [23]를 따름 (C.2) |
| 커리큘럼(중력, 간격, 허용오차) | ManipTrans [23] | 허용오차 일정은 부록에 없음 (`03` K5) |
| RSI(참조 상태 초기화) | [23] | C.4 |
| 행동 매개변수화(잔차 + 데드존) | HORA [27] + 소프트 데드존(본 논문) | C.1 |
| 외력 무작위화 | Visual Dexterity [28] | C.7 |
| 무작위 행동 마스킹 | 본 논문 | 3.1절, C.8 |
| 최적화기 | SAPG [38] (6블록) | 표 10 |
| 시뮬레이터 | Isaac Gym [37] | |
| 벡터 리타게팅 1단계 | AnyTeleop [5], DexPilot [6] | C.5 |
| SDF 표면·관통 손실 | Kaolin [39] | C.5 |
| 평활 에너지 | cuRobo [40] | C.5 |
| 자율 정책 | Diffusion Policy [43], DINOv2 | 부록 E |
| 손 | LeapHand [41], SharpaWave (참고문헌 없음) | 4.1절 |

# 02. 선행연구와 관련 자료: DexFLEX (arXiv:2610.14372)

이 문서의 계보와 구성 요소 출처는 논문 본문과 참고문헌(번호 [n])에서 가져온 것입니다. 논문이 어떤 연구를 가리키는지의 요약은 참고문헌 **제목과 본문 서술에 근거**한 것이고, 각 선행 논문을 이 분석에서 직접 다시 읽지는 않았습니다. 확인하지 않은 부분은 "미확인"으로 적었습니다.

## 1. 계보

### 1.1 공유 자율과 명령 정제

DexFLEX가 풀려는 문제의 뿌리는 **사용자(또는 상위 정책)의 불완전한 명령을 학습된 보조가 고쳐 주는** 공유 자율입니다(§1, §2).

| 갈래 | 대표(논문 인용) | DexFLEX와의 관계 |
|---|---|---|
| 정책 블렌딩·의도 추론 | Dragan & Srinivasa [28], Javdani 외 [29], Aarno 외 [24] 등 | 명령과 자율을 섞는 고전. DexFLEX는 섞지 않고 후보 집합에서 **고름** |
| 강화학습·잔차 보정 | Reddy 외 [1], Schaff & Walter [2] | 사용자 명령에 잔차를 더하는 방식. DexFLEX는 잔차가 아니라 프라이어 샘플을 고름 |
| 확산 기반 정제 | Yoneda 외 *To the noise and back* [3] | 명령에 노이즈를 섞고 확산 프라이어로 되돌려 정제. DexFLEX의 **초안 시드 분기**(초안을 중간 노이즈 수준으로 흐려 같은 flow에서 시작)와 구조가 같고, 새 부분은 **결과 예측 + 점수** |
| 접촉·구속 안전 | Schwarting 외 [27](자동차 parallel autonomy) 등 | 최소 개입 개념. 그림 4의 "선택적 개입" 주장과 닿음 |

### 1.2 숙련 조작용 재사용 제어기 (foundation controller)

- **DexterityGen (DexGen)** [4] (Yin 외, 2025): 논문이 가장 가까운 선행연구로 지목합니다(§2). 텔레옵으로 프롬프트되는 숙련 기초 제어기이고, DexFLEX는 "상태 조건 모션 프라이어와 명령 의존 보정을 분리"한다고 차별화합니다. 비교 기준선 "DexGen"은 **확산 기반 prior-only 보정**으로 정의됩니다(§4.1). 이 기준선이 원 논문의 시스템 전체와 같은지, 저자 재구현인지는 부록(없음)에 있다고 하며 확인할 수 없습니다. 공식 코드는 GitHub 검색(`dexteritygen`, `dexterity gen`)에서 찾지 못했습니다(2026-11-22).
- **로봇 foundation model**(RT-1 [8], RT-2 [9], π0.5 [10]): 시각운동 end-to-end 정책. DexFLEX는 그 아래층(저수준 실행)이라는 위치 구분(§2).
- **접촉이 많은 숙련 조작**([11]–[18]): 촉각·시각 표현, UniDexGrasp [12], Lodestar [14](같은 제1저자), 촉각 세계모델 계열 [13][15][16][17][18].

### 1.3 촉각을 쓰는 숙련 조작

*Rotating without seeing* [5], *General in-hand rotation with vision and touch* [6], *Dexterity from touch* [7], *Learning visuotactile skills with two multifingered hands* [30], *NeuralFeels* [31], *FingerEye* [32]. 논문의 평가는 이들이 "**과제별 정책이나 표현 학습기로** 쓰인다"는 점이고, DexFLEX는 촉각을 **명령 아래의 재사용 가능 제어기**에 둔다고 위치합니다(§1, §2). 촉각 센서는 FlexiTac [37](3D-ViTac [36]의 후속 하드웨어).

### 1.4 "후보를 꿈꾸고 고르기"

잠재 세계모델로 후보 행동의 결과를 예측해 고르는 구조는 모델 기반 계획과 같은 계열입니다. 논문은 시각-촉각 세계모델 [15][16][17][18]을 인용하지만 **이들과의 방법적 차이를 서술하지 않습니다**(참고문헌에만 있음). DexFLEX의 차별점은 (a) 세계모델이 **짧은 청크의 종단 잠재만** 예측하고 (b) 후보가 **flow 프라이어 샘플 + 초안 시드 샘플**로 제한되며 (c) 점수가 **초안 정렬과 예측 접촉 결과를 함께** 본다는 것입니다(§3.2–3.3).

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| consistency-flow 프라이어, 경량 트랜스포머 + 교차 어텐션 | ManiFlow [33] | shortcut-consistency 손실은 [33]과 부록 참조(부록 없음) |
| flow matching | Lipman 외 [38] | 정책 백본과 프라이어 학습 |
| 초안 시드 분기(노이즈 섞고 되돌림) | *To the noise and back* [3]의 아이디어와 같은 계통 | 노이즈 수준은 부록(없음) |
| 전문가 청크 | Any-Goal-Pose Reaching RL 전문가, SimToolReal [34] | Isaac Gym 데이터 |
| 촉각 센서 | FlexiTac [37], 3D-ViTac [36] | 손끝당 10x6 태셀 |
| IK·텔레옵 스택 | Mink [35], Rokoko 글러브, Vive tracker | 50 Hz |
| 정책 학습 백본 | 이미지 기반 flow matching 정책 | 인용이 [38](flow matching 일반론)이라 정책 구조의 출처는 불분명 |
| 대조 순위 점수 | 논문은 출처를 달지 않음 | 부록(없음) |
| 기준선 DexGen | DexterityGen [4] | 확산 기반, 결과 예측·점수 없음 |

## 4. GitHub와 코드 자료 (확인 2026-11-22)

**공식 코드는 없습니다.** 프로젝트 페이지 저장소만 공개되어 있고 코드 링크가 없습니다. 논문 PDF에는 코드·데이터 주소가 없고(링크는 `dex-flex.github.io`와 Mink 인용뿐), 저자 계정 `dex-flex`는 공개 저장소가 하나(웹사이트)입니다.

| 자료 | 링크 | 라이선스 | 별 / 마지막 push | 비고 |
|---|---|---|---|---|
| **공식 코드** | 없음 | - | - | 논문·프로젝트 페이지·GitHub 검색에서 찾지 못함 |
| 프로젝트 페이지(웹사이트) | https://dex-flex.github.io, 저장소 `dex-flex/dex-flex.github.io` | 표기 없음 | 별 0 / 2026-10-09 (`74db30b`) | 영상·표 복제·BibTeX. 내려받지 않음(HTML만 열람) |
| SimToolReal (전문가 출처) | https://github.com/tylerlum/simtoolreal | MIT | 437 / 2026-09-21 (`313d5ae`) | 메타데이터만. Isaac Gym·Sim 필요 |
| Mink (IK) | https://github.com/kevinzakka/mink | Apache-2.0 | 1,619 / 2026-11-19 (`1850518`) | 메타데이터만 |
| ManiFlow 정책 (consistency flow) | https://github.com/geyan21/ManiFlow_Policy | MIT | 222 / 2025-11-13 (`ef2f116`) | 메타데이터만. 프라이어 구조의 출처 [33] |
| FlexiTac | https://github.com/FlexiTac/FlexiTac | 표기 없음 | 190 / 2026-11-18 (`945d623`) | 촉각 하드웨어·펌웨어 |
| FlexiTac IsaacSim 시뮬레이션 | https://github.com/binghao-huang/FlexiTac-IsaacSim-Simulation | 표기 없음 | 59 / 2026-04-09 | 촉각 시뮬레이션. DexFLEX의 Isaac Gym 촉각 시뮬과 같은지는 미확인 |
| LEAP Hand MuJoCo 모델 | https://github.com/google-deepmind/mujoco_menagerie (`leap_hand/`) | 저장소 NOASSERTION, 폴더에 LICENSE 있음 | 4,401 / 2026-11-10 | 맥 축소판 후보(`04` §4) |
| LEAP Hand API | https://github.com/leap-hand/LEAP_Hand_API | NOASSERTION | 179 / 2025-10-20 | 하드웨어 제어 |
| DexterityGen (DexGen) | 공식 저장소 못 찾음 | - | - | 검색 `dexteritygen`, `dexterity gen` 결과 없음 |
| 제3자 복제 | 못 찾음 (`dexflex` 검색 결과 두 개는 무관한 저장소) | - | - | |

## 5. 가까운 대안과 동시기 연구

- **초안을 조건으로 넣는 직접 학습.** 합성 프록시 초안으로 (초안 → 교정 행동)을 곧바로 회귀하는 조건부 정책. DexFLEX는 이를 피하려고 "초안은 선택에만 쓴다"고 했는데(§3.2 "no paired examples"), **프록시 초안으로 점수를 학습**하므로 조건 정책의 분포 이동 문제(합성 초안 대 실제 텔레옵 초안)를 점수가 대신 떠안습니다. 직접 비교는 논문에 없습니다.
- **사람 보정 학습.** 사람이 로봇 정책 출력을 고치는 DAgger 계열. DexFLEX는 사람 보정 데이터 없이 시뮬레이션 전문가만 씁니다.
- **논문이 같은 시기에 인용한 촉각 세계모델**(OmniVTA [15], VTAM [17], touch dreaming [18])은 2026-03~04의 arXiv이며 이 분석에서 열어 보지 않았습니다.

# 02. 선행연구, 구성 요소의 출처, GitHub와 자료

## 1. 계보

H2O는 세 갈래를 한 시스템에 합칩니다. (a) 물리 기반 아바타 제어에서 온 대규모 모션 모방 RL, (b) 레그드 로봇의 시뮬레이션 보행 sim-to-real 레시피, (c) 사람 포즈 추정기를 이용한 원격조작입니다.

| 갈래 | 대표 선행 | H2O가 가져온 것 | 바꾼 것 |
|---|---|---|---|
| 모델 기반 전신 원격조작 | Ishiguro 외(2018, 2020), Ramos & Kim(2018, 2019), Penco 외(2018), Koenemann 외(2014) | 문제 설정(전신, 실시간) | 접촉 측정, 외골격, 힘 센서, 단순화 모델 없이 RGB 카메라 입력과 RL |
| 사람-휴머노이드 리타게팅 | Darvish 외(2019), Cisneros-Limón 외(2024), Radosavovic 외(2024, 다음 토큰 예측) | 직접 리타게팅 아이디어 | 대규모 AMASS 전체를 H1에 맞추고 걸러냄(§IV) |
| 물리 기반 아바타 모방 | DeepMimic(Peng 2018), AMP, ASE, **PHC**(Luo 2023, ICCV), **PULSE**(Luo 2024, ICLR) | 대규모 추적 정책, 특권 정책, hard-negative mining, 목표 상태 설계 | 하드웨어 한계 반영(토크, 관절 한계), 실물 가능 관측만 사용 |
| 레그드 sim-to-real | Rudin 외(2022, legged_gym), Li 외(2024, 이족 보행 RL), Radosavovic 외(2023), Siekmann 외(2021), Campanaro 외(2023, RFI), He 외(2024, 낮은 장애물 지형) | 학습 환경, PPO, DR 구성, 보상 정규화, 지형 | 전신 추적 목표 조건부로 확장 |
| 같은 시기 | **ExBody**(Cheng 외, arXiv:2402.16796): 상체 동작 모방과 하체 속도 추적, 오프라인 모션 | 비교 대상(§I, §II-B) | 전신 하체까지 실시간 추적 |

**후속.** OmniH2O(arXiv:2406.08858, CoRL 2024)는 같은 팀이 H2O를 확장한 후속으로 VR 3점/RGB/자율 정책 입력과 DAgger 증류를 더했고, H2O를 기준선으로 쓴 시뮬 성공률 94.10 대 87.52%를 보고합니다. 이 논문의 표 III 값(72.5%)과 다른 값입니다. 평가 집합과 프로토콜이 달랐을 가능성이 크지만 후속 논문의 H2O 재평가 방법은 직접 확인하지 못했습니다(`03` K7).

## 2. GitHub와 자료 (확인 2026-11-23)

| 저장소/자료 | 관계 | 라이선스 | ★ / fork | 생성 / 마지막 push | 상태 |
|---|---|---|---|---|---|
| [LeCAR-Lab/human2humanoid](https://github.com/LeCAR-Lab/human2humanoid) | **공식 구현** (H2O와 OmniH2O 공용, README가 "Official Implementation for H2O and OmniH2O") | README와 `LICENSE-CC-BY-NC-4.0.md`: CC BY-NC 4.0(상업 이용 금지), `legged_gym`과 `rsl_rl`은 ETH Zurich, NVIDIA 상속 라이선스. GitHub API license: 없음 | 1,083 / 97 | 2024-09-29 / 2025-02-21 | 얕은 복제 `repo/` 커밋 `750f1fa` (2025-02-21, "fixing eval during training"). 정적 읽기만. 실행, 설치 안 함. 열린 이슈 38개 |
| [human2humanoid.com](https://human2humanoid.com/) | 프로젝트 페이지 | - | - | - | `paper/project_page.html` 저장. IROS 2024 Oral 표기, 영상, 논문 PDF 링크 |
| CMU Box `amass_phc_filtered.pkl` | 공식 필터 통과 모션 데이터(README "Downloading Full retargeted motion dataset after feasibility filter") | 미확인(AMASS와 SMPL 라이선스를 따를 것으로 보임) | - | - | **받지 않음.** 크기와 시퀀스 수 미확인(`04` D11) |
| [ZhengyiLuo/PHC](https://github.com/ZhengyiLuo/PHC) | 특권 모방 정책의 원본 구현 (`phc/`가 H1용으로 포함) | NOASSERTION (사용자 정의 라이선스) | 1,310 / 132 | 2023-10-15 / 2025-08-21 | 읽지 않음. 이 저장소 안의 `phc/` 사본만 정적으로 읽음 |
| [ZhengyiLuo/PULSE](https://github.com/ZhengyiLuo/PULSE) | hard-negative mining 원본 (논문 §IV-B) | 없음(API) | 378 / 26 | 2024-03-09 / 2025-02-08 | 읽지 않음 |
| [jeffffffli/HybrIK](https://github.com/jeffffffli/HybrIK) | 3D 포즈 추정기 [27] (논문 §VI-B) | MIT | 1,635 / 189 | 2020-11-30 / 2025-01-08 | 읽지 않음. 공식 저장소는 이를 가리키기만 하고 연동 코드는 없음 |
| [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym) | 학습 환경 기반 [55] | NOASSERTION (파일 내 BSD-3) | 3,196 / 586 | 2021-10-18 / 2025-05-29 | 읽지 않음. 수정본이 공식 저장소에 포함 |
| NVIDIA Isaac Gym Preview 4 | 시뮬레이터(README 설치 2번) | NVIDIA 배포 라이선스 | - | - | **Linux + NVIDIA GPU 전제.** 내려받지 않음 |

- **가중치는 공개되어 있지 않습니다.** 저장소에 H2O 정책 파일이 없습니다(`.pt`는 `resources/actuator_nets/anydrive_v3_lstm.pt` 하나이고 코드에서 참조되지 않음). 이슈 #56 "Pretrained Model"은 닫혀 있지만 첨부를 확인하지 못했습니다.
- **저장소 이슈에서 확인한 것(외부 사용자 글이라 참고용).** #39 마찰 범위 음수 질문(답 없음), #45 "Feasibility Filter" 구현 질문(답 없음), #58 `feet_air_time_teleop` 보상이 항상 0, #65 기본 설정이 `rewards_teleop`을 못 찾음, #41 공저자가 "평가 스크립트를 제공하지 않았다"고 답함, #66 교사 학습이 1M 반복에 수 주로 추정됨.

## 4. 구성 요소별 출처

| 구성 요소 | 출처 | 논문 위치 |
|---|---|---|
| SMPL 인체 모델, AMASS 데이터 | Loper 외(2015), Mahmood 외(2019) | §III-B |
| 특권 모방 정책, 목표 상태, 성공 지표 | PHC (Luo 외, ICCV 2023) | §IV-B, §VI-A |
| hard-negative mining, 단일 정책 대규모 모방 | PULSE (Luo 외, ICLR 2024) | §IV-B |
| 학습 환경과 PPO 구현 | legged_gym, rsl_rl (Rudin 외 2022), Isaac Gym | §V |
| 토크 RFI | Campanaro 외 (2023) | 표 II |
| 지형(평지, 거침, 낮은 장애물) | He 외, Agile but safe (2024) | 표 II |
| 발 체공 시간 보상 | Rudin 외 (2022) | 표 I |
| 3D 포즈 추정 | HybrIK (Li 외, CVPR 2021) | §VI-B |
| 형상 맞춤과 리타게팅 | 이 논문(경사 하강, Adam) | §IV-A |
| sim-to-data 필터 | 이 논문 | §IV-B |

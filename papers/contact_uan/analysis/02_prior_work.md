# 02. 선행연구와 코드 저장소: Contact-UAN (arXiv:2610.18991)

## 1. 계보

Contact-UAN이 서 있는 줄기는 "시뮬레이터의 물리 구조는 두고 구동기 쪽의 차이만 데이터로 메운다"입니다. 이 줄기에서 논문이 이어받은 것과 새로 더한 것을 나눠 보면 다음과 같습니다.

| 갈래 | 논문 | 한 줄 | Contact-UAN과의 관계 |
|---|---|---|---|
| 지도학습 구동기 망 | Hwangbo et al., *Learning agile and dynamic motor skills for legged robots*, Science Robotics 2019 [7] | 출력 토크를 측정해 토크 입력 이력에서 토크를 예측 | 같은 "구동기 채널만 학습" 발상. 토크 센서가 필요해 휴머노이드에서는 어렵다고 논문이 지적(서론) |
| 비지도 구동기 망 (UAN) | Fey, Margolis, Peticco, Agrawal, *Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation*, RSS 2025 (arXiv:2502.10894) [8] | 하드웨어 궤적을 시뮬레이터에서 같은 명령으로 재생해 관절 위치가 맞도록 잔차 명령을 PPO로 학습 | **직접 선행.** 허공에 뜬 팔처럼 키네마틱스만으로 상태를 맞출 수 있는 경우였음. 이 논문은 접촉이 있는 떠 있는 베이스(floating base)로 확장 |
| 잔차 델타 동역학 | He et al., *ASAP: Aligning Simulation and Real-World Physics…*, RSS 2025 (arXiv:2502.01143) [28] | 모션캡처로 얻은 실제 궤적에서 델타 행동 모델을 학습 | 접촉이 있는 휴머노이드 재생의 선행. 논문은 "모션캡처에 의존"한다는 점과 "구동기 동역학이 아니라 인터페이스별 격차를 겨냥"한다는 점으로 구분 |
| 고유감각으로 접촉 추정 | Margolis, Wang, Fey, Agrawal, *SoftMimic* (arXiv:2510.17792) [4], Portela et al., *Learning force control for legged manipulation*, ICRA 2024 [30] | 짧은 고유감각 이력에서 접촉과 외력을 추정 | 2.1절이 "접촉력은 고유감각 이력에서 추론할 수 있다"는 근거로 인용 |
| 완전 학습 세계 모델 | Li, Krause, Hutter, *Robotic World Model* (arXiv:2501.10100) [19], *Uncertainty-aware RWM / RWM-U* (arXiv:2504.16680) [33] | 데이터로 동역학 전체를 학습해 상상 속 롤아웃에서 정책 최적화 | 부록 F의 데이터 효율 비교 대상(저자의 G1 이식본) |
| 같은 줄기의 위치 손실 | GenAN (arXiv:2604.09487, CoRL 2026) | 미분 가능 시뮬레이터로 한 스텝 굴린 위치가 맞도록 구동기 망 학습. 비교 기준으로 UAN 사용 | 이 논문이 인용하지 않은 같은 시기의 동일 계열 |
| 모방 추종 틀과 시뮬레이터 | Liao et al., *BeyondMimic* (arXiv:2508.08241) [2], Zakka et al., *mjlab* (arXiv:2601.22074) [29] | G1 노미널 파라미터, 점프 추종 방식, 시뮬레이션 프레임워크 | 노미널 시뮬레이터와 점프 과제의 출처. 공식 코드는 mjlab의 사본 위에 UAN을 얹은 구조(`04`) |
| 구동기 쪽 무작위화 대신 정확한 모델 | Bjelonic et al. (arXiv:2509.06342) [15], ZEST (arXiv:2602.00401) [1] | 최근 휴머노이드·사족 시스템이 무작위화를 줄이고 모델을 정확하게 | 논문 6절이 같은 흐름이라고 정리 |

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 이 논문에서 달라진 점 |
|---|---|---|
| 잔차를 입력(명령) 쪽에 두기, 보상 = 재생 위치 오차 | UAN (RSS 2025) | 보상에 관절 속도 항(−0.1)과 잔차 변화율(−0.01) 추가, 창 24스텝 |
| 같은 하드웨어 구동기 간 파라미터 공유, 짧은 이력 | Hwangbo 2019, UAN | 29관절을 구동기 계열 5개로 묶음(표 3) |
| 보행 데이터 수집 정책 | Walk-These-Ways [31] | 보행 주파수·발 높이 명령으로 스탠스 길이와 충격 다양화(부록 C) |
| 접촉 일관 리셋 | **새 기여** | 접촉 확률 → 지지 발 높이·기울기 회귀 → 확률 가중 융합(식 4–6) |
| 절대 높이를 입력에서 제외 | **새 기여(설계 선택)** | 높이 복사 지름길 방지 |
| 토크-속도 포락선 클리핑(식 7) | 부록 D가 [54]를 인용 | 달리기와 점프에 적용 |
| 정책 최적화 | PPO [32], RSL-RL [53] | 표 5 |
| 평가 틀 | mjlab G1 속도 추종, 힘 제어 [30], 점프 추종 [2] | 힘 제어는 가상 힘장 + 20스텝 이력 |

## 4. 코드 저장소와 라이선스 표

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| [nolie-rolie/contact-uan](https://github.com/nolie-rolie/contact-uan) | **공식 구현** (mjlab 1.4.0 사본 + UAN) | Apache-2.0 (LICENSE 파일, API 확인). 단 `third-party/openhomie/`는 CC BY-NC 4.0 고지 | 31 | 2026-10-15 | 커밋 `55de136a33f44b7f22190c5f9de882266268b5ab`(2026-10-15, 커밋 30개), 약 58.9 MB, 사전학습 정책·하드웨어 데이터 없음 |
| [contact-uan.csail.mit.edu](https://contact-uan.csail.mit.edu/) | 프로젝트 페이지 | - | - | - | 동영상, Q&A |
| [nolie-rolie/athletic-loco-manipulation](https://github.com/nolie-rolie/athletic-loco-manipulation) | 선행 UAN(RSS 2025) 코드 | 라이선스 정보 없음(API) | 53 | 2025-10-16 | 프로젝트 [uan.csail.mit.edu](https://uan.csail.mit.edu/) |
| [mujocolab/mjlab](https://github.com/mujocolab/mjlab) | 기반 프레임워크 | Apache-2.0 | 3,471 | 2026-11-26 | Contact-UAN 저장소는 이 프레임워크의 사본 위에 있음 |
| [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) | BeyondMimic | MIT | 2,442 | 2026-10-16 | 2026-10-16 이후 갱신 없음(후속은 mjlab) |
| [LeCAR-Lab/ASAP](https://github.com/LeCAR-Lab/ASAP) | ASAP | MIT | 2,148 | 2026-01-06 | 모션캡처 기반 델타 행동 |

- 별 수와 push 일자는 2026-11-26 GitHub API 값입니다. 공식 저장소의 마지막 커밋은 논문 제출 이틀 뒤입니다.
- 공식 저장소는 `README`에서 "훈련은 Linux와 NVIDIA GPU가 필요하고, CPU 전용(macOS 포함) 오프라인 작업은 `--extra cpu`"라고 안내합니다.

# 02. 선행연구와 관련 저장소

## 1. 선행연구 계보 (구성 요소별 출처)

| 구성 요소 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| VLM + flow matching action expert | π0 (Black 외, arXiv:2410.24164) [4], PaliGemma | 백본과 action expert. 가중치는 π0 base(`lerobot/pi0_base`)에서 초기화 (부록 A.1) |
| 구현 기반 | LeRobot v0.4.3 [32], OpenPI | π0 PyTorch 포트를 수정해 expert를 하나 더 둠 |
| 기하 임피던스 제어(GIC) | Seo 외 [11, 12] (SE(3) 임피던스, Lie 군/대수 포텐셜 비교) | 제어 법칙의 틀. 식 (3)–(5). 실제 구현은 단순화 변형 (부록 A.2) |
| 힘 기반 강성 라벨 규칙 | Adaptive Compliance Policy, ACP (Hou 외, ICRA 2025) [7] | 식 18의 구간 선형 강성, 힘 방향 첫 축 낮은 강성(식 6 계열) |
| 시연에서 강성 학습 | Comp-ACT [8], DIPCOM [9] (강성 라벨을 조작자 감독으로 수집) | 문제 설정(가변 강성을 데이터에서 배운다) |
| 강성 추정 계열 | EMG [23], 시연 분산 [24, 25], SPD 다양체 접촉력 [26] | 관련 연구 서술 |
| 힘/토크 입력 VLA | ForceVLA (arXiv:2505.22159) [13], TA-VLA (arXiv:2509.07962) [14], VTLA [18] | 기준선(ForceVLA, TA-VLA). VTLA는 촉각 센서가 없어 제외 |
| 출력 확장 VLA | ForceVLA2 (arXiv:2603.15169) [19], CompliantVLA-adaptor (arXiv:2601.15541) [10] | 가장 가까운 연구로 서술. 구현 비공개라 비교하지 않음 |
| 모방학습 기준선 | Diffusion Policy [27], ACT [28] | 표 1의 단일 run 기준선 |
| 하드웨어 | UMI 손가락 [31], ATI Axia F/T, Franka FR3 | 부록 B.1 |

**CompVLA의 새로움을 어디서 읽을 수 있는가.**
- ACP는 확산 정책이 기준 자세, 가상 목표, 스칼라 강성을 내는 구조입니다. CompVLA는 같은 종류의 라벨(힘 방향 낮은 강성, 힘/강성 크기의 가상 변위)을 VLA로 옮기고, **강성과 변위를 별도 expert가 기준 궤적을 조건으로 받아** 예측하며, 대각 강성 6축과 방향 프레임을 회전·병진으로 나눕니다(§2, §4).
- ForceVLA와 TA-VLA는 센서 입력을 늘리고 출력은 자세로 남깁니다. CompVLA는 센서를 라벨로만 쓰고 출력을 늘립니다(§2 "input augmentation 대 output augmentation").
- 논문 자신이 밝힌 가장 가까운 연구는 CompliantVLA-adaptor로, 별도 VLM이 단계와 힘 측정에서 강성을 고릅니다. 이 연구는 비공개 구현이라 직접 비교가 없습니다(§2).

## 2. 코드와 저장소 (GitHub 표, 2026-11-24 확인)

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [compvla/CompVLA](https://github.com/compvla/CompVLA) | 공식 코드 (모델, 학습, 데이터 변환) | Apache-2.0 | 0 | 2026-11-05 | 커밋 `8393e22` ("Initial release"), 얕은 복제 `repo/`, 정적 읽기만 |
| [compvla/compvla.github.io](https://github.com/compvla/compvla.github.io) | 프로젝트 페이지 [compvla.github.io](https://compvla.github.io) | 없음(API: null) | 0 | 2026-11-05 | 페이지 HTML을 읽음. 영상은 "TBA" 주석 처리 |
| 원자료 (Google Drive, README 링크) | 시연 원자료, 과제당 zip 1개 (총 5.8 GB, 풀면 7.1 GB) | CC BY 4.0 (README) | - | - | 받지 않음(크기 보고만) |
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | 기반 프레임워크 (v0.4.3 고정) | Apache-2.0 | 약 29k | 2026-11-24 | 받지 않음 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0 원 구현 | Apache-2.0 | 약 14.7k | 2026-08-24 | 받지 않음 |
| [ft-robotic/ForceVLA](https://github.com/ft-robotic/ForceVLA) | 기준선 ForceVLA 공식 | Apache-2.0 | 약 121 | 2026-01-27 | 받지 않음 |
| `lerobot/pi0_base` (Hugging Face) | 초기화 가중치, README가 리비전 `26b99b9` 고정 | 확인 안 함 | - | - | 받지 않음 |

- 저장소는 2026-11-05에 올라왔고 커밋이 하나입니다(v2 제출 전날). 이슈, 포크, 별은 없습니다.
- TA-VLA, DP, ACT의 공식 저장소는 확인하지 않았습니다. 논문 §5.1이 이들을 같은 GIC에 연결해 돌렸다고 적지만 연결 코드는 CompVLA 저장소에 없습니다(`04` D2).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **ACP의 정당화:** 힘 방향 속도를 제어하지 않으면(그 방향 강성을 낮추면) 접촉 제약을 만족하는 해가 있다는 정리와 구간 선형 강성이 CompVLA 라벨 규칙의 근거입니다. CompVLA 논문은 이 정당화를 다시 보이지 않고 인용합니다(§2, 부록 B.2).
- **GIC의 안정성:** Seo 외의 에너지 기반 증명은 고정 이득, F_ext = 0, 완전한 GIC에 대한 것입니다. v2는 시변 강성과 단순화 구현에서는 증명이 이 제어기를 "motivate"할 뿐이라고 적습니다(부록 A.2 끝).
- **π0 구조:** blockwise causal attention, 25스텝 청크, flow matching의 시간 샘플링(Beta)은 OpenPI와 LeRobot 구현을 따릅니다. 공식 코드에서 확인했습니다(`04` §2.2).

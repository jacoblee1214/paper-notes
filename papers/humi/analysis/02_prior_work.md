# 02. 선행 연구와 코드 지형

## 1. 선행 연구 계보

HuMI는 세 줄기를 잇습니다.

1. **로봇 없는 조작 데이터 수집(UMI 계열).** UMI [19](Chi 외, 2024)가 손잡이형 그리퍼 + GoPro + SLAM으로 현장 시연을 모으고, UMI on Legs [20](다리 달린 로봇, 조작 중심 전신 제어기), UMI-on-Air [21](드론), DexUMI [46], 이동 조작으로 확장한 Xu 외 [51]이 뒤를 잇습니다. 이들은 **손목(EE) 궤적만** 담습니다. HuMI의 차이는 골반·발 궤적을 추가하고 IK로 전신 자세를 복원하며, 그 위에 휴머노이드 전용 저수준 제어기와 인터페이스를 얹은 것입니다(§1, §2).
2. **휴머노이드 전신 추적 제어기.** BeyondMimic [27], SONIC [16], HOVER [28], OmniH2O [11], TWIST/TWIST2 [7, 8], HuB [25]가 "참조 동작을 추종하는 RL 정책"의 기반입니다. HuMI는 이 보상 틀(식 §3.2의 r(ē,σ))을 그대로 쓰되 **EE 허용 오차를 속도에 따라 줄이고**(적응형 EE 보상) **기준 속도를 무작위로 늦춰**(가변 속도 증강) 센티미터 이하 EE 정밀도를 노립니다. 공식 코드의 보상·커리큘럼·모션 명령 구조가 BeyondMimic 코드베이스와 매우 비슷합니다(구조 유사, 추정. `04` §0).
3. **텔레옵 기반 휴머노이드 데이터.** TWIST2 [8](휴대형 전신 텔레옵), HOMIE [5], CLONE [6], HumanPlus [10], OmniH2O [11], Open-TeleVision [13] 등이 비교 대상입니다. 논문은 이들을 "조작자가 균형과 추종 오차를 보상해야 하므로 처리량이 조작자에 묶인다"고 요약합니다(§1, §2).

**가까운 대안.**
- *BifrostUMI* [40](arXiv:2605.03452, 2026-05, v2 2026-07): 같은 문제("robot-free 시연 → 휴머노이드 전신 조작")를 다루는 동시기 연구입니다. v3에서 처음 인용되었고(v1의 "최초" 주장은 삭제), 본문은 "좁은 과제 집합·고정 환경에 머문다"고만 서술합니다(§2). 코드 여부와 수치는 확인하지 않았습니다.
- *OmniRetarget* [22], *GMR* [24]: 사람 동작을 로봇 형태로 **척도 조정**하는 리타게팅. HuMI는 척도 조정이 물체와의 시각-공간 정렬을 깨뜨린다는 이유로 쓰지 않고 시연자 적응으로 대신합니다(§3.1). 대가는 시연자가 G1 작업 공간에 맞춰 움직여야 한다는 점입니다(부록 A.2에서 손-골반 거리가 14 cm 변함).
- *Humanoid-VLA* [38], *VisualMimic* [39], *WholeBodyVLA* [15]: 사람 영상·VLA 경로. 논문은 "전신 EE 중심 정밀도가 약하다"는 정도로만 비교합니다.

## 2. GitHub와 자료 (조회일 2026-11-23)

| 저장소/자료 | 관계 | 라이선스 | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [Richard-coder-Nai/HuMI](https://github.com/Richard-coder-Nai/HuMI) | **공식 코드**(데이터 수집, 고수준 학습, 저수준 학습, 두 배포 코드) | MIT (고수준 UMI 파생 부분은 Columbia MIT 고지 별도) | 71 | 2026-04-21 | 커밋 `37a85a696f7558710bf0bd4c5a1641d37265a3e6`, 얕은 복제 146 MB(파이썬 81.8K줄, 파일 801개), 정적 읽기만. 이슈 1건(URDF와 메시 요청, 미해결), 릴리스 없음, 커밋 11개 |
| [프로젝트 페이지](https://humanoid-manipulation-interface.github.io/) | 논문 v3의 주소(Vue 단일 페이지, 소스는 humanoid-manipulation-interface/humanoid-manipulation-interface.github.io, 2026-11-17 갱신) | - | 0 | 2026-11-17 | 페이지 번들에 논문, 코드, 데이터 링크 확인 |
| [HF 컬렉션 Richard-Nai/humi](https://huggingface.co/collections/Richard-Nai/humi) | 데이터셋 7개 | CC BY 4.0 | 다운로드 118–1,761 | 2026-03-12 | LeRobot 형식 5개(Proposal 103에피소드·0.24 GB, Walk-Clean-Table 105·0.31 GB, Unsheathe 105·0.22 GB, Toss 104·0.13 GB, Squat-Pick-Ground 410·3.46 GB), Zarr 32.0 GB, 원시 데이터 150.3 GB. **받지 않음** |
| 하드웨어 CAD | 데이터 수집 그리퍼, G1 배포 그리퍼 | 공개 Google Drive(라이선스 미표기) | - | - | 확인만 |
| [real-stanford/universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface) | UMI 원본(고수준 코드의 모체) | MIT | 1,647 | 2026-05-26 | 받지 않음 |
| [real-stanford/umi-on-legs](https://github.com/real-stanford/umi-on-legs) | UMI on Legs | MIT | 570 | 2026-07-20 | 받지 않음 |
| [amazon-far/TWIST2](https://github.com/amazon-far/TWIST2) | 비교 대상 텔레옵 시스템 | MIT | 903 | 2026-09-23 | 받지 않음 |
| [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) | BeyondMimic(보상·모션 명령 원형) | MIT | 2,443 | 2026-10-16 | 받지 않음 |
| [kevinzakka/mink](https://github.com/kevinzakka/mink) | 시연 IK 솔버 | Apache-2.0 | 1,619 | 2026-11-19 | 받지 않음 |
| [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | 저수준 학습 환경 | BSD-3(API는 NOASSERTION으로 표시) | 8,701 | 2026-11-22 | 받지 않음 |
| [NVlabs/curobo](https://github.com/NVlabs/curobo) | 모션 로딩 FK | API 표시 Apache-2.0 | 2,037 | 2026-11-21 | 받지 않음 |

## 4. 구성 요소별 출처

| 구성 요소 | 출처 | HuMI의 변형 |
|---|---|---|
| 손잡이형 그리퍼 + GoPro + ArUco 폭 + 타임코드 동기 | UMI [19] | 상단 덮개를 추적기 마운트용으로 수정. G1용은 손목 요 모터 직접 구동 그리퍼(Wild LMA 영감, 부록 B.1) |
| 상대 키포인트 궤적 행동 표현 | UMI [19], UMI on Legs [20] | 골반 등 시각 피드백 없는 키포인트는 덩어리 내 상대 변환 |
| Diffusion Policy | Chi 외 [52], UMI 코드 | 손실을 흐름 정합으로 교체, ViT-B DINOv2 인코더, 5 Hz 덩어리 |
| 전신 추종 보상 틀 r(ē,σ) | BeyondMimic [27] | 적응형 EE 보상 + 커리큘럼 추가 |
| 교사-학생 + DAgger | [6, 7, 11, 28] | 학생 입력을 EE + 맹점 키포인트로 제한 |
| 전신 IK | Mink [64] + Viser [65] | 과제별 가중치와 CoM 과제(코드에만 있음, `04` D9) |
| 추적기 | HTC Vive Ultimate [56] | 기지국 없는 자체 추적 |
| 로봇 | Unitree G1 [59] | 25자유도(허리 요만 사용, 손목 요는 그리퍼에 사용) |

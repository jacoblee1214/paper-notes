# 02. 선행 연구와 관련 자료

인용 번호는 논문 참고문헌 번호입니다. 확인일은 2026-11-23입니다.

## 1. 선행 계보

WACL이 서 있는 줄기는 네 갈래입니다(논문 §2).

**(1) 팔 표면을 조작 인터페이스로 쓰는 연구.**
- 개념은 Eberman의 Whole-Arm Manipulation(MIT 석사논문, 1989 [1]).
- 학습 기반: 로봇 팔에 기구학적으로 맞춘 외골격으로 관절 공간 시연을 모으는 AirExo [10], AirExo-2 [25], 손에 맞춘 장치 DEXOP [26], 순응형 촉각 피부 휴머노이드의 과제별 sim-to-real 강화학습(Barreiros 외, Science Robotics 2025 [27]).
- 모델 기반: 평활화한 준동역학 접촉 모델의 전역 계획(Pang 외 [28], T-RO 2023), 텐서 분해 트리 탐색(Xue 외 [29]), 로봇 표면 표현 위의 연속 최적화(Levé 외 [30]).
- 저자 평가(§2): 이들 중 엔드이펙터 작업을 동시에 지키는 것은 없고, 학습 기반은 로봇 시연이나 촉각 피부가 필요하며 모델 기반은 물체 모델이 필요합니다.

**(2) 키포인트 조건부 모방 학습.** KALM [16](Fang, Huang, Mao, Shone, Tenenbaum, Lozano-Pérez, Kaelbling, ICRA 2025)의 키포인트 증류와 조건부 궤적 확산 정책을 그대로 가져옵니다. KALM 논문의 두 번째 저자가 이 논문의 첫 저자입니다. 같은 계열로 ReKep [36], CoPa [37](VLM이 키포인트나 부위 제약을 생성).

**(3) 인간 영상·맨손 시연.** Phantom [11], EgoMimic [12], 첫 저자의 이전 연구 Diffusion Imitation from Observation [13], HAND me the data [14], RoboCLIP [15]. 저자는 이들이 "로봇 행동을 제공하지 않고 사람-로봇 구현 간극이 있다"고 봅니다(§1). WACL은 사람 손끝을 링크 표면 접촉의 점 대용으로 쓰고, 로봇 시연 없이 정책을 학습합니다.

**(4) VLM을 제약·보상 생성에 쓰는 연구.** VoxPoser [31], MOKA [32], Code as Policies [33], Eureka [34], RL-VLM-F [35], ReKep [36], CoPa [37], AutoTAMP [38], 같은 연구실의 IMPACT [39](ICRA 2026, 환경의 접촉 허용 정도 추론). WACL은 VLM 출력을 **고정된 두 프리미티브 어휘**의 파라미터로 한정하고 하드 게이트로 걸러 안전을 보장하려 합니다.

## 2. 구성요소별 출처

| 구성요소 | 출처 | WACL이 한 것 |
|---|---|---|
| 키포인트 증류와 매처(식 1) | KALM [16], DINOv2 [47], FPFH [48], SAM3 [43] | 그대로 사용 |
| 조건부 궤적 확산 | Diffuser [49], KALM [16] | 출력을 SE(3) 링크 포즈에서 "접촉 위치 + 법선" T*로 교체 |
| 손 메시와 추적 | MANO [44], HaPTIC [45], SAM3 [43] | 단안 깊이 오차를 다중 뷰 광선 삼각측량으로 융합(Hartley-Sturm [46]), IoU 가중 |
| 표면 접촉 IK | cuRobo [52]의 배치 IK | 식 4(위치 + 법선 hinge), 롤 여유 활용 |
| 도달 가능 앵커 집합 | 최원점 샘플링 [51] | 가림, 매몰, 볼록 껍질 안쪽 앵커 제거 |
| 경로 비용 | Yoshikawa 조작성 [53] | 식 5의 단계 비용 안 항 |
| 빔 탐색 | 일반적인 빔 탐색 | 링크·앵커·관절의 이산-연속 혼합 탐색(폭 20, 후보 32) |
| 자유 공간 접근 | cuRobo 경로 최적화, MoveIt RRTConnect [54] | 제약 하에서 호출 |
| 제약 프리미티브 | 일반적인 원뿔·영역 제약 | AxisCone(각도 L1), PositionRegion(pseudo-Huber), 소프트 + 하드 이중 사용 |
| 제약 생성 | VLM 프롬프팅 일반(GPT-5.5) | 두 번 호출(장면 분해와 제약 인스턴스화), 결정적 사전 점검 |
| 시연 수집 | GELLO [9](기준선 원격조종) | 맨손 다중 카메라 대 GELLO |

**새로움의 위치.** 가장 새로운 부분은 "접촉 링크와 앵커를 이산 변수로 두고 샘플-탐색으로 함께 푸는 것"과 "로봇 비의존 표현을 최적화기 입력으로 고정하는 분해"입니다. 확산 정책, 키포인트, 손 추적은 기성 부품이라, 이 논문의 학습 기여는 작습니다(`analysis/03_analysis.md` §2의 재생 대조).

## 4. GitHub와 코드 자료 (확인 2026-11-23)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드** | 없음 | - | - | 프로젝트 페이지 링크 목록, WACL-robot 조직 저장소(`.github`, 페이지 저장소 두 개), arXiv 초록·코멘트, GitHub 검색에서 찾지 못함 |
| 프로젝트 페이지 | https://wacl-robot.github.io | 없음 | 저장소 WACL-robot/WACL-robot.github.io, 별 0, 커밋 22, 최신 푸시 2026-11-16 | 논문 PDF, 포스터 PDF(27 MB), 영상, BibTeX. 코드 링크 없음 |
| 비공식 후속 환경 | https://github.com/Jojohickman21/wam-rl-diagnostic | 없음 | 별 0, 2026-11-21 | 논문의 CloseCabinet을 MuJoCo xArm 7로 옮겨 PPO를 돌린 진단 환경(제3자, 저자 비관련). 확인하지 않았고 실행하지 않음. 논문 방법의 구현이 아님 |
| KALM(정책·키포인트 기반) | https://github.com/FANG-Xiaolin/KALM | MIT | 별 32, 최신 `689b96f`(2025-09-02) | WACL이 "아키텍처 변경 없이" 쓰는 정책의 공개 구현. 저장소를 받지 않았고 정적 비교는 하지 않음 |
| cuRobo | https://github.com/NVlabs/curobo | Apache-2.0 | 별 2,037, 최신 `163e335`(2026-11-21) | WACL 최적화기의 기반(CUDA 전제) |
| SAM 3 | https://github.com/facebookresearch/sam3 | 별도 SAM 라이선스(GitHub는 NOASSERTION) | 별 12,374, 2026-11-17 | 손·객체 분할. 가중치 게이트 확인하지 않음 |
| DINOv2 | https://github.com/facebookresearch/dinov2 | Apache-2.0 | 별 13,561, 2026-11-05 | 키포인트 특징 |
| MoveIt 2 | https://github.com/moveit/moveit2 | BSD-3-Clause | 별 2,130, 2026-11-19 | 접근 계획 대체 경로(논문은 MoveIt, 버전 미기재) |
| GELLO | https://github.com/wuphilipp/gello_software | MIT | 별 566, 2026-11-10 | 기준선 시연 수집 |
| xArm Python SDK | https://github.com/xArm-Developer/xArm-Python-SDK | BSD-3-Clause | 별 333, 2026-10-21 | 로봇 구동 |
| HaPTIC | 저장소 찾지 못함 | 미확인 | - | 4D 손 추적기(Ye 외, 3DV 2026 [45]) |
| GPT-5.5 | 호스티드 API(코드 없음) | - | - | 스냅샷 식별자 미기재(B.2: "one GPT-5.5 snapshot") |

공개 코드가 없으므로 `repo/`는 만들지 않았고, 코드 대조 대신 `analysis/04_claim_verification.md`에서 주장 C1–C20을 재계산으로 검증했습니다.

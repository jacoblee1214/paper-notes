# 02. 선행 연구와 공식 저장소: ProxiDex (arXiv:2609.16586)

참고문헌 번호는 논문의 [n]입니다. 확인하지 못한 항목은 "미확인"으로 적었습니다.

## 1. 계보와 구성 요소별 출처

ProxiDex는 새 부품을 하나 만들었다기보다 **이미 있는 부품 여섯 개를 근접도라는 한 신호 주위로 묶은 구성**입니다. 새 부분은 (a) 지역 근접도 벡터를 정책 조건 신호이자 동역학 정렬 목표로 쓰는 것, (b) 근접도를 IDM으로 추정하는 것, (c) FDM 예측과 실제 관측의 잠재 거리로 dream 모드를 전환하는 것입니다.

| 구성 요소 | 이 논문의 사용 | 출처 | 근접한 차이 |
|---|---|---|---|
| 상호작용 점군 | 복원 물체 메시와 손 링크 표본의 합집합(§4.1) | CordViP [1] (복원 기반 상호작용 점군 + 팔 협응), D(R,O) Grasp [39] (로봇-물체 상호작용 표현) | CordViP는 점군에서 대응 관계를 쓰고 이 논문은 거리를 영역 요약해 별도 토큰으로 둠 |
| 물체 복원·자세 | Grounded-SAM [36] → SAM 3D [37] → 깊이로 배율 보정 → FoundationPose++ [38] | 각 도구 | 메시 보정(식 9, 10)이 부록에만 있음 |
| 확산 정책 | Transformer 확산 정책, 비대칭 인과 인코더(§4.3) | Diffusion Policy [10], DP3 [11], ManiFlow [26], Flow Policy [27] | 근접도 토큰열에 인과 마스크, 전역 관측 토큰은 전체 가시 |
| 잠재 동역학 보조 | FDM(DiT 잡음 제거) + 행동 조건(식 1, 2, 18) | 동역학 사전학습 AFRO [28], LDA-1B [29], DiT [40] | 사전학습이 아니라 정책과 같은 데이터로 학습, 근접도 정렬(IDM)이 추가 |
| 세계 모델 계열 | 비교·동기(§2 "Dynamics-aware representation learning") | OmniVTA [30], Visuo-Tactile World Models [31], VTAM [32], 동역학 인지 정책 [33], 입자 세계 모델 [34, 35] | 촉각·이미지가 아니라 기하 근접도를 예측 대상으로 함 |
| 원격조작 | VR 손-팔 공동 조작, 키포인트 재타깃(식 6–8) | Hand Tracking Streamer [14], DexPilot [5], AnyTeleop [19], Open-TeleVision [20], Bunny-VisionPro [21], OpenTeach [6] | 점군 근접도를 헤드셋으로 되돌리는 시각 피드백 |
| 벤치마크 | Adroit [41], DexArt [42], Isaac Lab [43] | 각 원 논문 | Isaac Lab 과제 세 개는 자체 제작 |
| 같은 그룹의 선행 | FAR-Dex [2] (arXiv:2603.10451, 같은 제1저자) | [2] | 이 분석에서 읽지 않음(미확인) |

### 가까운 대안과의 관계

- **CordViP [1]** 가 가장 가까운 기준선입니다. 둘 다 복원 Int.PC를 쓰므로 표 1의 CordViP 대비 격차(+2.9)와 표 2의 격차(+9.7)가 "근접도 + 동역학"의 순효과에 가장 가깝습니다. 이 논문은 §5.1에서 이를 직접 지적합니다.
- **AFRO [28]** 는 같은 장면 점군 입력에서 동역학 사전학습만 더한 기준선이라 "동역학 사전학습 효과"의 대조입니다.
- **촉각 정책 계열**: 3D-ViTac [7], Contact-Grounded Policy [4], 적응형 시각-촉각 융합 [12, 13]. ProxiDex는 이들을 "하드웨어 의존·보정 부담"으로 대비하지만, 접촉력을 재지 못한다는 한계는 스스로 적습니다(§7). 촉각과의 직접 비교는 그림 9 하나입니다.

## 2. 공식 저장소와 코드 현황

| 저장소 | 관계 | 라이선스 | 스타 | 마지막 푸시 | 상태 |
|---|---|---|---|---|---|
| (공식 구현) | 없음. 프로젝트 페이지 Code 버튼이 "Code: Coming Soon" | - | - | - | 2026-11-24 기준 미공개 |
| [ProxiDex/ProxiDex.github.io](https://github.com/ProxiDex/ProxiDex.github.io) | 프로젝트 페이지 소스(학술 페이지 템플릿 기반, README는 템플릿 그대로) | 없음 | 0 | 2026-09-16 | 코드 없음. 복제하지 않음 |
| [proxidex.github.io](https://proxidex.github.io/) | 프로젝트 페이지 | - | - | - | 페이지에 실린 표 6이 논문 부록의 표 6과 같은 값 |
| [wengmister/hand-tracking-streamer](https://github.com/wengmister/hand-tracking-streamer) | 원격조작 기반 [14] (논문 참고문헌의 URL이 `githu.com`으로 오타) | Apache-2.0 | 269 | 2026-06-07 | 참조만 |
| [teal024/FoundationPose-plus-plus](https://github.com/teal024/FoundationPose-plus-plus) | 자세 추적 [38] | 없음 | 390 | 2025-05-15 | 참조만 |
| [xuanxuanzzzii/cordvip](https://github.com/xuanxuanzzzii/cordvip) | CordViP 공식 구현 [1] (README에 "Official Implementation", RSS 2025) | MIT | 56 | 2025-11-26 | 참조만. ProxiDex가 어느 설정으로 돌렸는지는 미공개 |
| [YanjieZe/3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) | DP3 [11] | MIT | 1,475 | 2025-10-17 | 참조만 |
| [geyan21/ManiFlow_Policy](https://github.com/geyan21/ManiFlow_Policy) | ManiFlow [26] | MIT | 222 | 2025-11-13 | 참조만 |
| [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) | Grounded-SAM [36] | Apache-2.0 | 17,795 | 2024-09-05 | 참조만 |
| [facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects) | SAM 3D [37] | NOASSERTION (API, 별도 SAM 라이선스) | 7,719 | 2026-06-02 | 참조만 |
| [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | 시뮬 환경 [43] | NOASSERTION (API) | 8,709 | 2026-11-23 | 참조만 |
| [Kami-code/dexart-release](https://github.com/Kami-code/dexart-release) | DexArt [42] | Apache-2.0 | 154 | 2026-05-11 | 참조만 |
| [aravindr93/hand_dapg](https://github.com/aravindr93/hand_dapg) | Adroit [41] | Apache-2.0 | 330 | 2023-04-09 | 참조만 |

- AFRO [28](arXiv:2512.00074)과 FAR-Dex [2](arXiv:2603.10451)의 저장소는 GitHub 검색에서 찾지 못했습니다(미확인).
- 공식 코드가 없으므로 clone한 것이 없고 커밋 해시도 없습니다. 코드 대조 대신 `04`에서 논문 주장을 C1–Cn으로 나눠 검증합니다.

## 4. 요약

- 새 부분은 근접도 벡터를 정책 조건과 동역학 정렬 목표로 함께 쓰는 설계와 dream 모드 전환이고, 나머지는 기존 부품입니다.
- 가장 가까운 기준선 CordViP도 같은 Int.PC를 쓰므로 순효과는 표 1에서 +2.9점, 표 2에서 +9.7점입니다(`03` V1, V4).
- 공식 코드는 없고, 기준선 중 CordViP, DP3, ManiFlow만 공개 저장소를 확인했습니다.

# 02. 선행연구와 자료: BoxWAM (arXiv:2610.17526)

## 1. 계보: 이 논문이 서 있는 자리

BoxWAM은 새 구성 요소를 만든 논문이라기보다 **"WAM의 미래 상태를 무엇으로 둘 것인가"라는 한 축**을 고정된 흐름 정책 위에서 바꿔 본 논문입니다. 따라서 계보는 (a) WAM이라는 틀, (b) 미래 상태 표현의 선택지, (c) 상태-행동 결합 방식, (d) 구현 뼈대로 나눠 읽는 것이 편합니다. 인용 번호는 논문 참고문헌입니다.

### 1.1 WAM이라는 틀 (1절, 2절)

- 행동 전용 정책(BC): ACT [6], Diffusion Policy [3], DP3 [5], π0 [1], π0.5 [2], OpenVLA [4, 11], RDT [10] 등. 논문은 이들을 "행동만 예측하는 정책"으로 묶습니다.
- WAM: Motus [13], Cosmos Policy [14], "World action models are zero-shot policies" [15], Fast-WAM [16], 인과 월드 모델링(LingBot-VA) [17]. 공통점은 행동과 미래 상태를 함께 생성하는 것입니다.
- 행동 조건 월드 모델과 계획: Finn & Levine [36], Ebert 외 [37], UniSim [38], DINO-WM [39], Cosmos [40, 41]. 후보 행동을 평가하는 방식이라 테스트 시점 비용이 크다는 점이 BoxWAM과의 대비로 쓰입니다(2절).

### 1.2 미래 상태 표현의 선택지 (1절, 2절, 표 4)

| 표현 | 대표 | 논문의 평가 |
|---|---|---|
| 미래 영상 프레임 | Gen2Act [18], Video generators are robot policies [19], UVA [20], Genie Envisioner [21], Large Video Planner [22], Wang 외 [23] | 표현력은 높으나 배경, 조명까지 예측해야 해 비쌈 |
| 의미 마스크 | Mask World Model [24] (MaskWAM 계열) | 언급만 하고 표 4에서 비교하지 않음 |
| 잠재/패치 토큰 | V-JEPA 2 [25], Terver 외 [26], LeWorldModel [27] | 학습이 어렵고 해석이 안 됨 (표 4는 DINOv3 ViT-L 토큰으로 대표) |
| 키포인트, 자세, 상자 | ReKep [57], MimicFunc [59], Where2Act [58] 등 | 모듈형 파이프라인(MPC, IK, 조건 입력)에 주로 쓰였고 WAM 안의 학습된 미래 상태로는 아님 (2절) |
| 운전 분야의 객체 궤적 | Zeng 외 [42, 43] 등 | 동기 부여로만 인용 |

표 4는 위 가운데 VAE 토큰(Wan2.2 [62]), DINO 토큰, 6D 자세, 2D OBB, 기능 키포인트(MimicFunc [59]), EE 궤적을 같은 학습 틀에서 바꿔 끼웁니다. **어느 계열도 "WAM 안의 학습된 미래 상태" 자리에서 이미 비교된 적은 없다**는 것이 논문의 위치 주장입니다. 이 문장은 논문의 서술이고, 우리가 선행 연구를 전수 조사해 확인한 것은 아닙니다.

### 1.3 상태-행동 결합 (2절, 표 5)

상태 → 행동 [14], 공동 예측 [13, 15], 보조·대리 상태 예측 [16]. BoxWAM은 "행동은 미래 상태를 보되 상태는 미래 행동을 보지 않는" 한 방향 어텐션을 고르고 네 가지 대안(분리, 결합, 행동 → 상태)과 비교합니다.

### 1.4 구현 뼈대 (3.2절, 공식 코드)

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| 정류 흐름 목적식, Beta(1.5, 1) 시각 표집 | π0.5 [2] | 3.2절에 명시 |
| 동결 DINOv3 인코더 | DINOv3 [60] | 3.2절. 코드는 ViT-L/16 |
| 트랜스포머 블록, 회전 위치 임베딩 | V-JEPA 2 (MIT) | 논문에는 없고 공식 코드의 THIRD_PARTY_NOTICES.md에 적힘 |
| π0.5 행동 전문가에 상자 토큰 추가 | π0.5 / openpi 포크 | 3.2절 |
| 시뮬레이션, 벤치마크 | RoboTwin 2.0 [28] | 4.1절 |
| 정답 상자의 실세계 재구성 | SAM 3, FoundationPose | 부록 A. 코드에는 없음(`04` D11) |
| IK 기준선 | Mink [63], M2T2 [64] | 4.3절 |

## 2. 가까운 대안과 차이

- **MaskWAM [24]:** 객체 마스크를 입력과 출력에 쓰는 WAM입니다. BoxWAM과 둘 다 "객체 중심 압축 상태"를 주장하지만, 마스크는 픽셀 영역이고 상자는 16개 수입니다. 표 4에 마스크 변형은 없습니다.
- **Fast-WAM [16]:** 테스트 시점 상상을 없앤 WAM. 그림 6에서 Fast-WAM은 BoxWAM보다 성공률이 높고(약 92%) 지연은 더 깁니다(그림 판독 값 약 83 ms 대 65 ms).
- **"행동만" 변형:** 같은 아키텍처에서 상자 토큰과 손실을 뺀 것입니다. 표 1, 3, 4, 5의 핵심 대조군이고, 공식 코드에서는 `model.box_dim=0`으로 만들 수 있습니다(`04` D8).
- **보조 과제로서의 상자 예측:** 표 5의 "완전 분리"는 상자를 보조 출력으로만 둔 것과 같고, 이 변형이 BoxWAM에 가장 가깝습니다(`03` V6). 저자도 "분리 예측이 경쟁적이다"라고 적습니다(4.3절).

## 4. GitHub와 자료 (확인 2026-11-24)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드** | https://github.com/quincy-u/box-wam | MIT (LICENSE 파일). GitHub API는 NOASSERTION | 별 12, 포크 0 / 2026-10-15 푸시, 커밋 2개, 최신 `7afa9c6` | `repo/` 얕은 clone(정적 읽기). 단일 과제 표준 구현 + openpi 포크 + RoboTwin 포크 포함, 추적 파일 1,776개(약 30 MB) |
| 프로젝트 페이지 | https://box-wam.github.io | - | 페이지 저장소 box-wam/box-wam.github.io, 별 0 / 2026-11-10 | 데모 영상, 방법 요약, 표 1·2·3·4와 그림 6의 값. 데모 탭에 Latte, Toast 영상이 있으나 논문 표 3은 커피, 물주기, 포켓몬 세 과제 |
| 단일 과제 체크포인트 | https://huggingface.co/quincyu/box-wam-standalone | 카드에 명시 없음 | 약 4.70 GB (RoboTwin 아홉 과제 .pt) | 받지 않음 |
| π0.5 + Box-WAM 체크포인트 | https://huggingface.co/quincyu/pi05_robotwin_obb_eef | 카드에 명시 없음 | 약 12.4 GB (스텝 34999) | 받지 않음 |
| 학습 데이터 | https://huggingface.co/datasets/quincyu/robotwin_eef_obb_27500 | MIT (데이터 카드) | 약 119.9 GB | 받지 않음. LeRobot v2.1, 27,500 에피소드 |
| RoboTwin 2.0 | https://github.com/RoboTwin-Platform/RoboTwin | MIT | 별 3,080 / 2026-09-24 | 공식 코드는 이 저장소의 포크(`eval/RoboTwin-boxwam`)를 씀 |
| openpi | https://github.com/Physical-Intelligence/openpi | Apache-2.0 | 별 14,745 / 2026-08-24 | 포크 기준 커밋 `650c5b0` |
| DINOv3 | https://github.com/facebookresearch/dinov3 | DINOv3 License (GitHub: NOASSERTION) | 별 11,799 / 2026-11-02 | 가중치는 HF에서 약관 동의 후 받음 |
| V-JEPA 2 | https://github.com/facebookresearch/vjepa2 | MIT | 별 4,912 / 2026-10-29 | 트랜스포머 블록의 출처 |
| Mink | https://github.com/kevinzakka/mink | Apache-2.0 | 별 1,622 / 2026-11-19 | IK 기준선 |

공식 코드가 다루는 범위: 표 1(단일 과제)과 5의 평가 틀, 표 2·그림 6용 π0.5 + Box-WAM 포크, 실로봇 학습 설정 세 개. **다루지 않는 범위:** 표 4의 VAE·DINO·6D 자세·2D OBB·키포인트·EE 변형, 실로봇 정답 상자 재구성(SAM 3, FoundationPose), 표 2의 2,500 시연 관절 제어 설정, 기준선(DP, ACT, DP3, π0)과 Diffusion Policy 실로봇 학습(`04` §2).

## 5. 구성 요소별 한눈에

- **새로운 것(논문의 주장):** 미래 상태로 상자 16차원 + 어텐션 한 방향 + 이를 같은 틀에서 다른 표현과 비교한 절제.
- **빌려온 것:** 정류 흐름, π0.5 시각 표집, DINOv3, V-JEPA 2 블록, π0.5 행동 전문가, RoboTwin.
- **논문에 안 적힌 것:** 트랜스포머 크기(코드: 깊이 12, 폭 768, 약 130M), 이미지 해상도(코드: 224), 학습률과 배치(코드: 1e-4, 384), 위치 인코딩(코드: 3D 회전), 정지 구간 제거(코드: `filter_static_chunks.py`). 이 값들은 `04`에서 코드와 대조합니다.

# 02. 선행연구, 코드 자료, 이 저장소의 관련 분석

## 1. 선행연구 계보

아래 중 [번호]는 논문 참고문헌 번호이고, 번호가 없는 항목은 논문이 인용하지 않은 일반 지식입니다(확인한 것만 적었습니다).

| 계보 | 대표 작업 | 이 논문과의 관계 |
|---|---|---|
| 의도 수준 모방 (인지과학) | Tomasello [1], Meltzoff [11], Whiten 외 [42, 44], Gergely 외 [45], Byrne & Russon [48] | L0–L3의 이름과 구조를 빌린 출처. 저자는 "네 단계를 정확히 나열한 기존 분류는 모른다"고 쓰고 모방/에뮬레이션/목표지향 모방/유추 전이에 대응시킵니다 (부록 A.2) |
| 사람 영상 → 로봇 (궤적 리타게팅) | DexMV [2], Robotic Telekinesis [4], Masquerade [5] (손을 로봇 그리퍼로 인페인팅) | 시연 궤적을 옮겨 닫힌 장면에서 재생. 논문이 "궤적 수준"이라 부르는 계열 |
| 광학 흐름·객체 중심 단서 | Tang 외 [10], X-Sim [12] | 마찬가지로 장면이 비슷하다는 전제 |
| 스킬 표현 | XSkill [3], UniSkill [9], RHyME [13], ImMimic [14] | 베이스라인으로 쓴 두 가지(XSkill, UniSkill)의 원출처. RHyME은 몸체 차이를 단계화하고 장면은 고정, 이 논문은 반대 (부록 A.2) |
| 영상 조건 정책 | Vid2Robot [8] | 영상 프롬프트로 제어. 동일 분포 평가 |
| 짝 데이터셋 | MIME [33], BC-Z [34], RH20T [35], H&R [36], DexWild [37], HRDexDB [38] | 표 2의 비교 대상. IG-10K는 환경 정렬, 시뮬, 다중 시점, 손 자세, 언어, 분할을 모두 갖춘다고 주장 |
| 일반화 벤치마크 | COLOSSEUM [6], RoboTwin 2.0 [15], REALM [7] | 학습 장면의 섭동 강건성. 이 논문은 "시연-실행 장면 불일치"로 축이 다름 |
| 언어 조건 벤치마크 | CALVIN [16], LIBERO [17], VLABench [18] | 과제를 언어로 지정. 이 논문은 언어 지정을 없애고 영상이 과제를 정의 |
| 대규모 로봇 데이터 | Open X-Embodiment [19], DROID [20], BridgeData V2 [21], FMB [22] | 다양성 축. 사람-로봇 짝이 없음 |
| 로봇 정책 | ACT [23], Diffusion Policy [24], VQ-BeT [25], OpenVLA [26, 27], π0.5 [28, 29], GR00T [30], RDT-1B [31], H-RDT [32] | 베이스라인 구성 요소 |
| 일반 지식 (논문 미인용) | One-Shot Imitation Learning (Duan 외, NeurIPS 2017), Domain-Adaptive Meta-Learning (Yu 외, RSS 2018) | 영상·시연을 조건으로 새 과제를 푸는 문제 설정의 원조. 이 논문은 장면 불일치를 단계화해 평가한다는 점이 다름 |

### 구성 요소별 출처

- **레벨 정의 S ⇒ G ⇒ A:** 저자의 운용 가정(부록 A.1). 일반 존재론이 아니라 인스턴스 구성 규칙입니다.
- **시뮬:** ManiSkill3 + SAPIEN, 자산은 YCB, PartNet-Mobility, RoboTwin, Sketchfab (부록 B.1, 저장소 README).
- **데이터 형식:** LeRobot 0.5.0.
- **손 자세·마스크:** WiLoR(MANO), Grounded-SAM-2 (부록 D, E).
- **언어 주석:** 사람 작성 + Qwen3-VL 증강 (부록 B.1, B.4).
- **Arena 판정 방식:** 쌍별 블라인드 선호(아레나형 평가). 논문은 인용하지 않았습니다.

### 가까운 대안과 차이

- **궤적 수준 사람 영상 학습**(DexMV, Masquerade 등)은 L0–L1에 해당하는 능력을 다룹니다. 이 논문의 L2–L3는 다른 방법군(VLM 기반 의도 추론, 어포던스 추론)이 풀어야 한다고 저자가 결론에서 제안합니다.
- **RHyME**은 몸체 불일치 축을 따로 평가합니다. 두 축은 직교하며 결합은 향후 과제로 남습니다 (부록 A.2).
- **프로젝트 페이지 리더보드**는 In-context learning 계열 Zero-WAM(Robbyant)을 추가했습니다. 논문 본문에는 없습니다 (§2 표 참조).

## 2. GitHub와 코드 자료 (확인 2026-11-23)

| 저장소 | 관계 | 라이선스 | 별 / 마지막 push | 대조 기준 | 상태 |
|---|---|---|---|---|---|
| [imitator-game/The-Imitator-Game](https://github.com/imitator-game/The-Imitator-Game) | **공식 코드** (시뮬 프레임워크, 데이터 도구, 베이스라인 12종 폴더) | MIT (저작권 Xunzhe Zhou). 안에 포함된 서드파티 코드(openpi, Gemma, HaMeR, WiLoR 등)는 각자 라이선스 | 68 / 2026-11-09 (포크 3) | `3193957e3b0187d086e44d79e0e4af9845fbd4fd` (2026-11-09, 얕은 복제, 체크아웃 약 552 MB) | 정적 읽기만 (`repo/`, 실행·설치 안 함) |
| [imitator-game.github.io](https://github.com/imitator-game/imitator-game.github.io) | 프로젝트 페이지 소스 (리더보드 수치 JS 포함) | MIT | 0 / 2026-10-27 | 해시 기록 안 함 | 페이지의 공개 정적 파일(JS 3개, CSV, JSON)만 텍스트로 읽음 |
| HF `imitator-game/IG-10K-Dataset` | 데이터셋 (LeRobot 0.5.0) | 카드에 라이선스 필드 없음 | 다운로드 13,215 / 2026-10-16 | - | **받지 않음.** API 기준 저장 용량 약 880 GB(LFS 포함 표기), 파일 12,638개 |
| HF `imitator-game/IG-10K-Assets` | 시뮬 자산 (tar.zst 4개) | 카드 없음 | 564 / 2026-08-17 | - | 받지 않음. 약 20 GB |
| HF `imitator-game/IG-10K-Checkpoints` | 체크포인트 | 카드 없음 | 0 / 2026-10-20 | - | 받지 않음. 약 3.5 GB. API 파일 목록은 `uniskill/` 아래 58개뿐이었음 (README는 "논문에서 평가한 베이스라인 체크포인트"라고 표기) |
| [mani-skill/ManiSkill](https://github.com/haosulab/ManiSkill) | 시뮬 기반 (저장소가 포크해 내장) | Apache-2.0 | 3,535 / 2026-08-04 | - | 참고 |
| [KimHanjung/UniSkill](https://github.com/KimHanjung/UniSkill) | UniSkill 원 구현 | MIT | 92 | - | 참고 |
| [real-stanford/xskill](https://github.com/real-stanford/xskill) | XSkill 원 구현 (저장소에 사본 + LICENSE) | MIT | 74 / 2024-03-25 | - | 참고 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0.5 (저장소에 `pi/`로 내장) | Apache-2.0 (+ Gemma 약관 파일) | 14,732 / 2026-08-24 | - | 참고 |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | GR00T N1.6 | Apache-2.0 | 8,403 / 2026-11-17 | - | 참고 |
| [thu-ml/RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer) | RDT-1B | MIT | 1,821 / 2026-01-21 | - | 참고 |
| [moojink/openvla-oft](https://github.com/moojink/openvla-oft) | OpenVLA-OFT | MIT | 1,450 / 2025-09-09 | - | 참고 |
| [tonyzhaozh/act](https://github.com/tonyzhaozh/act), [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy), [jayLEE0301/vq_bet_official](https://github.com/jayLEE0301/vq_bet_official) | ACT, DP, VQ-BeT 원 구현 | MIT | 2,352 / 4,738 / 222 | - | 참고 |
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | 데이터 형식 | Apache-2.0 | 29,024 / 2026-11-22 | - | 참고. README는 알려진 lerobot 버그를 특정 커밋의 파일로 덮어쓰라고 안내 |

- 공식 저장소의 README 뉴스: 2026-08-23 논문·코드 공개, 09-18 리더보드 과제별 결과, 10-10 CoRL 채택과 NeurIPS ODI 워크숍 구두, 10-21 체크포인트, 10-28 Zero-WAM 추가. 마지막 커밋은 2026-11-07의 Astra 기준선·200 에피소드 평가 결과 내보내기를 병합한 PR #2입니다.
- 프로젝트 페이지는 "Nine models, sixteen variants"라고 쓰며 논문(15개 변형)에는 없는 Zero-WAM이 추가된 수입니다.
- 코드·데이터·체크포인트는 설치하거나 받지 않았습니다.


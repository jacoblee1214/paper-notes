# 02. 선행 연구와 공식 저장소

## 0. 방법

- 논문 참고문헌 54개(v2)와 서론, 관련 연구(§2)를 읽고 계보를 정리했습니다.
- 공식 저장소는 GitHub API(2026-11-21 조회)로 찾았습니다. 검색어는 "RealDexUMI", "2606.06033", "dexumi", 저자 조직 BeingBeyond의 공개 저장소 29개 설명 문자열입니다.
- 프로젝트 페이지 `research.beingbeyond.com/realdexumi`의 링크를 확인했습니다.

## 1. 계보: UMI에서 손 모듈까지

1. **UMI** (Chi et al., RSS 2024, 참고 [17]): 로봇 없이 그리퍼로 시연하고, 말단 기준 관찰과 행동을 씁니다. 이 논문의 출발점입니다.
2. **UMI 확장 계열** (§2.1, 참고 [22–35]): FastUMI [24, 25], 촉각을 붙인 3D-ViTac [26], Touch in the Wild [27], exUMI [28], TacUMI [29], OmniUMI [30], ActiveUMI [32] 등. 대부분 그리퍼 기반입니다. 논문은 그리퍼의 제한된 구동과 단순한 접촉 기하가 손재주 작업에 부족하다고 씁니다.
3. **손 시연 인터페이스** (§2.2, Table 1): 모션캡처 장갑(DexViTac [1], DexWild [8], Bunny-VisionPro [39], DexCap [40], ARCap [41], Glovity [42])은 리타게팅이 필요합니다. 링키지 외골격(DEXOP [18], Dexexo [19], 그 밖의 [43–49], MILE [48], ExoStart [49] 포함)은 측정 상태를 주고 명령을 주지 않습니다. 팔에 붙이는 방식(DEX-Mouse [20], Exo-ViHa [21])은 큰 구조물 때문에 작업 공간과 손목 움직임을 줄입니다(논문 §2.2). **DexUMI** [2]는 사람 손에 착용하는 외골격과 시각 인페인팅으로 손 상태를 로봇 손 형태로 바꿉니다.
4. **RealDexUMI** (이 논문): 사람 손에 끼우지 않고 **로봇이 배포될 때 쓰는 손 모듈을 수집기로 씁니다.** 장갑은 손바닥에 고정된 등척 장갑으로 명령을 직접 냅니다. 관측과 행동이 배포 때와 같습니다.

Table 1의 비교표(논문 값)는 RealDexUMI가 리타게팅 없음, 자유 시야(시각 정렬), 촉각, 행동-상태 대응을 모두 갖춘 유일한 항목이라고 주장합니다. 이 표의 무게와 자유도 값은 인용값이며 우리가 확인하지 않았습니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 (논문 참고문헌 번호) | 비고 |
|---|---|---|
| 정책 백본 ACT | Zhao et al., ICML 워크숍 2023 [51] | 주 백본. 청크 예측과 CVAE 잠재 변수 |
| 보조 백본 Diffusion Policy | Chi et al., IJRR 2024 [52] | 같은 관측과 행동 인터페이스 |
| 시각 인코더 | ImageNet 사전학습 ResNet-18 (Table 6) | DP는 사전학습 없음 (Table 6 "None") |
| 교차 로봇 IK | 로봇별 제어기 (논문에 명시 없음) | FR3 7 DoF, RM65 6 DoF, Adam-U 7 DoF |
| 텔레옵 기준선 | AVP [53] (Park & Agrawal), Manus 리타게팅 [54] (Xin et al., 2026) | 부록 F에 설정 |
| 상대 엔드이펙터 행동 | FastUMI [25], TouchGuide [50]의 좌표 정렬과 대비 | 논문은 이 방식들을 "고정 작업 공간을 가정한다"며 피합니다 |
| 저자 선행 연구 | Being-H0 [6, 7] (Luo et al., 2025–2026), UniTacHand [13] (Zhang et al., 2025) | 저자 여러 명이 겹치는 자기 인용. v2에서 추가됨 |

## 4. 공식 코드와 저장소

| 저장소 | 관계 | 라이선스 (GitHub API) | 마지막 push | 스타 | 상태 |
|---|---|---|---|---|---|
| `github.com/realdexumi/realdexumi.github.io` | 이름이 같은 저장소. 내용은 익명 제출 보충 영상 페이지 (`index.html` 593 B, 영상 18.9 MB, 커밋 fe8ef7a, 2026-06-19) | 없음 | 2026-06-19 | 0 | **코드 아님.** 클론하지 않음 |
| `github.com/BeingBeyond` (공개 저장소 29개) | 저자 소속 단체. 이 논문 이름의 저장소 없음 (설명 문자열에 umi, dexumi, glove, isomorph 일치 없음) | — | — | — | 공식 코드 후보 아님 |
| `research.beingbeyond.com/realdexumi` | 프로젝트 페이지. "Accepted: CoRL 2026". GitHub 링크는 조직 전체 페이지만 가리킴. 코드, 데이터, 모델 링크 없음 | — | — | — | 참고 |
| `real-stanford/DexUMI` | 선행 연구 DexUMI의 공식 코드 (임베디드, 링크 최적화, mocap 폴더 포함) | MIT | 2026-04-18 | 280 | 선행 코드 (RealDexUMI 아님) |
| `real-stanford/universal_manipulation_interface` | UMI 공식 코드 | MIT | 2026-05-26 | 1,647 | 선행 코드 |
| `tonyzhaozh/act` | ACT 원저 코드 | MIT | 2024-07-23 | 2,352 | 정책 백본 (재현 시 쓸 수 있는 후보) |
| `real-stanford/diffusion_policy` | Diffusion Policy 공식 코드 | MIT | 2024-12-24 | 4,737 | 보조 백본 후보 |
| `Improbable-AI/VisionProTeleop` | AVP 텔레옵 라이브러리 | MIT | 2025-12-28 | 838 | Fig. 8 기준선 |
| `BeingBeyond/Being-H0` | 저자 선행 VLA (ICML 2026) | MIT | 2026-05-04 | 61 | 참고 |

- **RealDexUMI 공식 코드는 공개되어 있지 않습니다.** 따라서 `04_claim_verification.md`로 주장을 검증했습니다(코드 대조 없음).
- 데이터셋과 하드웨어 설계는 공개되지 않았습니다. 논문은 데이터를 "공개할 예정"이라고만 씁니다.

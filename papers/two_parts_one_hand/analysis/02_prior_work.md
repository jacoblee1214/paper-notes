# 02. 선행연구와 관련 자료: Assembling Two Parts in One Hand (arXiv:2609.10137)

참고문헌 번호는 논문(v1)의 번호입니다. 논문이 직접 서술한 계보와, 구성 요소별 출처를 구분했습니다.

## 1. 계보

### 1.1 다지 손으로 하는 일: 파지와 손 안 조작

- **파지:** 대규모 데이터셋 합성(DexGraspNet[3], GraspXL[12], Dex1B[13])과 새 물체 파지 정책(DexPoint[4], RobustDexGrasp[5], DextrAH-G[14], FunGrasp[15], ClutterDexGrasp[16], DexGraspVLA[17]). 논문은 이런 파지가 손을 한 물체를 감싸는 도구로 줄이며 평행 그리퍼로도 가능하다고 봅니다[18].
- **한 손으로 여러 물체:** 손가락을 부분집합으로 나눠 각 물체를 따로 감싸는 순차 다물체 파지[19–21]. 논문은 이를 "독립된 그리퍼들이 함께 들고 있는 것"으로 구분합니다.
- **손 안 조작(sim-to-real RL):** Rubik's cube[22], 손 안 조작[23], 재방향[7], 회전[8, 9, 24–26], 이동[27]. 쉽게 정식화되는 단일 물체 기술이 중심입니다. DexNDM[9]은 "손 안 회전은 더 다양한 손으로 가능하다"는 근거로 §4.2에 인용됩니다.
- **시연·리타게팅:** AnyTeleop[10], DexForce[11], Open-TeleVision[28], ACE[29], 두 손 시각촉각[30], ManipTrans[31], DexTrack[32]. 논문은 체현 차이가 정교한 기술을 막는다고 봅니다.

### 1.2 조립

- **시뮬레이션 접촉 조립 RL:** Factory[33](PhysX SDF 접촉, 이 논문의 시뮬레이션 설정의 출처), IndustReal[34], AutoMate[35], Fabrica[36]. 거의 모두 평행 그리퍼가 한 부품을 고정하고 다른 부품은 고정구나 두 번째 팔이 잡습니다.
- **손이 달린 조립:** Lee 외[37](양팔 + 로봇 손의 열쇠구멍 peg-in-hole, 손가락은 쥐고 미리 회전만), Lin 외[38](두 손으로 병뚜껑 열기, 분해), Triyonoputro 외[39](특수 이중 턱 기구의 손 안 조립).
- 이 논문의 위치: 범용 의인형 손의 손가락 협응만으로 고정구 없이 조립(plug-in)합니다.

### 1.3 보상과 사람 기준

- 사람 영상에서 뽑은 기준으로 손목 움직임과 파지 자세를 이끄는 계열(ViViDex[42], Lum 외[43]). 이 논문은 이를 **단일 프레임 스냅샷**으로만 씁니다(§3.4). 손가락의 미세 움직임은 체현 차이와 추출 잡음 때문에 다중 프레임 기준이 도움이 안 된다는 주장입니다.
- 이력 입력: 8프레임 입출력 이력 MLP는 Li 외[51]의 방식을 비교군으로 가져온 것입니다(§4.3).
- 형상별 손 안 조작에서 "소수 접촉점 핀치가 시뮬레이션에서 불안정"한 문제는 Ye 외[53]를 인용합니다(§4.5).

### 1.4 도구

FoundationPose[41](물체 자세 추적), Isaac Sim[46], PhysX[52], Isaac Lab[55], rl_games[56], PPO[57], RealSense D435[47], Franka Research 3[45], Sharpa Wave[44], Wuji[48], Allegro[49], XHand[50].

## 2. 구성 요소별 출처와 새로움 (논문 서술 기준)

| 구성 요소 | 논문 위치 | 출처 | 새로움 (논문 주장) |
|---|---|---|---|
| PhysX SDF 접촉 시뮬레이션 설정 | §3, Appx. A | Factory[33] | 새롭지 않음 |
| 손 안 조립 문제 설정과 세 과제 | §1, §3.2 | Triyonoputro[39] 등 | 새 설정(고정구 없음, 범용 의인형 손) |
| 목표 상대 자세 도달 보상 (식 3) | §3.4 | 조립 RL의 상대 자세 보상 계열[33–35] | 형태는 지수형 두 항, 상세 비교 없음 |
| 손가락 기능 보조 보상 ρ_pinch·ρ_grasp | §3.4, 식 4 | 논문 자체 설계("empirically motivated") | 새 구성 요소 |
| 단일 스냅샷 자세 정규화 ρ_pose | §3.4–3.5 | 사람 영상 기준 계열[42, 43] | 단일 프레임으로 축소 |
| 스냅샷 기반 초기화와 지연 교란 정착 | §3.5, Appx. A | AnyTeleop식 리타게팅[10] | 새 구성 요소(검지 30% 감소 포함) |
| 순환 정책(LayerNorm-LSTM) + PPO | §3.3, 표 2 | rl_games[56], Isaac Lab[55] | 새롭지 않음 |
| 관측 잡음 도메인 무작위화 | §3.3, 표 3 | sim-to-real RL 전반 | 새롭지 않음 |
| FoundationPose + 깊이 일관성 게이트 | §3.3, Appx. B | FoundationPose[41] | 게이트는 논문 자체 |

## 4. GitHub와 코드 자료 (확인 2026-11-24)

| 자료 | 링크 | 라이선스 | 별 / 마지막 push | 비고 |
|---|---|---|---|---|
| **공식 코드** | https://github.com/LTBGbird/in-hand-assembly | 없음 (API license: null) | 14 / 2026-09-10 (`b90909d`) | 단일 커밋. README에 "TODO: 시뮬레이션 학습 코드(Bottle, Syringe, Marker), 실로봇 배포 코드, before November 2026". 구현 파일 없음(`size: 0`), 릴리스 없음, 브랜치 `main`만 있음 |
| 프로젝트 페이지 | https://ltbgbird.github.io/in-hand-assembly-page/ | 표기 없음 (저장소 `LTBGbird/in-hand-assembly-page`, license null) | 0 / 2026-09-11 (`5a5c2aa`) | HTML만 열람. Code 버튼 "Code (before November)" |
| Isaac Lab | https://github.com/isaac-sim/IsaacLab | NOASSERTION | 8,709 / 2026-11-23 | 학습 래퍼(논문 [55]) |
| rl_games | https://github.com/Denys88/rl_games | MIT | 1,409 / 2026-11-23 | PPO 구현(논문 [56]) |
| FoundationPose | https://github.com/NVlabs/FoundationPose | NOASSERTION | 3,713 / 2026-04-29 | 실로봇 물체 추적(논문 [41]) |
| Isaac Sim | https://github.com/isaac-sim/IsaacSim | NOASSERTION | 4,528 / 2026-11-16 | 시뮬레이터(논문 [46]) |
| PhysX | https://github.com/NVIDIA-Omniverse/PhysX | BSD-3-Clause | 4,862 / 2026-11-18 | 논문 [52]는 NVIDIAGameWorks 주소를 적음. 현재 저장소는 위 주소 |
| IndustReal | https://github.com/NVlabs/industreallib | NOASSERTION | 85 / 2024-01-05 | 조립 sim-to-real 선행[34] |
| IsaacGymEnvs (Factory) | https://github.com/isaac-sim/IsaacGymEnvs | NOASSERTION | 2,949 / 2024-10-26 (보관됨) | Factory 환경[33]의 원 구현 |

공식 저장소에는 코드가 없어 `repo/`는 만들지 않았고, 어떤 것도 내려받거나 실행하지 않았습니다. 저자 계정에는 프로젝트 페이지 저장소 외에 관련 저장소가 없습니다(저자 계정의 공개 저장소 목록 확인). 2026-11-24 현재 약속한 시점("before November 2026")이 지났습니다.

# 02. 선행 연구와 자료

## 1. 계보

HiFun은 "실로봇 RL + 사람 개입(HIL)" 줄기에 "손 기술을 동결해 행동 공간을 줄이는 계층화"를 붙인 구조입니다. 아래는 논문이 인용한 것과 그 관계입니다(번호는 논문 참고문헌).

### 1.1 실로봇 RL과 사람 개입

- **SERL [17] → HIL-SERL [18] (Science Robotics 2025).** 사전 시연 + 온라인 RL + 실시간 사람 교정을 합쳐 그리퍼 정밀 과제를 거의 100%로 올린 방법입니다. HiFun의 백본(RLPD)과 개입 흐름, 기준선이 모두 이것입니다. 논문은 HIL-SERL의 한계를 "고차원 팔-손 시스템에서 탐색 부담이 크고, 결합된 손가락 관절을 전 자유도로 일관되게 교정하기 어렵다"고 정리합니다(2절).
- **RLPD [40] (Ball et al., 2023).** 오프라인 데이터를 절반씩 섞어 쓰는 샘플 효율 온라인 RL. HiFun 조정 정책의 학습 엔진입니다.
- **The ingredients of real-world robotic RL [43].** 실로봇 RL에서 리셋, 보상, 안전이 병목이라는 정리. 논문은 손가락 교정의 어려움을 여기에 연결합니다.
- **HG-DAgger [41], DAgger [42].** 사람이 개입하고 그 구간을 모방하는 계열. HiFun의 IAW는 이 모방 항에 가치 기반 가중을 곱한 것입니다.

### 1.2 계층화와 기술 재사용

- **Sequential Dexterity [44]**: 여러 손 정책을 체인으로 묶는 장기 조작. HiFun은 체인이 아니라 "얼린 기술 + 활성 강도"를 씁니다.
- **From simple to complex skills [4]**: 학습된 회전 기술을 더 어려운 재배향에 재사용.
- **Simple sub-skill controllers [45]**: 단순한 하위 기술 제어기로 RL 탐색을 안내.
- 이들은 모두 시뮬레이션 학습이고, HiFun은 하위 기술도 실로봇에서 얻는다는 점이 다릅니다.

### 1.3 손 동작의 출처와 텔레옵

- **직접 교시 사전 지식:** KineDex [38]의 hand-over-hand 인터페이스로 θ_ref를 얻습니다.
- **텔레옵 기준선:** DexPilot [14], AnyTeleop [15] (표 3의 DexPilot*, AnyTeleop*, HIL-SERL Glove+Gello), Gello [48] (팔 개입과 기준 시연). 다른 수집 도구로 DexCap [34], DexUMI [35], DexOp [36]와 재타게팅 [37]을 인용하며 "형태 차이에서 접촉 잡음이 생긴다"는 논거로 씁니다.
- **DMP:** Cartesian DMP [47] (Ude et al., 2014). 목표 고정, 시작 자세 무작위화로 오프라인 데이터를 만듭니다.

### 1.4 모방 학습 기준선

BC [52], Diffusion Policy [7]. 논문은 ACT [8], VLA [9–12]도 인용하지만 비교하지 않았습니다.

## 2. 구성 요소별 출처와 변경

| 구성 요소 | 출처 | HiFun에서 바뀐 점 |
|---|---|---|
| 온라인 RL 엔진 | RLPD [40], SAC [46] | 정책 출력을 [Δx, α]로 줄임 |
| 개입 흐름 | HIL-SERL [18] | 손은 버튼으로 얼린 π_skill 호출(전 자유도 교정 대신) |
| 팔 개입 | Gello [48] | 상대 SE(3) 재타게팅으로 매끄러운 인계 |
| 손 기술 | KineDex 교시 [38] + SAC 잔차 | 교시 궤적을 되감는 접촉 회복 |
| 데이터 증강 | Cartesian DMP [47] | 오프라인 버퍼를 DMP 롤아웃으로 대체 |
| 다중 헤드 비평가 | 표준 SAC 이중 Q | Q_nav와 Q_mod 분리, 행동 성분별 정지 기울기 |
| IAW (식 7–8) | 이점 가중 회귀(AWR 계열)와 개입 모방 | 개입과 정책 행동의 가치 차를 가중으로 사용. 논문은 AWR 계열을 인용하지 않음 |
| 보상 | HIL-SERL의 분류기 보상 | 활성 보상 r_mod 추가 |

IAW와 AWR 계열(이점으로 모방 가중)의 관계는 논문에 서술이 없어서 이 표에 "비슷한 계보"로만 적었습니다.

## 4. GitHub와 자료 (확인 2026-11-26)

| 자료 | 링크 | 라이선스 | 별 / 마지막 푸시 | 비고 |
|---|---|---|---|---|
| **공식 코드** | https://github.com/Hly-123/HiFun_code | 확인 불가 | 확인 불가 | 프로젝트 페이지의 "Code" 버튼과 페이지 저장소 README가 가리키는 주소. 2026-11-26 기준 공개된 저장소를 확인하지 못함 |
| 프로젝트 페이지 | https://hly-123.github.io/HiFun/ | 표기 없음 | (저장소 Hly-123/HiFun) 1 / 2026-11-10, 커밋 `97104097ce5459c200ea9486b796fec5d925ee36` | 정적 페이지. 영상, 방법·결과 요약, 인용. 저장소 약 700 MB(영상) |
| 제3자 구현 | 찾지 못함 | - | - | GitHub 검색(HiFun dexterous, 제목) 결과 없음 |
| HIL-SERL | https://github.com/rail-berkeley/hil-serl | Apache-2.0 | 1603 / 2025-10-27, 커밋 `c32939bccb65f3b8c43a9f9add3d322d4ab0264a` | 논문의 기준선 알고리즘과 백본 계열. 이 논문의 설정으로 실행한 기록은 없음 |
| RLPD | https://github.com/ikostrikov/rlpd | MIT | 422 / 2023-02-13 | 조정 정책 학습 엔진의 공개 구현 |
| Gello | https://github.com/wuphilipp/gello_software | MIT | 569 / 2026-11-10 | 팔 개입·기준 시연 장치 소프트웨어 |
| dex-retargeting (AnyTeleop) | https://github.com/dexsuite/dex-retargeting | MIT | 1322 / 2025-08-21 | 텔레옵 기준선 재타게팅 |
| Diffusion Policy | https://github.com/real-stanford/diffusion_policy | MIT | 4747 / 2024-12-24 | DP 기준선 |
| SAC (원 저자) | https://github.com/haarnoja/sac | 파일 판별 불가(NOASSERTION) | 1310 / 2023-11-29 | 손 기술·조정 정책의 SAC |
| KineDex | https://github.com/DinoMini00/KineDex_code | MIT | 28 / 2025-05-02 | 직접 교시 인터페이스 후보. 논문 [38]의 공식 저장소인지 저자 일치는 확인하지 않음 |
| 데이터, 가중치 | 공개 확인 안 됨 | - | - | 시연, 체크포인트, 분류기 데이터 |

별 수와 푸시 날짜, 커밋은 `gh api`로 읽은 값입니다. 라이선스 열의 "확인 불가"는 저장소를 열 수 없어서 읽지 못했다는 뜻입니다. 프로젝트 페이지와 논문의 차이: 페이지는 부록 그림을 "Appendix Fig. 4A"로, 논문 v1은 같은 패널을 그림 8A로 번호 매깁니다. 페이지는 XHand 추가 과제(드릴 작동, 수도꼭지)를 영상으로 보이지만 논문에는 수치가 없습니다.

# 02. 선행연구와 관련 저장소

WT-UMI의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공개 자료가 무엇인지 정리합니다. 대괄호 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| WT-UMI의 요소 | 출처 | 관계 |
|---|---|---|
| 휴대형 시연 인터페이스, 로봇 없이 수집 | UMI [36] | 이름과 발상의 원형. UMI는 손목 카메라와 운동학 위주이고 접촉 힘이 없음(논문 §2) |
| 힘을 더한 UMI 계열 | UMI-FT [26], ForceMimic [25], TacUMI [11] | 손목·손가락 힘 센서로 시연의 힘을 기록. 논문은 이들의 측정이 "국소적"이라고 구분(§2). TacUMI는 HRI 확장 초록 |
| 전신 촉각 피부 | Punyo [10], HEX-o-SKIN [9], 모델 기반 전신 접촉 제어 [12, 13], 대형 물체 계획 [14] | 몸 전체 촉각이라는 센서 쪽 계보. Punyo는 압력 피부 + 예시 유도 RL, HEX-o-SKIN은 셀별 운동학 보정이 필요(§2) |
| 촉각 모방학습(휴머노이드) | TACT [22], Humanoid Touch Dreaming [3] | 같은 문제 영역. [3]은 미래 촉각을 잠재 공간에서 예측해 보조 정규화로 쓰고 접촉은 암묵적(§2) |
| 힘·촉각 인식 VLA와 정책 | TAF-VLA [27], ForceVLA [28], ForceVLA2 [29], FD-VLA [30], 촉각 조건 확산 정책 [7], ACP [4] | 힘·촉각 토큰, 힘 증류, 하이브리드 힘-위치 제어 등. 논문은 이들이 손끝·그리퍼·손목의 국소 접촉과 탁상 과제에 집중한다고 구분(§2) |
| 사람 → 로봇 촉각 격차 | TactAlign [37] | 정렬로 해결하는 상보적 접근. WT-UMI는 공유 하드웨어로 센서 격차를 줄임(§2) |
| 힘 헤드: 위치 질의 교차 어텐션 디코더 | 일반 트랜스포머 디코더(DETR류) | 논문이 출처를 달지 않음. 구조는 학습 가능한 질의 + 자기 어텐션 + h에 대한 교차 어텐션 |
| 행동 생성기 | Diffusion Policy [2], flow matching [42], DDIM [48] | ViT-FMT, ViT-DiT |
| 기반 모델 | π0.5 [45](OpenPI 구현을 FASTER [47]에서 채택), Ψ0 [46] | 촉각 이미지(와 고정 언어 지시)를 입력으로 미세조정 |
| 실시간 청킹 | RTC [49] | 모든 정책에 훈련 시점 방식 적용 |
| 보정·정렬 | DTW (접촉 모드 순서 정렬) | 논문이 출처를 달지 않음 |
| 목표 자세 보정의 발상 | 사람 → 로봇 변환의 오프셋 학습 | 인간 시연의 로봇 실행 가능성 문제는 UMI 계열의 기본 문제. 접촉 게이트와 원격조종 짝으로 오프셋을 배우는 방식이 이 논문의 새 점 |
| admittance 제어 | 고전 어드미턴스·임피던스 제어, ACP [4] | 예측 힘을 기준으로 쓰는 점이 특징 |
| 원격조종 | PICO + XRoboToolkit [38, 39], TWIST2 [35], Mobile ALOHA [33], AnyTeleop [32] | 데이터 수집 쪽 계보 |
| 보행·IK | RL 보행 정책 [43](GR00T N1 논문으로 인용), Pink [44] | 하체는 기존 정책을 그대로 사용 |
| 같은 그룹의 선행 | Refine-DP [21], Opt2Skill [16], Learn-to-Teach [20], 휴머노이드 개관 [1] | 휴머노이드 로코-매니퓰레이션 계보 (저자 겹침) |

**가까운 대안과의 차이.** 가장 가까운 것은 (1) 힘 UMI인 UMI-FT와 ForceMimic(국소 힘, 손목·손가락 센서), (2) 휴머노이드 촉각 정책 Humanoid Touch Dreaming과 TACT(전신 촉각이지만 힘 기준과 제어기 없음), (3) 힘 인식 VLA 계열(탁상용)입니다. 이 논문의 새 점은 **"전신에 걸친 촉각 이미지로 힘을 감독 신호로 쓰는 인간 시연 + 원격조종 짝을 통한 행동 라벨 보정 + 예측 힘을 기준으로 한 admittance"**를 한 하드웨어에서 묶은 것입니다. 논문에는 위 대안과의 **실험 비교가 없습니다**(비교 대상은 같은 파이프라인 안의 네 정책 백본, `04` C10).

## 2. 코드와 저장소 (GitHub 표)

2026-11-24에 `gh api`와 프로젝트 페이지로 확인했습니다.

| 저장소 | 관계 | 라이선스 | 별 / 마지막 push | 대조 기준 | 상태 |
|---|---|---|---|---|---|
| **공식 학습·추론 코드 없음** | 프로젝트 페이지의 Code 버튼이 "Code (Coming Soon)", Dataset 버튼 링크는 `#` | — | — | — | 코드 대조 대신 주장 검증(`04`) |
| [wt-umi/WTUMI](https://github.com/wt-umi/WTUMI) | 프로젝트 페이지(GitHub Pages)의 소스. 파일은 index.html, app.js, styles.css, 영상·그림 자산뿐이고 Python 파일 없음 | LICENSE 없음 (API `null`) | 0 / 2026-10-14 | 마지막 커밋 `612cb90` ("update figure again"), 저장소 크기 약 608 MB(영상) | 복제하지 않음. 페이지 HTML만 읽음 |
| [wt-umi/WTUMI-Hardware](https://github.com/wt-umi/WTUMI-Hardware) | 하드웨어: 가슴판·팔뚝 덮개·GripTac의 STEP CAD, FPC 기판 거버 파일, BOM, 조립 안내(README) | LICENSE 없음 (API `null`) | 0 / 2026-10-12 | 마지막 커밋 `9ebc50d`("Updated project name from LATUMI to WT-UMI"), 크기 약 84 MB | 복제하지 않음. README만 읽음. 데이터 용량 때문 |
| [real-stanford/universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface) | UMI [36] 공식 코드 | MIT | 1,647 / 2026-05-26 | — | 참조만 |
| [real-stanford/UMI-FT](https://github.com/real-stanford/UMI-FT) | UMI-FT [26] 공식 코드 | MIT | 113 / 2026-08-10 | — | 참조만 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0.5 [45] 구현(논문은 FASTER [47]의 OpenPI 기반 구현을 채택) | Apache-2.0 | 14,743 / 2026-08-24 | — | 참조만 |
| [XR-Robotics/XRoboToolkit-Teleop-Sample-Python](https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python) | 원격조종 틀 [39]의 예제로 보임(저장소가 논문의 것인지 소유자 확인 안 함) | MIT | 181 / 2026-01-01 | — | 참조만 |
| [pink-kinematics/pink](https://github.com/pink-kinematics/pink) | Pink IK [44] | Apache-2.0 | 885 / 2026-09-27 | — | 참조만 |
| [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) | 논문은 보행 정책 [43]을 GR00T N1 논문으로 인용했을 뿐 저장소를 지정하지 않음. 관련 후보로만 기록 | NOASSERTION | 3,939 / 2026-11-03 | — | 참조만 |
| [binghao-huang/3D-ViTac_Tactile_Hardware](https://github.com/binghao-huang/3D-ViTac_Tactile_Hardware) | 팔뚝 센서의 원형인 FlexiTac 계열(논문은 [51] 3D-ViTac으로 인용) | LICENSE 없음 | 147 / 2025-07-15 | — | 참조만 |
| [lucidrains/touch-dreaming](https://github.com/lucidrains/touch-dreaming) | Humanoid Touch Dreaming [3]의 제3자 구현(공식 아님) | MIT | 7 / 2026-09-02 | — | 참조만 |
| Ψ0 [46], FASTER [47], TactAlign [37] | 기반 모델·기법 | 미확인 | 미확인 | — | 저장소를 확인하지 못함 (Ψ0는 검색해 보았으나 시뮬레이터만 나옴) |

- **공개 범위 요약:** 하드웨어 설계(CAD, 기판, BOM, 조립 안내)만 공개됩니다. 학습 코드, 체크포인트, 시연 데이터는 공개되어 있지 않습니다(프로젝트 페이지). 하드웨어 저장소와 프로젝트 저장소 모두 LICENSE 파일이 없어 사용 조건이 명시되지 않았습니다.
- 하드웨어 README(조립 안내)의 BOM은 Unitree G1 EDU(견적), SensX 187(950 USD) 1개, SensX 160(369 USD) 2개, FlexiTac 센서와 기판(견적), Arduino Nano 2개, 3D 프린트 PLA·TPU, 실리콘 고무와 PVC 직물 등입니다. 견적이 아닌 품목의 목록 가격을 합치면 약 2,100 USD입니다(나사류 포함 대략).
- 하드웨어 저장소의 이전 이름은 LATUMI("Large Area Tactile UMI")였습니다(마지막 커밋 메시지, README 부제).

## 4. 참고

- 번호가 붙은 참고문헌은 `paper/paper.txt`의 References를 따릅니다.
- 논문 [43]은 GR00T N1 논문(모델 보고서)이고 문장에서는 "pre-trained RL locomotion policy"를 가리킵니다. 어느 체크포인트인지는 논문에 없습니다.

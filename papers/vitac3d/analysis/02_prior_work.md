# 02. 선행연구와 관련 저장소

이 문서는 3D-ViTac의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드가 어디에 있는지를 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| 3D-ViTac의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 정책 학습 (조건부 확산) | Diffusion Policy [13] (Chi 2023), DDPM [64] (Ho 2020) | 행동을 확산으로 생성합니다. 기준선 RGB Only가 [13] 구현입니다 |
| 점군 백본 | PointNet++ [65], Fast Marching FPS [63] | 점군을 계층적으로 줄이고, 시각 점을 FPS로 512개로 줍니다 |
| 촉각 센서 구조 | STAG 장갑 [11] (Sundaram 2019) | 도전성 실과 압저항층 구조를 따릅니다(§3.1) |
| 얇은 촉각 센서 (광학식 대안) | GelSight Svelte [5], GelSlim 3.0 [3], DIGIT [10], OmniTact [9] | 저자는 이들이 두껍거나 강성이 커서 얇은 연한 손가락에 맞지 않는다고 봅니다(§2) |
| 저비용 촉각 피부 | ReSkin [6], All the Feels [7], uSkin [4] | 넓은 면적 감지의 대안. 저자는 해상도가 낮다고 봅니다 |
| 촉각을 정책 입력으로 넣기 (이진 신호) | Robot Synesthesia [27] (Yuan 2023) | 가장 가까운 선행. 저자는 이진 촉각만 쓴다고 비판하고, 3D 통합 표현으로 차별화합니다(§2) |
| 촉각-시각 두 손 정책 | Lin 외 [14] (2024) | 가장 가까운 대안 중 하나. 촉각 해상도가 낮고 시각과 촉각의 공간 관계를 명시하지 않는다고 비판합니다 |
| 촉각 해상도와 회전 | Rotating without seeing [26] (Yin 2023) | 연속 촉각이 접촉 패턴과 힘을 준다는 주장의 근거로 인용합니다(§5.3) |
| 시각 촉각 융합의 선행 (표현 학습) | Lee 외 [53], Dave 외 [54], Suresh 외 (Neural feels) [51] | 시각과 촉각의 다중 모달 표현. 3D 공간 정렬은 다루지 않습니다 |
| 원격조작 플랫폼 | ALOHA 계열, ACT [16] (Zhao 2023) | 마스터-퍼펫 양팔 원격조작의 구성과 저비용 하드웨어 |
| 유연한 그리퍼의 순응성 | UMI [42] (Chi 2024) | 그리퍼 연함이 접촉 순응을 준다는 근거로 인용합니다(§3.1) |
| 촉각 포즈 추정 | 입자 필터 [66] (Djuric 2003) | 부록 A.2.2의 포즈 추적 방법 |
| 점군 기반 정책 (촉각 없음) | DexPoint [28] (Qin 2023) | 점군 강화학습으로 인용됩니다. 촉각을 쓰지 않으며, 이 논문은 점 표현의 선례로만 언급합니다 |

**가까운 대안과의 차이.** 저자가 가장 많이 비교하는 대상은 Robot Synesthesia [27]와 Lin 외 [14]입니다. 두 방식은 촉각을 정책 입력으로 넣지만, 촉각을 이진이나 낮은 해상도로 쓰거나 시각과 촉각을 따로 인코딩합니다. 3D-ViTac은 촉각 단위를 기구학으로 3D에 올려 시각 점과 같은 집합에 넣는 것이 차이입니다. 그러나 논문에는 **이 표현과 "촉각 이미지를 CNN에 넣는 방식"을 같은 조건에서 비교한 실험이 있습니다**(표 1, RGB w/ Tactile Image와 PC. w/ Tactile Image). 이 비교는 점 표현의 효과를 보는 가장 직접적인 자료이며, 결과는 `03` §1.1에 있습니다.

## 2. 코드와 저장소 (GitHub 표)

GitHub API 조회 날짜는 2026-11-21입니다. 저장소 정보는 메타데이터 기준입니다.

| 저장소 | 관계 | 라이선스 | 최근 커밋 (push) | 별 | 상태 |
|---|---|---|---|---|---|
| [binghao-huang/3d-vitac-training-code](https://github.com/binghao-huang/3d-vitac-training-code) | **프로젝트 페이지의 "Method Code" 링크.** README 한 개뿐이며 학습 코드가 없습니다. VT-Refine의 사전학습 부분을 가리킵니다 | 없음 (LICENSE 파일 없음, API null) | c38a5b7 (2025-10-26) | 10 | 얕은 복제 `repo/training/` (정적 읽기). 정책 코드 아님 |
| [binghao-huang/3D-ViTac_Tactile_Hardware](https://github.com/binghao-huang/3D-ViTac_Tactile_Hardware) | 프로젝트 페이지의 "Hardware Codebase". Arduino 펌웨어, Python 시각화, ROS 노드 | **불일치.** readme는 "FlexiTac © 2025 Columbia University, CC BY-NC 4.0", .ino 머리말은 MIT 식 허가 문구, package.xml은 `TODO`, LICENSE 파일 없음 | e151ea6 (2025-07-15) | 146 | 얕은 복제 `repo/hardware/` (정적 읽기) |
| [binghao-huang/3D-ViTac](https://github.com/binghao-huang/3D-ViTac) | 프로젝트 사이트 소스(정적 페이지, 영상, 논문 PDF 사본). 학습과 센서 코드 없음 | 없음 | 4771912 (2026-10-30) | 9 | 복제 안 함 (약 194 MB, 트리만 확인: 79개 항목) |
| [NVlabs/vt-refine](https://github.com/NVlabs/vt-refine) | CoRL 2025 VT-Refine 공식 코드. 학습 저장소가 "3D-ViTac 학습은 이 저장소의 사전학습 부분을 보라"고 가리킵니다 | NVIDIA License (LICENSE 파일). 사전학습 데이터는 CC BY-NC 4.0 | 81beb9e (2025-10-17 커밋) | 134 | 얕은 복제 `repo/vt-refine/` (정적 읽기). 하위 모듈(dppo, easysim)은 받지 않음 |
| [FlexiTac/FlexiTac](https://github.com/FlexiTac/FlexiTac) | 후속 센서 FlexiTac-V2와 펌웨어 저장소(프로젝트 페이지의 "Hardware Repo") | 없음 | 945d623 (2026-11-18) | 190 | 복제 안 함 (3D-ViTac 논문 코드 아님) |

- **결론: 공개된 정책 학습 코드는 없습니다.** 논문의 정책 구조(점군 + 촉각 점 + 확산)를 실행할 수 있는 코드는 저장소에 없습니다. 공개된 것은 센서 펌웨어와 판독 소프트웨어(`repo/hardware/`)입니다. 학습 파이프라인은 다른 논문(VT-Refine)의 사전학습 설정을 가리키는데, 그 설정 파일은 하위 모듈(dppo)에 있어 이번 정적 읽기에서는 볼 수 없었습니다(`04` §2).
- 논문 §6은 "코드와 하드웨어 설정을 공개할 것"이라고 씁니다. 하드웨어는 공개되었지만 정책 코드는 "곧 공개(cleaned 3D-ViTac-only version soon)"라고 적혀 있습니다(`training/readme.md`).
- 프로젝트 페이지의 venue 표기: "Conference on Robot Learning (CoRL) 2024"(`project.html` 상단). 공식 목록(PMLR v270)과 일치합니다.
- 프로젝트 페이지는 센서를 "FlexiTac-V1(3D-ViTac에서 쓴 센서)"과 "FlexiTac-V2(후속)"로 나눕니다. V1의 제작 시간 약 30분, 격자 3 mm, 셀당 $5, V2는 약 3분, 2 mm, $2.5입니다. 논문 본문의 "$20"은 판독 보드를 포함한 값으로 보입니다(`04` C2).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- 확산 정책이 여러 모달 입력의 행동 분포를 잘 학습한다는 결과 [13]. 이 논문은 같은 학습 방식을 점군과 촉각 점에 그대로 씁니다.
- 연속 촉각이 접촉 패턴과 힘을 이진 신호보다 많이 준다는 결과 [26]. 해상도 절제(그림 6)의 근거입니다.
- 연한 그리퍼가 접촉 순응을 준다는 결과 [42]. 취약한 물체(계란, 포도)의 성공률 설명에 쓰입니다.
- 이 논문은 센서 설계, 3D 표현, 정책 학습을 한 시스템으로 묶었습니다. 각 요소의 효과를 따로 측정한 실험은 표 1(점군 대 이미지), 표 2(카메라), 그림 6(해상도) 세 가지입니다.

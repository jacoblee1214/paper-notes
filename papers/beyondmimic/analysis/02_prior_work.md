# 02. 선행연구와 관련 자료: BeyondMimic

## 1. 계보

| 줄기 | 대표 연구 (논문 번호는 `paper/paper.txt` 참고문헌) | BeyondMimic과의 관계 |
|---|---|---|
| 모델 기반 계층 제어 | 저수준 추종 + 저빈도 계획 (1–12) | 논문이 한계로 든 대안. 동작 자연스러움이 떨어진다고 주장 (p.2–3) |
| 모션 추적 RL (DeepMimic 계열) | DeepMimic (32), ASAP (33), 단일 다리 균형 (34), 무술 (35) | 직접 선행. 동작마다 튜닝이 필요하다는 점을 비판. ASAP의 게인은 비교 대상 (보충 S2) |
| 스타일 사전 분포 | AMP (36), 스타일 학습 (37) | 재사용이 어렵다는 점을 비판 |
| 계층형 추적 + 계획 | 과제 무관 추적기 (38–41) + 계획기 (42, 43) | 계획과 추적의 불일치 (44, 45) |
| 다과제 생성 모델 | VAE 기반 기술 조합 (46, 47) | 명시적 목표 조건이 필요하다는 점을 비판 |
| 확산 기반 제어 | 행동만 모델링하는 확산 정책 (60, 61); 물리 캐릭터 확산 (45 Diffuse-CLoC, 67, 68) | 상태-행동 공동 잠재 확산이 차이점 (p.18–19) |
| 분류기 안내 | 이미지 생성 (65), RL (66) | 제어 비용 함수를 테스트 시점에 더함 |
| 잠재 행동 표현 | 잠재 공간 (74, 78) | 행동을 VAE 잠재로 압축 |

## 2. 구성 요소별 출처

| 구성 요소 | 논문 출처 | 확인 |
|---|---|---|
| Rot6D 관찰 | (62) | 코드 `robots`·관찰 설정 확인, 04 참조 |
| 대칭 증강 (시상면) | (93) | 논문 보충 S4. 공식 코드에는 없음 (04 D) |
| PDP 섭동 데이터 | (85) | 보충 S3 |
| OU 잡음 | (86) | 보충 S3 |
| 비용 장벽 함수 | (87) | 보충 S3, 식 S8 |
| 트랜스포머 | (92) | 보충 S4 |
| 상태 추정 (관성·칼만 필터) | (89), LIO (90) | 보충 S4 |
| 자코비안 자동 미분 | CppAD (94) | 보충 S4 |

## 3. 이 저장소의 관련 분석

| 폴더 | 논문 | BeyondMimic과의 연결 |
|---|---|---|
| `supermimic/` | SuperMimic (arXiv:2610.28480, CoRL 2026 구두) | 제1저자 Liao가 같음. 부록 A가 BeyondMimic 추적 보상을 가져와 자기충돌 항을 빼고 관절 한계 가중치를 −10에서 −2로 낮춤. 종료 조건 |e_z| > 0.25 m, e_tilt > 0.8도 같음 |
| `sonic_humanoid/` | SONIC (arXiv:2511.07820) | 동작 추종 기준선으로 BeyondMimic을 씀. test-content 성공률 82.0%(BeyondMimic) 대 98.5%(SONIC). 기준선은 다른 데이터로 학습됨 |
| `phuma/` | PHUMA (표 19 등) | 실물 G1 실험에 BeyondMimic 기반 정책을 씀(MaskedMimic 학습 코드는 BeyondMimic 기반 수정본이라고 밝힘) |
| `impedance_mimic/` | ImpedanceMimic (arXiv:2611.19153) | 임피던스를 행동 공간으로 쓰는 추종 방식. BeyondMimic의 고정 PD 게인과 대비되는 설계 (직접 비교는 하지 않음) |


## 4. 공식 코드와 저장소 (GitHub 표)

조회일 2026-11-22, `gh api`로 메타데이터만 확인했습니다. 클론은 `HybridRobotics/whole_body_tracking` 하나만 했고(얕은 클론, 커밋 `9db915838b17`, 2026-10-15), 정적 읽기만 했습니다.

| 저장소 | 역할 | 라이선스 | 별 | 마지막 push | 상태·비고 |
|---|---|---|---|---|---|
| [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) | 논문의 추종 학습 코드 (Isaac Lab) | MIT | 2,443 | 2026-10-16 | **아카이브됨**. README 첫 줄에 "retired, no longer maintained"와 mjlab 이전 안내. 클론 커밋 `9db915838b17`(2026-10-15) 기준 43개 파일. 확산 제어 코드는 없음 |
| [HybridRobotics/motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller) | 시뮬-시뮬, 실로봇 배포 컨트롤러 (README가 가리킴) | MIT | 552 | 2026-09-10 | 메타데이터만 확인. 클론하지 않음 |
| [mujocolab/mjlab](https://github.com/mujocolab/mjlab) | 후속 프레임워크 (MuJoCo-Warp), 추적 과제 구현 | Apache-2.0 | 3,441 | 2026-11-21 | 활발히 유지보수됨. 논문 본문도 MJLab을 기본 방법으로 채택했다고 씀 (p.11) |
| Zenodo 데이터 10.5281/zenodo.17529720 | 보충 자료 dataset (79) | 확인 안 함 | - | - | 다운로드하지 않음 |
| beyondmimic.github.io | 프로젝트 페이지 | - | - | - | 영상과 링크. 학회 표기 없음 |

- 라이선스 확인: `whole_body_tracking`의 LICENCE 파일은 MIT입니다. 내부 파일 중 `Copyright (c) 2024, The Isaac Lab Project Developers`는 Isaac Lab의 템플릿 라이선스 문구입니다(LICENCE 파일 첫 줄).
- 논문이 말하는 "공개 코드"는 학습 추종 부분입니다. 확산 제어 부분의 코드, 학습된 VAE·확산 가중치, 실로봇 배포 C++ 코드의 공개 여부는 이 확인으로 판단할 수 없습니다 (미확인, U1).

## 5. 확인이 안 된 것

- 학회 채택 여부: 웹 검색에서 RSS·CoRL·ICRA 기록을 찾지 못했습니다. arXiv 주석과 프로젝트 페이지에도 학회 이름이 없습니다. 학회 표기 없음으로 둡니다.
- 확산 제어 코드 공개 여부 (U1 위 참조).

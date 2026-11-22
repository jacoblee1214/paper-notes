# 02. 선행연구와 관련 자료

## 1. 계보

AirVLA는 네 갈래를 한 시스템으로 묶은 논문입니다. 새 알고리즘이라기보다 **기존 부품을 비행체에 맞춰 조립한 시스템 논문**에 가깝습니다.

| 갈래 | 핵심 선행연구(논문 참고문헌 번호) | AirVLA가 가져온 것 |
|---|---|---|
| 조작 VLA 기반 모델 | RT-1 [6], RT-2 [7], Open X-Embodiment [12], Octo [34], π0 [3], GR00T N1 [2], OpenVLA [26] | 공개 π0 베이스 체크포인트를 그대로 미세조정 (부록 VII.A1) |
| 청크 정책의 실시간 실행 | Real-Time Chunking, Black, Galliker, Levine, arXiv:2506.07339 [4] | prefix 고정과 인페인팅, 속도장 수정 구조 (§III-C, III-D) |
| 항공 조작에 임바디먼트 인식 가이던스 | UMI-on-Air, Gupta 등, arXiv:2510.02614 [19] | 아이디어 (추론 시 가이던스). 대상이 확산 정책, 가이던스 원천이 MPC 기울기라는 점이 다릅니다 |
| 가우시안 스플랫에서 드론 정책 학습 | Sous Vide/FiGS, Low 등, RA-L 2025 [31]; SINGER [1], GRaD-Nav++ [9] | 장면 복원, 반운동학 동역학 (식 12–14), 교정 궤적 합성 (§III-E) |
| 조작 데이터 수집 하드웨어 | UMI, Chi 등, RSS 2024 [11]; UMI on Legs [20] | 손가락 형상 (§III-B1) |

가이던스의 수학적 뿌리는 분류기 없는 확산 가이던스(Ho & Salimans [22])와 의사역 가이던스(Song 등 [38])로 인용되지만, 실제 구현 구조는 RTC의 VJP 보정을 따라 쓴 것입니다.

## 2. 가장 가까운 대안과의 차이

| | UMI-on-Air [19] | SINGER [1] / GRaD-Nav++ [9] | AirVLA |
|---|---|---|---|
| 정책 | 단일 과제 확산 정책, 핸드헬드 UMI 시연으로 처음부터 학습 | 언어 조건 항법용 시각운동 정책, 3DGS 합성 학습 | 일반 VLA(π0)를 드론 데이터로 미세조정 |
| 가이던스 | 저수준 MPC의 기울기로 궤적을 동역학적으로 실현 가능하게 유도 | 해당 없음 | 고도 차원 하나에 페이로드 가중 손실 (식 6–9) |
| 접촉·조작 | 있음 (고정밀 항공 조작) | 없음 (항법만) | 있음 (집기·놓기) |
| 평가 | 논문이 "높은 정밀 과제에서 인상적인 결과"라고만 서술 | 논문이 sim-to-real 항법 성공을 서술 | 20회 단위, 실기체 460회 |

논문의 "첫 체계적 연구"라는 선점 주장(§I, §II-C)은 이 비교 위에 서 있습니다. 근거가 되는 문헌 조사 범위는 본문에 없어서, 우리는 이 주장을 검증하지 않았습니다(`04` C2).

## 4. 구성 요소별 출처

| 구성 요소 | 출처 | 논문에서의 위치 |
|---|---|---|
| 흐름 매칭 정책 | π0 [3], Lipman 등 [30] | §III-C, 식 1 |
| RTC (prefix 고정, 소프트 마스크) | Black 등 [4] | §III-C, §III-D1 |
| VJP 가이던스 구조 | RTC [4], 분류기 없는 가이던스 [22], 의사역 가이던스 [38] | §III-D, 식 2–3 |
| 3DGS | Kerbl 등 [24], Nerfstudio [39], gsplat [44] | §III-E1 |
| 그리퍼 마스크 | SAM [27], Grounded SAM [36] | §III-E2 |
| 드론 동역학과 교정 궤적 | FiGS/Sous Vide [31], ACADOS [43] | §III-E3, E4 |
| 기준선 | ACT [47], Diffusion Policy [10], LeRobot [8] | §IV-C, 부록 VII.A |
| OOD 해석 | LIBERO-Plus [15], LIBERO-PRO [48] | 부록 VII.B |

## 5. GitHub와 코드 자료 (확인 2026-11-23, GitHub API와 프로젝트 페이지)

**공식 코드는 없습니다.** 프로젝트 페이지의 Code와 Data 버튼은 "Coming Soon"입니다. 논문은 데이터셋을 "공개 예정"이라 씁니다(§I 기여).

| 항목 | 주소 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| 논문 | https://arxiv.org/abs/2603.25038 | CC BY 4.0 (arXiv 링크) | – | v1 2026-03-26 | v2 없음 |
| 프로젝트 페이지 | https://airvla.github.io | 사이트 하단 CC BY-SA 4.0(템플릿 문구) | – | 2026-03-28 (저장소 `airvla/airvla.github.io`) | Code·Data "Coming Soon". 본문과 표는 논문과 같음 |
| 공식 코드 | 없음 | – | – | – | 아래 비공식 후보도 비어 있음 |
| `Shadow581/AirVLA` | https://github.com/Shadow581/AirVLA | 없음 | 0 | 2026-06-12 | 커밋 하나(README만). "The code will be released when the paper be accepted." 논문 저자 계정인지 확인되지 않아 **비공식으로 취급**, 코드 없음 |
| π0 (openpi) | https://github.com/Physical-Intelligence/openpi | Apache-2.0 | 14,732 | 2026-08-24 | 베이스 모델. 최신 커밋 215abfb |
| LeRobot | https://github.com/huggingface/lerobot | Apache-2.0 | 29,025 | 2026-11-22 | ACT와 DP 기준선 학습 도구(부록 VII.A). 최신 커밋 e5be0c1 |
| RTC (Kinetix 실험) | https://github.com/Physical-Intelligence/real-time-chunking-kinetix | MIT | 637 | 2025-12-08 | RTC 원 구현. 최신 커밋 9296f31 |
| UMI-on-Air | https://github.com/LeCAR-Lab/UMI-on-Air | GitHub API 값 없음 | 75 | 2025-11-29 | 가장 가까운 기법의 코드. 최신 커밋 4830a3e |
| UMI | https://github.com/real-stanford/universal_manipulation_interface | MIT | 1,647 | 2026-05-26 | 그리퍼 손가락 출처 |
| Sous Vide (FiGS) | https://github.com/StanfordMSL/SousVide | GPL-3.0 | 26 | 2025-11-18 | 3DGS 드론 데이터 파이프라인. 최신 커밋 a2400aa |
| ACADOS | https://github.com/acados/acados | GitHub API 값 NOASSERTION | 1,529 | 2026-11-22 | 합성 궤적 적분 |
| Nerfstudio | https://github.com/nerfstudio-project/nerfstudio | Apache-2.0 | 12,240 | 2025-07-29 | 3DGS 학습 |

- 관련 저장소는 **복제하지 않았습니다.** 공식 코드가 없어서 `repo/`는 만들지 않았습니다. 대신 `04`에서 논문 주장을 C 항목으로 검증합니다.
- 프로젝트 페이지의 방법 설명은 논문 §III을 요약한 것이고, 표 I–III은 논문과 같은 값입니다. 프로젝트 페이지에는 식, 하이퍼파라미터(λz, Δz, γ, K), 부록이 없습니다.
- 프로젝트 페이지 하단의 CC BY-SA 4.0 문구는 사이트 템플릿(Nerfies) 라이선스 표기입니다. 논문 자체 라이선스와는 별개입니다.

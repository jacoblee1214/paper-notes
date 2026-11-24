# 02. 선행연구와 자료: Coupled Local and Global World Models (arXiv:2602.06219 v3)

## 1. 계보

| 흐름 | 대표 연구 | 이 논문과의 관계 |
|---|---|---|
| 모델 기반 RL, 잠재 상상 | DreamerV1–V3 [6, 7], DayDreamer [23] | 국소 RSSM의 출처(DreamerV3). DayDreamer는 실로봇 학습이지만 VAE 기반이라 표현력이 제한된다고 평가(§2.1) |
| 1차 기울기(FoG) MBRL | MAAC [8], PWM [30], SAPO [42] | 역전파로 해석적 기울기를 얻는 계열. SAPO 목적함수를 DMO에 결합(부록 A.5) |
| 순전파·역전파 모델 분리 | DMO [15] (Amigo 외, CoRL 2025, arXiv:2509.00215) | 직접적인 선행 연구. 순전파는 시뮬레이터, 역전파는 학습 모델. 이 논문은 시뮬레이터를 확산 세계모델로 대체 |
| 영 차수 최적화, 계획 | MBPO [25], PETS [16], TD-MPC [17, 18], MCTS 계열 [19–22], zero-order 개관 [29] | 기준 대안. 확산 표본이 많이 필요해 비싸다는 논지(§2.2, §2.3) |
| 확산 세계모델 | DIAMOND [31], DreamerV4 [14], shortcut [32], diffusion forcing [33] | 전역 모델의 구성 요소 |
| 세계모델 안의 RL (실로봇 지향) | World-Gymnast [37], residual latent action [36], URWM [28] | 이 논문은 BC 정규화 없이 처음부터 학습한다는 점을 차별점으로 제시(§4.1) |
| 보상 학습 | Rank2Reward [34] | Push-T 대조 보상 모델 |
| 저수준 제어, BC | SONIC [38], ACT [40], LeRobot [43] | G1 저수준 제어, BC 기준선 |

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 논문 위치 |
|---|---|---|
| 분리형 기울기(순전파 값, 역전파 야코비안) | DMO [15] | §3.1 |
| 이미지 공간 결합(디코드 후 재인코딩) | 이 논문의 변경 | §3.4 |
| 국소 모델 RSSM | DreamerV3 [7] | §3.4, 표 4 |
| Push-T 전역 모델 | DIAMOND [31] | §3.2 |
| Push Cube, G1 전역 모델 | DreamerV4 [14] 구조(공개 비공식 구현 `dreamer-v4`) | §3.2 |
| 정책 목적함수 | SAPO [42] | 부록 A.5 |
| 세계모델 패칭 | 이 논문(착취 정책을 실로봇에 배포해 데이터 수집) | §3.5, 부록 A.2 |

## 4. GitHub와 코드 자료 (확인 2026-11-24)

| 자료 | 링크 | 라이선스 | 별 / 마지막 push | 비고 |
|---|---|---|---|---|
| **이 논문의 공식 코드** | 없음 | - | - | 프로젝트 페이지에 "Code (release soon)" 단추만 있음. GitHub에서 찾지 못함 |
| 프로젝트 페이지 | https://coupled-global-local-wm-rl.pages.dev/ | - | - | 영상, 슬라이드, BibTeX. 코드 링크 없음 |
| DMO 코드(직접 선행 논문) | https://github.com/machines-in-motion/DMO_code | NOASSERTION (LICENSE.md 있음) | 26 / 2025-11-05, `f60a9dd` | DFlex, IsaacGym 환경의 DMO-SHAC. README에 DMO-SAPO는 "coming soon". 이미지 공간 코드 없음 |
| DMO 프로젝트 사이트 저장소 | https://github.com/machines-in-motion/DMO | 없음 | 0 / 2025-09-18 | 페이지 템플릿 |
| Dreamer-V4 비공식 구현(같은 연구실) | https://github.com/machines-in-motion/dreamer-v4 | GPL-3.0 | 10 / 2026-07-08, `c319a28` | 토크나이저와 동역학 모델, KV 캐시 실시간 환경. 체크포인트는 HF에 공개(PushT 110M, SOAR 110M). RL 코드 없음 |
| DIAMOND | https://github.com/eloialonso/diamond | MIT | 2,134 / 2024-12-06, `5bcd159` | Push-T 전역 모델의 원형 |
| DreamerV3 (공식, JAX) | https://github.com/danijar/dreamerv3 | MIT | 4,014 / 2026-10-04, `e01491f` | |
| DreamerV3 PyTorch 재구현 | https://github.com/NM512/dreamerv3-torch | MIT | 888 / 2026-03-08, `6ef8646` | |
| LeRobot (ACT 기준선) | https://github.com/huggingface/lerobot | Apache-2.0 | 29,057 / 2026-11-24 | |
| SONIC | https://github.com/NVlabs/GR00T-WholeBodyControl | 소스 Apache-2.0, 가중치 별도 | 3,943 / 2026-11-03 | G1 저수준 제어 |

공식 코드가 없으므로 clone하지 않았고, 논문과 코드의 대조(D 항목) 대신 주장 검증(`04_claim_verification.md`)을 썼습니다. 관련 저장소는 README와 파일 목록만 `gh api`로 읽었고 실행하지 않았습니다.

## 5. 가까운 대안

- **영 차수 정책 최적화를 세계모델 안에서:** PPO, SAC(이 논문의 기준선), Dreamer 계열 액터-크리틱. 이 논문의 비교 대상이지만 세계모델 롤아웃 수와 시간으로만 비교됩니다.
- **계획:** CEM-MPC를 같은 세계모델에 적용했을 때 128 병렬 롤아웃에 H200 16장 이상이 필요하다는 보고(부록 A.8, 지연 시간 수치 없음).
- **전역 모델을 직접 미분:** 참 기울기에 가까운 기준이지만 단일 상태 16스텝에 45.6 GB와 1.5 s(부록 A.3).

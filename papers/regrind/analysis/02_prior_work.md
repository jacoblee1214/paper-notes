# 02. 선행연구와 자료 지도: REGRIND (arXiv:2607.11874)

## 1. 연구 계보 (논문이 스스로 쓰는 위치)

논문은 두 흐름을 잇습니다(§1, §2).

1. **휴머노이드 전신 제어의 "리타게팅 후 추적" 레시피.** 사람 동작을 로봇 운동학 기준으로 바꾸고, 시뮬레이션에서 RL로 추적하는 방식입니다. BeyondMimic [1], OmniRetarget [2]이 대표입니다. 논문은 OmniRetarget의 상호작용 메시 공식을 손 매니퓰레이션으로 옮깁니다(§2 "Motion Retargeting from Human Data").
2. **다지 손 RL의 외부 사전 정보.** 접촉 사전 정보나 레퍼런스 궤적을 탐색 분포에 넣는 방식입니다. DexMachina [7](기능적 리타게팅 + 궤적 추적), ManipTrans [6](잔차 학습), SimToolReal [21], CHORD와 비슷한 접촉 보상 계열, VLM 계획 [23]이 같은 흐름입니다. 논문은 "접촉 사전 정보를 명시하지 않고 기준 궤적에 암묵적으로 담는다"는 점을 대비점으로 씁니다(§3.3, Reward Function).

REGRIND의 새로운 점은 **(a) 손-물체 상호작용 보존 리타게팅을 기준으로 쓰는 것, (b) 기준을 탐색 분포(RSI)와 행동 기준(잔차)으로 두 번 쓰는 것, (c) 궤적 증강을 리타게팅 재실행 없이 강체 변환으로 하는 것**입니다(§3.3, Data Augmentation).

## 2. 구성 요소별 출처 (주장에 대한 근거)

| 구성 요소 | 출처 (논문 번호) | 절 |
|---|---|---|
| 상호작용 메시 리타게팅 | OmniRetarget [2], Ho 외 [44] | §3.2, App. A.2 |
| 라플라시안 좌표와 Delaunay 사면체분할 | Si & Gärtner [48] | Eq. 4–5 |
| 탐색 분포(RSI) | Kakade & Langford [45], DeepMimic [46] | §3.1, §3.3 |
| 잔차 행동과 PD 제어 | 논문 자체 (기준 + α⊙π) | §3.3 |
| 6D 회전 표현 | Zhou 외 [49] | §3.3 |
| 키포인트 MANO 손 모델 | Romero 외 [47] | §3.2, App. A.1 |
| ARCTIC 가위 데이터 | Fan 외 [50] | §4.1, App. C |
| LEAP 손, WUJI 손 | Shaw 외 [51] / 논문 본문 | §4.1, App. E |
| HaMeR 손 자세 추정 | (App. C 인용) | App. C |
| SPIDER 기준선 | Pan 외 [41] | §4.1, App. D.2 |
| DexMachina 기준선 | Mandi 외 [7] | §4.1, App. D.2 (자체 구현) |
| Mink IK | Zakka [52] | §4.1, App. D.2 |
| PPO, RSL-RL | (App. B.6, 저장소 `rsl_rl`) | App. B.6 |
| IsaacSim 5.1 | (App. B.1) | App. B.1 |

## 4. GitHub와 코드 자료 (확인 2026-11-15)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| 프로젝트 페이지 | https://yunhaifeng.com/REGRIND | - | - | 영상, PDF, 코드 버튼 (`github.com/yunhaif/regrind`) |
| 공식 코드 | https://github.com/yunhaif/regrind | MIT (저장소 LICENSE, 2026) | 별 122, 포크 11, 커밋 1개(`38347a9`, 2026-07-14 커밋 시각) | Isaac Lab 확장 `source/regrind`, 학습·재생 스크립트, 리타게팅 스크립트. 정적 읽기만 함 (`repo/`, sparse clone) |
| 공식 코드 데이터 | 같은 저장소의 `data/` | 저장소 라이선스에 포함 | 미확인 | 시연 h5, 리타게팅 궤적 h5, 키포인트 npy, 영상 포함. 받지 않음 |
| LEAP 손 URDF | `source/regrind/regrind/assets/leaphand_urdf/` | 디렉터리 LICENSE.txt (2023, MIT 계열 문구) | - | 공식 저장소 안의 자산 |
| IsaacSim 5.1.0, IsaacLab 2.3.0 | NVIDIA | 확인 안 함 | - | 학습에 필요. 이 맥에서는 설치 불가(CUDA 없음) |
| Drake (pydrake) | pip | 확인 안 함 (BSD 계열로 알려짐, 이 세션에서 미확인) | - | 리타게팅에 필요 (`[retargeting]` extra) |
| MOSEK | 상용 (학술 무료 라이선스 있음) | 상용 | - | 기본 QP 솔버. Clarabel이 무료 대안 |
| SPIDER | https://github.com/facebookresearch/spider | 확인 안 함 | - | App. D.2의 공식 구현. 이 저장소에 없음 |
| Mink | https://github.com/kevinzakka/mink | 확인 안 함 | - | App. D.2의 IK 기준선 |
| DexMachina | 공개 저장소 없음(확인 안 함) | - | - | 논문의 자체 구현 (App. D.2) |
| HaMeR | https://github.com/geopavlakos/hamer | 확인 안 함 | - | App. C의 손 자세 추정 |
| ARCTIC | https://github.com/zc-alexfan/arctic | 확인 안 함 | - | 가위 시연 원자료 |

- 공식 저장소를 **정적으로만** 읽었습니다. 설치하거나 실행하지 않았습니다(`analysis/04_code_verification.md` §1).
- `repo/`는 `.gitignore`에 넣었습니다.

## 5. 이 논문이 기대는 외부 결과

- **OmniRetarget의 상호작용 메시 공식 [2]:** 리타게팅 목적과 SQP 해법을 그대로 따릅니다(§3.2). 논문은 이 공식이 손 매니퓰레이션에서도 같은 역할을 한다고 가정하며, 그 근거는 제시하지 않습니다.
- **DexMachina [7]의 기능적 리타게팅과 충돌 후처리:** 논문은 DexMachina의 충돌 후처리가 손가락을 엉뚱한 위치로 밀어낸다고 주장합니다(§4.2 Q1). 이 주장은 그림 5의 정성 비교에 기대고 있고 정량 근거가 없습니다.
- **Isaac Lab 학습 환경:** 4096 병렬 환경, 도메인 무작위화 구성 방식은 Isaac Lab의 표준을 씁니다(App. B.4).
- **SPIDER 실패(Table 1, 2에서 0%):** 논문은 "여러 하이퍼파라미터에서" 실패했다고 쓰지만(§4.2 Q2), 튜닝 범위와 시간 예산은 보고하지 않습니다. 기준선 공정성의 가장 큰 물음표입니다.

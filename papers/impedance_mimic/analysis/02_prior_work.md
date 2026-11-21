# ImpedanceMimic 선행연구와 연결 (arXiv:2611.19153)

## 1. 선행연구 계보 (구성 요소별 출처)

| 구성 요소 | 가져온 것 | 출처 (논문 절) | 이 논문의 차이 |
|---|---|---|---|
| 물리 기반 모방 | 시뮬레이션에서 기준 동작을 추종해 동역학 일관성 얻기 | Peng et al. (2018), Luo et al. (2023) PHC (§2) | 측정 힘과 CoP로 보상 추가 |
| 행동 공간 비교 | PD 목표 위치가 토크, 속도, 근육보다 낫다는 결과 | Peng & Van De Panne (2017) (§3.1) | 이 결과를 힘 정확도까지 넓힘 |
| 가변 임피던스 학습 | 접촉에서 강성, 감쇠를 학습 | Buchli et al. (2011), Bogdanovic et al. (2020) (§2) | 관절 단위 PD 이득, 힘 측정과 비교 |
| 물리 모방에서의 이득 학습 | 기준 PD 이득을 매 스텝 조정, 잔여 외력 추가 | SimPoE (Yuan et al., 2021) (§2, §4.4) | 잔여 외력 없이 비교, 잔여 외력이 생존율을 떠받친다는 점 보고 |
| 강성, 감쇠 행동 | 강성과 감쇠를 행동에 포함 | ReGAIL (Boursin et al., 2025) (§2) | 힘 측정과 맞춤 |
| 힘 추정 기준 | 운동학에서 GRF 회귀 | GaitDynamics (Tan et al., 2026) (§3.2, A.3.7) | 시뮬 없는 회귀기를 힘 정확도 참조로 사용 |
| 데이터 | 힘판과 관절각 쌍 | AddBiomechanics (Werling et al., 2024), Scherpereel et al. (2023) (A.1) | 9개 데이터셋, 129명 |
| 근육 모델 | 80개 Hill형 근육 | MyoConverter (Wang et al., 2022; Ikkala & Hämäläinen, 2022), MuscleMimic (Li et al., 2026) (A.2.3) | 근육 기준선 |
| 인체 임피던스 | 발목 강성과 감쇠 | Rouse et al. (2014), Loram & Lakie (2002), Lee & Hogan (2015) (A.3.6) | 크기는 맞지 않음 (논문 자체 보고) |
| 시뮬레이터 | 배치 MuJoCo | MuJoCo Warp (Google DeepMind & NVIDIA, 2025), Todorov et al. (2012) (A.2.2) | — |

## 2. 코드와 저장소 (GitHub 표)

확인일 2026-11-21. GitHub API로 얻은 메타데이터입니다. 라이선스 "없음"은 API가 `null`을 돌려준 경우입니다.

| 저장소 | 관계 | 라이선스 | 최근 push | 별 | 상태 |
|---|---|---|---|---|---|
| [ImpedanceMimic/ImpedanceMimic](https://github.com/ImpedanceMimic/ImpedanceMimic) | 논문 재현성 항목이 가리키는 저장소. 실제로는 프로젝트 페이지 저장소 | 없음 (API null) | 2026-11-16 (생성 2026-08-22) | 4 | **코드 없음.** README "Code Coming (very) Soon!". 루트에는 README.md, index.html, assets/(영상)만 있음. 커밋 5개(웹사이트와 영상). 클론하지 않음 |
| [stanfordnmbl/GaitDynamics](https://github.com/stanfordnmbl/GaitDynamics) | GaitDynamics 공식 코드 (Tan et al., 2026), 기준선 | MIT | 2026-11-15 | 138 | 확인만. 설치, 실행 안 함 |
| [ZhengyiLuo/PHC](https://github.com/ZhengyiLuo/PHC) | PHC 공식 코드, 기준선 | NOASSERTION (LICENSE 확인 필요) | 2025-08-21 | 1309 | 확인만 |
| SimPoE 공식 코드 | 기준선 | 검색으로 찾지 못함 | — | — | 미확인. 논문은 "as faithfully as paper and code allow"로 재구현했다고 씀 |
| [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp) | 학습 시뮬레이터 | Apache-2.0 | 2026-11-20 | 1646 | 확인만. Warp의 GPU 경로는 CUDA 전제 (`04` §4) |
| [MyoHub/myoconverter](https://github.com/MyoHub/myoconverter) | OpenSim→MuJoCo 변환 | Apache-2.0 | 2024-10-22 | 129 | 확인만 |
| [keenon/AddBiomechanics](https://github.com/keenon/AddBiomechanics) | 데이터 처리 파이프라인 | NOASSERTION | 2026-08-03 | 69 | 확인만 |

**결론.** 이 논문의 "공식 코드"는 아직 없습니다. 따라서 `04_code_verification.md` 대신 `04_claim_verification.md`로 주장을 검증했습니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- Peng & Van De Panne (2017)의 행동 공간 비교는 운동학 추종에서 PD 목표가 최고, 토크가 최악이라는 결과입니다. 이 논문은 그 결과가 힘에서는 뒤집히지 않는다고 보여 줍니다. 그러나 토크 기준선이 힘 정확도에서 이 논문과 거의 같다는 점(Table 1 seen, 0.82 대 0.80)은 그 결론을 부분적으로만 지지합니다.
- Bogdanovic et al. (2020)의 "접촉에서 k_d 효과가 작다"는 관찰이 Bound 1의 근거입니다. 이 관찰이 일반적으로 맞는지는 로봇 손(외부 접촉 객체) 맥락에서 따로 확인해야 합니다.
- SimPoE의 잔여 외력 결과 (Table 19)는 "생존율"이 물리 일관성을 보장하지 않는다는 이 논문의 경고를 뒷받침합니다.

# 02. 선행 연구, 코드, 구성 요소별 출처

## 1. 선행 연구 계보

이 논문은 세 갈래를 합칩니다. 각 갈래에서 이 논문이 새로 한 것과 빌려 온 것을 나눠 적습니다. 참고문헌 번호는 v2 기준입니다.

### 1.1 로봇 학습 데이터 오염과 백도어

| 갈래 | 대표 | 이 논문과의 관계 |
|---|---|---|
| DRL 백도어 | TrojDRL [45], BackdooRL [40], SleeperNets [44], Beware untrusted simulators [42] (같은 1저자·2저자·교신 저자의 이전 연구) | 백도어 정의(공격 성공, 공격 은밀성, 식 2)와 "트리거 객체 + 목표 행동" 구성을 [42,44]에서 가져옵니다. SleeperNets는 보상 조작으로 정책을 바꾸고, [42]는 시뮬레이터를 신뢰하지 않는 설정입니다. 이 논문은 조작 지점을 **월드 모델 입력**으로 옮깁니다. |
| BC/VLA 데이터 오염 | BadVLA [43], State backdoor [41], Goal-oriented physical backdoor [52] | 직접 위험한 궤적을 데이터셋에 넣기 때문에 "검사하면 보인다"는 것이 이 논문이 지적하는 한계입니다(§3). |
| 멀티모달 임베딩 공격 | Adversarial Illusions [50] (공동 저자 Bagdasarian 포함) | 이미지 섭동으로 의미 임베딩을 목표 입력의 것에 맞추는 아이디어를 VPH에서 월드 모델의 조건 인코딩에 적용합니다. |
| 간접 프롬프트 주입, 탈옥 | Greshake et al. [22], Zou et al. [20] | 동기 부여로 인용합니다. 월드 모델에는 텍스트를 이미지에 넣는 방식이 통하지 않는다는 점이 차이로 제시됩니다(§5.1). |

### 1.2 월드 모델과 합성 로봇 데이터

| 구분 | 대표 | 비고 |
|---|---|---|
| 텍스트 조건 영상 월드 모델 | Cosmos-Predict2.5 [10], Cosmos 플랫폼 [23], Wan [49], Waymo world model [19] | 공격 대상은 Cosmos-Predict2.5 2B |
| 합성 궤적 → IDM → 정책 | DreamGen [55, NVIDIA 블로그 17], MimicGen [9] | 이 논문이 다루는 파이프라인(그림 2 위쪽). 논문은 DreamGen의 IDM이 새 장면에 일반화되지 않아 완성하지 못했다고 적음 |
| 행동 조건 월드 모델 | DINO-WM [15], LeWorldModel [16], Cosmos AC, VLAW [12], RISE [13], Interactive World Simulator [39], Dreamer 4 [11] | 공격 대상은 DINO-WM(Single Wall)과 Cosmos-Predict2.5 AC, 부록에서 r2Dreamer(참고문헌 없음) |
| 월드 모델 안 RL | Dreamer 4 [11], VLAW [12] | VTH의 DRL 시나리오 근거 |
| 정책 평가용 월드 모델 | Gemini Robotics + Veo [14] | 평가 경로의 보안은 이 논문이 다루지 않음 |

### 1.3 적대적 섭동

| 요소 | 출처 | 비고 |
|---|---|---|
| PGD + 모멘텀 | Madry et al. [59] | 논문은 "표준 PGD with momentum"이라 적음. 모멘텀 계수 0.9 |
| LAB 색공간 제약 | CIEDE2000 [46], Zhao et al. [47], Aydin & Temizel [48] | 지각 색거리 기반 제약. 이 논문은 L, AB 채널에 서로 다른 예산(5, 20)을 둠 |
| 적대 예제의 화이트박스 평가 | Carlini & Wagner [21] | "화이트박스는 강한 가정이지만 취약성 이해에 핵심" 인용 |

## 2. 공식 코드와 자료 (2026-11-24 확인)

| 항목 | 위치 | 라이선스 | 스타 / 포크 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| 논문 / arXiv | arxiv.org/abs/2606.09499 (v1 2026-06-08, v2 2026-10-07) | CC BY 4.0 | | | |
| 프로젝트 페이지 | **찾지 못함**. arXiv 페이지, 논문 본문, 저자 홈페이지(UMass), GitHub 검색, Hugging Face 검색 모두 결과 없음 | | | | |
| 코드 | [EthanRath/cosmos-predict2.5](https://github.com/EthanRath/cosmos-predict2.5) (nvidia-cosmos/cosmos-predict2.5의 포크) | Apache-2.0 (포크 원본의 LICENSE) | 0 / 0 | 2026-06-16 | **논문과 arXiv가 링크하지 않음.** 제1저자 계정으로 보임(같은 계정의 다른 저장소 README가 SleeperNets와 Q-Incept를 "우리 연구"로 적고, 논문 참고문헌 [44]가 SleeperNets). 브랜치 5개: `main`(원본 미러), `wm-poison-modifications`(2026-04-10, 12커밋), `delta`(04-06), `ethan`(05-12), `gamma-branch`(06-16, 원본보다 54커밋 앞). 읽은 브랜치와 커밋: `gamma-branch` `8a924ea`. README와 설명 없음 |
| 베이스 모델 코드 | [nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) | Apache-2.0 | 약 1,400 | 2026-06-08 | 논문의 공격 대상 |
| DINO-WM | [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) | MIT | 약 630 | 2025-03-24 | VTH 공격 대상(Single Wall) |
| 백도어 RL 통합 구현 | [EthanRath/Backdoors-In-RL](https://github.com/EthanRath/Backdoors-In-RL) | MIT | 4 | 2025-08-02 | SleeperNets, Q-Incept, BadRL, TrojDRL의 CleanRL PPO 구현. 월드 모델 코드는 없음 |
| 선행 논문 코드 | [EthanRath/Daze-Backdoor-Attack](https://github.com/EthanRath/Daze-Backdoor-Attack) ([42]의 코드) | 파일 없음 | 1 | 2026-03-17 | 같은 저자의 시뮬레이터 백도어 |
| 선행 논문 코드 | [EthanRath/SleeperNets_NeurIPS](https://github.com/EthanRath/SleeperNets_NeurIPS) | MIT | 8 | 2025-06-03 | [44] |
| 모델 가중치 | Hugging Face에서 검색되지 않음 | | | | 논문 부록 A.1은 "탈익명화 뒤 Hugging Face에 LIBERO 미세조정 모델을 올릴 계획"이라고 적음 |
| 데이터 | LIBERO 미세조정 데이터(Cosmos-Policy 재렌더), 새 과제 3개(Stove, Microwave, Gift Box)와 위험 자산 | 공개 안 됨 | | | 저장소에는 장면 메타데이터 CSV 형식과 물체 이름만 코드에 있음 |

**읽은 범위.** `gamma-branch`를 필요한 경로(스크립트, 월드 모델 수정 4개 파일, 실험 설정 1개)만 얕게 clone해서 정적으로만 읽었습니다. 코드는 실행하거나 설치하지 않았습니다. 코드 대조는 `04`에 있습니다.

## 4. 구성 요소별 출처와 가까운 대안

| 구성 요소 | 출처 | 이 논문의 변형 |
|---|---|---|
| 확산/정류 흐름 벡터장 손실 | Cosmos-Predict2.5 [10], DDPM [51] | 두 프롬프트에서 나온 속도장의 MSE를 섭동 이미지에 대해 최소화 (부록 A.1) |
| PGD + 모멘텀, 스텝 400 | [59] | α를 채널 예산의 비율로 해석, L과 AB 예산 분리 |
| 백도어 정의, 은밀성 지표 | [42,44] | 월드 모델 입력 경로로 확장 |
| 트리거 객체 | [42,44,45] | 영상 안 물체. VPH 실험에서는 별도 트리거 없이 위험 물체가 장면에 이미 있음 |
| 행동 조건 붕괴 손실 (식 4, 5) | 인코더 코사인 공격 일반 | 행동 샘플링 4+4, a⁺ 근처 ε 교란 |
| 보상 (식 6) | DINO-WM의 잠재 거리 | 표기 확인 필요 (`03` K6) |
| PPO | CleanRL [60] | |
| 수동 라벨링 | MLLM-as-a-Judge [54]의 한계 인용 | Qwen3-VL 정확도 표(표 4) |

**가까운 대안(비교 관점)**: (1) 학습 데이터에 직접 위험 궤적을 넣는 BadVLA류 [43]는 은밀성이 낮고 효과가 직접적입니다. (2) 월드 모델 가중치 자체를 오염시키는 방식(체크포인트 공급망)은 이 논문의 범위 밖이며 같은 위협 모델에서 더 강한 공격자입니다. (3) 프롬프트 주입 방어(필터링 후 입력)와 달리 이 공격은 이미지 픽셀에 들어 있어 텍스트 필터가 소용없습니다.

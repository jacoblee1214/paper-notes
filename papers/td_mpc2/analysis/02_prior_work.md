# 02. 선행연구와 관련 저장소

이 문서는 TD-MPC2의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드가 어디 있는지를 정리합니다. 번호는 논문 참고문헌 번호가 아니라 저자 이름과 연도로 씁니다(논문 `paper/paper.txt`의 References).

## 1. 선행연구 계보 (구성 요소별 출처)

| TD-MPC2의 요소 | 출처 | 관계 |
|---|---|---|
| 잠재 공간의 국소 궤적 최적화 (MPPI) | TD-MPC (Hansen, Wang, Su, ICML 2022); MPPI (Williams 외, 2015) | TD-MPC2의 직접 전신. 계획 절차는 TD-MPC를 따르고, 운동량을 빼고 처리량을 약 2배 올렸다고 씁니다(부록 A) |
| 디코더 없는 잠재 세계 모델 | TD-MPC; 목표 불일치 논의 (Lambert 외 2020) | 재구성 대신 보상, 가치, 정책 사전 분포로 학습합니다. 저자는 재구성 목표가 긴 지평에서 제어에 도움이 안 된다고 씁니다(§3.1) |
| 결합 임베딩 예측 (stop-grad) | BYOL (Grill 외, 2020) | 식 3의 잠재 예측 항 |
| 두 Q 최솟값 TD 목표, Q 앙상블 | REDQ (Chen 외, 2021); 이중 Q (van Hasselt 2016) | 목표는 무작위 두 Q̄의 최솟값, Q는 5개 |
| 로그 변환 공간 이산 회귀 (two-hot) | C51 (Bellemare 2017); DreamerV3 (Hafner 2023) | 보상 크기 차이에 무관한 손실 |
| SimNorm (잠재 단순형 정규화) | Simplicial embeddings (Lavoie 외, 2022); VQ-VAE (van den Oord 2017)의 연속 완화 | 잠재 희소성, 노름 제한. 저자는 VQ-VAE의 연속 대응이라고 설명합니다(부록 H) |
| 최대 엔트로피 정책 사전 분포 | SAC (Haarnoja 2018); Ziebart 2008 | 식 4. TD-MPC의 결정적 정책과 잡음 스케줄 대신 사용 |
| LayerNorm, Mish, 드롭아웃 | Ba 2016; Misra 2019; Srivastava 2014 | 모든 MLP 층 |
| 다과제 학습 | Gato (Reed 2022), GSL (Jia 2022), 다과제 오프라인 Q 학습 (Kumar 2023) | 저자는 행동 복제형 대형 모델과 달리 온라인 RL 데이터로 다과제 학습을 한다고 구분합니다(§6) |
| 테스트 시 불확실성 패널티 | Feng 외 2023 (Finetuning offline world models in the real world) | 부록 J의 계획 패널티 (c 계수 하나로 과제 무관하게 정규화) |
| 기준선 | SAC (Haarnoja 2018), DreamerV3 (Hafner 2023), TD-MPC (2022), CURL (Srinivas 2020), DrQ-v2 (Yarats 2021) | 저자가 공식 구현을 씀 (부록 H) |

**가까운 대안과의 차이.** 같은 목적을 가진 두 방향이 있습니다. 하나는 재구성형 생성 세계 모델(Dreamer 계열)이고, 다른 하나는 모델 없는 RL(SAC)입니다. TD-MPC2는 둘 사이에서 잠재 계획 쪽에 머무릅니다. DreamerV3와 비교해 논문이 강조하는 차이는 (1) 학습 목적이 보상과 가치 예측뿐이라는 점, (2) 계획이 행동 샘플링이라서 UTD 1로 학습 가능하다는 점입니다. 특히 DreamerV3는 UTD 512를 씁니다(§4).

## 2. 코드와 저장소 (GitHub 표)

GitHub API와 저장소의 파일 내용(2026-11-21 확인). 복제는 얕은 복제(`--depth 1`)이고, 정적으로 읽기만 했습니다. 실행, 설치, import는 하지 않았습니다.

| 저장소 | 관계 | 라이선스 | 별, 최근 push | 커밋 | 상태 |
|---|---|---|---|---|---|
| [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2) | 공식 구현 (논문 저자) | MIT (LICENSE 파일, "Copyright (c) Nicklas Hansen (2023)", API spdx: MIT) | 1004, push 2026-10-20 | `332f19e4aba4bffe2dc736f45ea7f7bdc585903a` (2026-10-20, 커밋 메시지 "Preserve target Q history and evaluation mode during conversion") | 얕은 복제 `repo/`, 정적 읽기 완료 (`04`) |
| [nicklashansen/tdmpc](https://github.com/nicklashansen/tdmpc) | TD-MPC 공식 구현 (논문의 TD-MPC 기준선) | MIT (API spdx: MIT) | 530, push 2023-11-25 | `f4d85eca7419039b71bab2234ffc8aca378dd313` (2023-11-25) | 복제 안 함. 기준선 결과 파일은 `repo/results/tdmpc/`에 있음 |
| [tdmpc2.com](https://www.tdmpc2.com) (프로젝트 페이지, 모델, 데이터셋) | 300+ 체크포인트(다과제 12개 포함), 30과제·80과제 데이터셋 | 저장소와 같다고 명시되지 않음. 확인 못 함 | — | — | 받지 않음. 데이터셋과 체크포인트는 이 맥에서 쓰지 않음 |
| `tdmpc2.com/files/mjkey.txt` | README가 안내하는 MuJoCo 2.1 라이선스 키 (Meta-World용) | 재배포 키. 조건 확인 필요 | — | — | **받지 않음.** 키를 우리 폴더에 두지 않음 |
| [ShaneFlandermeyer/tdmpc2-jax](https://github.com/ShaneFlandermeyer/tdmpc2-jax) | 커뮤니티 JAX 구현 (GitHub API로 확인) | MIT (API spdx: MIT) | 84, push 2026-07-28 | — | 복제 안 함. 비공식 |

- 공식 저장소의 학습 곡선 파일(`repo/results/`, 4개 알고리즘 × 104과제 × 시드 3, 그리고 픽셀 12과제)은 `analysis/recompute.py`의 입력입니다.
- README가 GPU 전제이고, `tdmpc2/` 코드에 CUDA를 지정하는 줄이 13줄 있습니다. 이 맥에서 실행하려면 수정이 필요합니다(`04` §4).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **TD-MPC의 불안정성.** 저자는 TD-MPC가 연속 회귀와 비정규화 잠재 때문에 기울기가 폭발한다고 씁니다(부록 A, G). 저자는 TD-MPC 공식 구현을 DMControl과 Meta-World에서 권장 설정으로 돌렸다고 씁니다(부록 H). 발산이 설정 탓인지는 논문만으로 구분할 수 없습니다. 공개 곡선으로 본 결과는 `03` §3입니다.
- **DreamerV3의 UTD 512.** 기준선 비교가 "같은 샘플 수, 다른 계산량"이 됩니다. 샘플 효율 비교는 계산량 차이를 가리므로 벽시계 주장과 분리해 읽어야 합니다.
- **시각 입력 결과.** 시각 TD-MPC2 결과는 DrQ-v2, DreamerV3와 비슷하다고만 씁니다(그림 10). 기준선의 공개 결과 파일은 저장소에 없습니다.

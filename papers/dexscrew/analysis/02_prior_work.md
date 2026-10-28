# 02. 선행연구와 계보

번호 [n]은 논문의 참고문헌 번호입니다. 각 논문을 별도로 열어 검증하지는 않았습니다.

## 1. 계보: Malik·Qi 그룹의 손 안 회전 연구

```
Hora [5] (CoRL 2022)            Allegro, 회전 관절 없는 손 안 물체 회전, oracle → proprio 히스토리 적응(RMA [11] 방식)
  └ General in-hand rotation with vision and touch [17] (CoRL 2023)   시각과 촉각 추가 (촉각은 proxy 표현)
  └ Lessons from learning to spin pens [13] = penspin (CoRL 2024)    시뮬레이션 oracle → 실제 replay로 데이터 수집 → BC로 미세조정
  └ Twisting lids off with two hands [44] (CoRL 2024)               병뚜껑을 회전 관절과 휴리스틱 마찰로 근사
  └ Learning visuotactile skills with two multifingered hands [32]   텔레옵 + 촉각 BC
  └ ★ DexScrew                                                      회전 관절로 단순화한 RL 기술 + 그 기술을 쓰는 텔레옵 + 촉각 BC
```

**코드 계보:** README에 따르면 DexScrew 코드는 penspin, Hora, IsaacGymEnvs를 기반으로 합니다. privileged info 인덱스 버그(`04` D1)는 penspin의 `start_index = 9`가 `0`으로 바뀐 데서 생겼습니다.

**penspin과의 차이:** penspin [13]도 "시뮬레이션 정책으로 실제 데이터를 모으고 BC로 학습"하는 구조였지만, 그때는 시뮬레이션 정책을 그대로 **replay**했습니다. DexScrew는 정책을 **사람이 켜고 끄는 기술 버튼**으로 만들고 팔은 사람이 움직이게 해서, 시뮬레이션에 없는 축(하강)을 사람이 채우도록 했습니다.

## 2. 구성 요소별 출처

| 역할 | 선행연구 | 이 논문에서의 쓰임 |
|---|---|---|
| Privileged 학습 | Learning by Cheating [45], RMA [11] | oracle → sensorimotor 증류 |
| 증류 | DAgger [47] | 행동 + latent 손실 |
| RL | PPO [46], IsaacGym [6] | 8,192 env |
| 도메인 랜덤화 | Peng et al. [7], OpenAI [1] | 외력 2.0m, 확률 0.25 (OpenAI [1]의 설정을 따름) |
| BC | ACT [21], Diffusion Policy [20] | action chunking (H=16) |
| 정규화 | LBM [50] arXiv:2507.05331 | 관측 정규화 방식 |
| 네트워크 | Stacked Hourglass [49] | hourglass encoder |
| 해석 | Filtered BC [51] (Kumar et al., ICLR 2022) | BC가 replay보다 나은 이유로 인용 (`03` §4) |
| 비교 | Asymmetric actor-critic [52] arXiv:1710.06542 | Fig. 7 baseline |

## 3. 비슷한 문제를 다른 방법으로 푼 연구 (논문 Related Work 기준)

| 방법 | 접근 | DexScrew와의 차이 |
|---|---|---|
| DexNDM [36] arXiv:2510.08556 | 실제 상호작용으로 관절별 신경 동역학 잔차 모델을 만들어 sim-to-real 간극을 보정 | 동역학을 보정하는 쪽. DexScrew는 동역학을 고치지 않고 데이터로 우회함 |
| DexterityGen [37] arXiv:2502.04307 | 시뮬레이션 정책을 안정화 제어기로 쓰고 텔레옵을 결합 | 촉각을 쓰는 자율 정책을 만들지 않음 |
| Kumar et al. [38] | 확산 기반 접촉 탐색 + 궤적 최적화로 드라이버 회전 | 모델 기반 |
| FORGE [39] | 힘 기반 탐색으로 sim-to-real, 평행 그리퍼 | 다지 손이 아니고 재파지가 없음 |
| Transic [40] | 사람의 개입으로 sim-to-real 정책을 실제에서 보정 | 단순한 엔드이펙터 |
| Sim-and-real co-training [41] | 시뮬레이션과 실제 데이터 공동 학습 | 비전 기반 |

## 4. 참고문헌에서 발견한 문제
- **[42]의 서지 정보가 본문과 맞지 않습니다.** 본문은 "RLPD [42]"라고 인용하지만, [42]는 Zhang et al., "Traversability-aware legged navigation by learning from real-world visual data"(T-RO 2025)로 적혀 있습니다. 흔히 RLPD라고 부르는 논문은 Ball et al.의 오프라인 데이터 활용 온라인 RL 연구입니다(*제 사전 지식 기준이라 확인이 필요합니다*). 어느 쪽이든 본문의 약칭과 참고문헌 항목이 맞지 않습니다.

## 5. 논문이 다루지 않았지만 관련 있는 것 (내 판단)
- **MuJoCo의 joint equality 제약:** 나사의 회전-하강 결합을 시뮬레이터에서 싸게 표현할 수 있는 방법입니다(`03` §3).
- **꼬리 위험 지표(CVaR, 최악 k%)로 평가하기:** 촉각의 효과가 평균보다 꼬리에서 드러나므로, 이런 지표로 보고하는 편이 더 정확했을 것입니다.
- **같은 저장소 군의 Choice Policy:** minbc가 지원하는 다중 제안(multi-proposal) BC입니다. 이 논문에서는 쓰지 않았습니다.

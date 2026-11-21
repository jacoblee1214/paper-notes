# 02. 선행 연구와 관련 자료

## 1. 계보

```
MuZero (Schrittwieser et al., 2020)
 └─ EfficientZero (Ye et al., 2021): 자기지도 환경 모델 + 적응형 TD, MCTS, 이산 행동 (Atari)
     └─ EfficientZero V2 (Wang et al., 2024, ICML): 샘플링 Gumbel 탐색 + SVE + 혼합 가치 목표, 연속 행동 + 시각·저차원 입력
```

- **Gumbel 탐색 쪽.** Gumbel MuZero(Danihelka et al., 2021)는 Gumbel top-k와 Sequential Halving으로 큰 이산 행동 공간의 탐색 비용을 줄였습니다. 연속 행동은 다루지 않습니다(논문 §2.2).
- **연속 행동 MCTS 쪽.** Sampled MuZero(Hubert et al., 2021)는 행동 부분집합을 샘플하는 MCTS로 연속 제어에 MuZero를 적용했습니다. EZ-V2는 이 방식(Sample MCTS)을 기준선으로 삼고, 시뮬레이션 수가 절반 이하(32 대 50)에서도 낫다고 주장합니다(Fig. 3).
- **MPC 쪽.** TD-MPC(Hansen et al., 2022)와 TD-MPC2(Hansen et al., 2023)는 MPPI로 계획합니다. 논문은 TD-MPC2가 한 번의 결정에 9216개의 잠재 상태를 예측한다고 비교합니다(§5.2, 부록 J.3).
- **범용 세계 모델 쪽.** DreamerV3(Hafner et al., 2023)는 상상 공간에서 정책을 학습합니다. 논문은 이 방법이 여러 영역에서 SOTA와 차이가 있다고 보고, EZ-V2의 주된 비교 대상으로 삼습니다.

## 2. 가까운 대안 (이 논문이 밝힌 것)

| 방법 | 연속 행동 | 이산 행동 | 시각 입력 | 계획 방식 | 논문의 비교 |
|---|---|---|---|---|---|
| EfficientZero | 불가 | 가능 | 가능 | MCTS | 기준선, 적응형 TD 대비 SVE |
| Gumbel MuZero | 불가 | 가능 | 가능 | Gumbel + SH | 탐색 아이디어의 출발점 |
| Sample MuZero | 가능 | — | 가능 | 샘플 MCTS | 절제 실험 기준선 (Fig. 3, Table 4–5) |
| TD-MPC2 | 가능 | — (연속 중심) | 가능 | MPPI | Proprio 기준선, 계산량 비교 |
| DreamerV3 | 가능 | 가능 | 가능 | 정책 학습(상상) | 주요 기준선 |
| BBF | 제한 (Atari 중심) | 가능 | 가능 | 모델 없음 | Atari 기준선 |
| SAC | 가능 | — | 불가 | 모델 없음 | Proprio 기준선 |
| DrQ-v2 | 가능 | — | 가능 | 모델 없음 | Vision 기준선 |
| CURL | 가능 | — | 가능 | 모델 없음 | Vision 기준선 (표현 학습) |

## 3. 논문의 기여 판단

- **새로운 것.** 샘플링 기반 Gumbel 탐색(연속 행동에서 정책 개선 조건을 만족하는 탐색), 탐색 기반 가치 추정(SVE), 혼합 가치 목표, 행동 임베딩과 우선순위 사전 계산(부록 A).
- **기존 것.** Gumbel top-k와 Sequential Halving(Danihelka et al.), 범주형 가치·보상(EfficientZero), 자기지도 일관성 손실(SimSiam, Chen & He, 2021), 정책 재분석(MuZero).
- **새 조합의 성격.** 개별 부품은 대부분 기존 것이고, 기여는 연속 행동에 맞춘 탐색과 오래된 전이를 위한 가치 목표를 함께 묶은 데 있습니다. 이것은 분석자의 판단입니다.

## 4. 외부 자료 (이 분석에서 메타데이터만 확인)

- 공식 저장소, 기반 저장소, 기준선 저장소의 별, 라이선스, 마지막 push는 `04_code_verification.md` §1에 있습니다.
- 논문 자체는 arXiv non-exclusive 라이선스입니다(`README.md`).

## 5. 확인하지 못한 것

- 위 표의 "논문의 비교" 열은 EZ-V2 논문 본문과 표에서 읽은 것입니다. 각 기준선 논문의 원래 수치는 확인하지 않았습니다.
- 이 저장소의 다른 폴더 분석 결과를 EZ-V2 수치와 직접 비교하지는 않았습니다. 필요하면 `03_analysis.md`의 DMC 표와 해당 폴더의 표를 나란히 놓는 작업이 다음 단계입니다.

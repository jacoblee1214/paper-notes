# 02. 선행연구와 관련 저장소

## 1. 계보 (구성 요소별 출처)

```
DQN (Mnih et al., 2015) ── 리플레이 비율 0.25, 3층 CNN
 ├─ Rainbow (Hessel et al., 2018) ── 6개 구성요소 결합, n-step 3 기본값
 │   └─ DER (van Hasselt et al., 2019) ── Atari 100K용 하이퍼파라미터 조정, 리플레이 비율 1
 ├─ 샘플 효율 라인 (Atari 100K, Kaiser et al., 2020)
 │   ├─ DrQ (Kostrikov et al., 2020) ── 데이터 증강
 │   ├─ DrQ(ε) (Agarwal et al., 2021b) ── 하이퍼파라미터만 조정
 │   ├─ SPR (Schwarzer et al., 2021) ── BYOL 계열 자기지도 예측 손실
 │   └─ SR-SPR (D'Oro et al., 2023) ── 주기적 재설정(shrink-and-perturb)으로 리플레이 비율 16까지
 │       └─ BBF (이 논문, ICML 2023) ── 폭 4배 ResNet + 강한 재설정 + n-step 감쇠 + γ 증가 + NoisyNets 제거
 ├─ 재설정 근거 ── Nikishin et al. (2022), 초기 경험에 과적합 (primacy bias)
 └─ 모델 기반 샘플 효율 (비교 기준) ── MuZero Reanalyse (2021), EfficientZero (Ye et al., 2021), IRIS (Micheli et al., 2023), DreamerV2/V3 (Hafner et al.)
```

- **BBF의 직접 조상은 SR-SPR입니다.** 논문 §4 "Base agent"가 명시합니다.
- **재설정 계보.** Nikishin et al. (2022)은 마지막 층 재설정, D'Oro et al. (2023)은 합성곱 층의 shrink-and-perturb를 더했습니다. BBF는 섞는 비율을 0.2에서 0.5로 올렸습니다(§4 "Harder resets").
- **폭 확장 계보.** Espeholt et al. (2018)의 IMPALA ResNet을 쓰고, 폭을 늘리는 근거는 Cobbe et al. (2020)과 Farebrother et al. (2023)의 ResNet 특징 폭 확장 결과, 그리고 오프라인 RL의 확장 결과(Kumar et al., 2022)에서 옵니다(§3 Related Work).
- **모델 기반 쪽.** EfficientZero(MuZero 계열, 8.5시간, 512 CPU 코어 + GPU 4대)가 이 논문의 주된 비교 기준입니다. BBF는 이 논문에서 모델-프리 에이전트가 같은 샘플 효율에 도달했다는 점을 강조합니다.

## 2. 가까운 대안과 차이

| 비교 대상 | 같은 점 | BBF와 다른 점 | 논문에서 쓴 근거 |
|---|---|---|---|
| SR-SPR (D'Oro et al., 2023) | 같은 SPR 손실, 재설정, 리플레이 비율 | 폭 4배 ResNet, 강한 재설정(50%), n-step 감쇠, γ 증가, NoisyNets 제거, 리플레이 비율 8 | 그림 2, 3, 6 |
| SPR (Schwarzer et al., 2021) | 자기지도 손실 | 재설정 없음, 폭 1 | 그림 3, 12 |
| EfficientZero (Ye et al., 2021) | 자기지도 일관성 손실과 비슷 (§5), Atari 100K 성능 | 모델 기반(MCTS), 배치 8배 큼, 리플레이 비율 1.2 | 그림 2, 표 A.1 |
| IRIS (Micheli et al., 2023) | 같은 벤치마크 | 세계 모델(Transformer), 100K 기준 표 A.1에서 IQM 0.501 | 그림 2, 표 A.1 |
| DER, DrQ(ε) | 모델-프리 Atari 100K | 하이퍼파라미터만 조정 (논문 §3) | 표 A.1 |

## 4. 공개 구현 (GitHub 표)

| 저장소 | 관계 | 라이선스 | 규모·최근 활동 | 확인 커밋 | 상태 |
|---|---|---|---|---|---|
| [google-research/google-research](https://github.com/google-research/google-research/tree/master/bigger_better_faster) (하위 폴더 `bigger_better_faster/`) | 공식 구현 (JAX, Dopamine 기반), 점수 파일 포함 | Apache-2.0 (모노레포 API `spdx_id`). 하위 폴더에는 LICENSE 파일이 없음 | 모노레포 stars 39,245, 모노레포 push 2026-11-20 | 하위 폴더 최근 커밋 `5b09c22d73` (2026-02-12, API `commits?path=`). 얕은 복제 HEAD `9394dfc803d4` (2026-11-20) | `repo/gr/`에 희소 체크아웃 (정적 읽기만, 실행·설치 안 함) |
| 별도 프로젝트 페이지 | 없음 | — | — | — | 논문 PDF와 arXiv 초록 어디에도 프로젝트 페이지 링크가 없음. 공식 자료는 위 저장소 하나입니다 |

- stars는 모노레포 전체 값입니다. BBF 폴더만의 값은 아닙니다.
- 저장소에는 `scores/` 폴더가 있습니다. 25개 CSV는 게임 × 시드별 정규화 점수입니다(RR=2 기본과 절제, RR=8 기본과 절제, 스티키 액션, 학습 곡선). 이 파일들이 03의 재계산 원자료입니다.
- 이번 분석에서 찾지 못한 것: 기준선(SR-SPR, DrQ(ε), DER, IRIS, EfficientZero)의 CSV. SR-SPR은 `configs/SR_SPR.gin`만 있습니다.

## 5. 이 분석에서 중요한 선행연구 질문

1. **재설정과 리플레이 비율의 관계.** BBF 그림 6의 "BBF − SR-SPR ≈ 0.45 (모든 RR)"가 선형이라는 주장은 설명되지 않았습니다. 재설정 계열 연구(Nikishin 2022, D'Oro 2023)를 함께 읽어야 합니다.
2. **n-step 감쇠의 이론적 근거.** Kearns & Singh (2000)의 최적 감쇠 n 결과를 BBF가 그대로 적용한다고 보는데, 이 이론이 함수 근사와 재설정이 있는 설정까지 성립하는지는 논문에서 확인하지 않았습니다.
3. **벤치마크 포화.** Atari 100K의 26게임, 스티키 액션 없음, 10만 스텝은 BBF가 이미 넘긴 기준입니다. Kaiser et al. (2020)의 원래 목표와 맞는지는 논문 §6–7이 논의합니다.

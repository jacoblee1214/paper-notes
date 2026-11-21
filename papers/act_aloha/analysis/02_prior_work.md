# 02. 선행연구와 관련 연구

*기준:* 논문의 참고문헌(번호는 논문 기준)과 공식 저장소 기록. 논문 밖의 후속 연구는 arXiv 초록 페이지나 GitHub에서 제목과 날짜를 확인한 것만 적었습니다.

## 1. 계보

```
BC (Pomerleau; [44])  →  구조 개선: BC-Z [26], BeT [49], RT-1 [7]
                      →  비모수: VINN [42]
                      →  복합 오차: DAgger [47], DART [36], HG-DAgger [30]
                      →  행동 청크(행동 조각을 한 번에 예측): 심리학 개념 [35] (Lai et al., 2022, 행동 청크 = 정책 압축)
                      →  트랜스포머 + CVAE: DETR [8] + CVAE [55] + β-VAE [23]
                      ⇒ ACT (이 논문)
   ↓ 후속 (논문 밖, 확인한 것만)
   Mobile ALOHA (arXiv:2401.02117, 2024-01-04, "Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation"): 이동 기지 위 양팔 확장 (제목과 날짜만 확인)
   act-plus-plus (MarkFzp, 저장소 이름만 확인): ACT 구현 기반 저장소 (내용은 대조하지 않음)
```

## 2. 구성 요소별 출처와 가까운 대안

| 요소 | 논문에서 가져온 것 | 참고문헌 | 가까운 대안 (논문의 비교) |
|---|---|---|---|
| 행동 청크 | 한 번에 k개 행동 예측 | 행동 청크 개념 [35]; 시간 상관 교란 [61] | BeT (단일 스텝), RT-1 (단일 스텝) |
| 시간 앙상블 | 겹치는 청크 지수 가중 평균 (추론) | 이 논문의 제안 | 없음 (비교 없음; VINN에는 −) |
| CVAE 학습 | 인코더는 학습 때만, 디코더가 정책 | 조건부 VAE [55]; VAE [33]; β-VAE [23] | 없음 (ablation: CVAE 제거) |
| 트랜스포머 구조 | 인코더–디코더, 질의 k개 | Transformer [65]; DETR [8] | BeT [49] (트랜스포머, 이산 행동) |
| 시각 백본 | ResNet18 4대 카메라 | ResNet [22] | BC-ConvMLP [69, 26] |
| 연속 행동 출력 | L1 회귀 (이산화 안 함) | 이 논문 (근거: 정밀도) | BeT, RT-1 (이산화) |
| 리더–팔로워 매핑 | 관절 공간 1:1 | Kim et al. [32] (바나나 껍질 벗기기) | VR 과제 공간 매핑 (Holo-Dex [6] 등) |
| 시뮬레이션 과제 | 이 논문이 만든 Transfer Cube, Insertion | MuJoCo [63] | 없음 |

## 4. 코드와 프로젝트 페이지 (2026-11-21 조회)

| 구분 | 저장소 / 페이지 | 라이선스 | 스타 | 마지막 push | 커밋 (복제 여부) | 역할 |
|---|---|---|---|---|---|---|
| 공식 코드 (ACT) | [tonyzhaozh/act](https://github.com/tonyzhaozh/act) | MIT | 2,352 | 2024-07-23 | `742c753` (얕은 복제, 정적 읽기) | 정책 학습, 시뮬레이션, 평가 |
| 공식 코드 (ALOHA) | [tonyzhaozh/aloha](https://github.com/tonyzhaozh/aloha) | MIT | 2,371 | 2024-04-19 | `06369f0` (얕은 복제, 정적 읽기) | 실로봇 수집·제어 (ROS 1 Noetic) |
| 프로젝트 페이지 | [tonyzhaozh.github.io/aloha](https://tonyzhaozh.github.io/aloha/) | — | — | — | — | 소개 페이지. "RSS 2023"이라고 씀 |
| 후속 (ACT 기반) | [MarkFzp/mobile-aloha](https://github.com/MarkFzp/mobile-aloha) | MIT | 4,481 | 2024-06-22 | 복제 안 함 | 이동 기지 확장 (Mobile ALOHA, arXiv:2401.02117) |
| 후속 (ACT 확장) | [MarkFzp/act-plus-plus](https://github.com/MarkFzp/act-plus-plus) | MIT | 3,677 | 2024-05-15 | 복제 안 함 | ACT 개선 구현 |

- 두 공식 저장소 모두 2024년 이후 커밋이 없습니다. 논문 v1 이후 코드 개정이 있었는지는 확인하지 못했습니다. 두 복제본은 저장소에 넣지 않았습니다.
- 공식 저장소 ACT의 `README.md`에는 논문 정식 제목과 RSS 2023 인용이 있는지 이 분석에서 확인하지 않았습니다.

## 5. 이 논문을 읽을 때 기억할 점

- 논문의 "기준선"은 행동 복제 계열 (BC-ConvMLP, BeT, RT-1, VINN)입니다. 참고문헌 목록에 확산 정책(diffusion policy) 계열이 없으므로 이 논문은 그 계열과 비교하지 않습니다. 이 분석에서 그 비교가 필요하다는 주장은 하지 않습니다.
- 행동 청크 (chunking)는 이 논문에서 새로운 개념이 아니라 결합 방식의 제안입니다. 기여는 청크 + 시간 앙상블 + CVAE 조합과 저가 장비입니다.

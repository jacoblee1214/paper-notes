# 02. 선행연구와 자료: RoboTTT (arXiv:2607.15275)

분석일 2026-11-24. 참고문헌 번호는 논문 v1의 번호입니다.

## 1. 계보와 구성 요소별 출처

| 구성 요소 | 출처 (논문 참고문헌) | RoboTTT에서 한 일 |
|---|---|---|
| TTT 층: 은닉 상태를 작은 모델의 가중치로, 갱신을 경사 한 걸음으로 | Sun 외 2024 "Learning to (learn at test time)" [64], 원조 TTT [63], 선형 연상 메모리와의 관계 Schlag 외 2021 [58] | 식 1–2를 그대로 사용. 입력 토큰을 시간 축으로 처리 |
| 큰 청크·비선형 고속 모델·학습 안정화 | "Test-time training done right" [78], 비디오 TTT [13], End-to-end TTT [65], TNT [42] | 2층 MLP, 학습 가능한 내부 학습률, 기본 학습률 0.1(부록 A.1). TNT는 후속 과제로 남김(§6) |
| 선형 순환 기준선 | Gated DeltaNet [74], FLA 라이브러리 [73] | TTT 층을 GDN 층으로 바꾼 기준선 (부록 B.2) |
| 시퀀스 행동 강제 | Diffusion Forcing [10] | 청크마다 독립 노이즈 수준을 뽑음(식 5). 이 설정에서 필요하다는 주장은 그림 12의 한 점(≈0.01)이 근거 |
| TBPTT | 고전적 기법 | 고속 가중치를 구간 사이에 이월하고 경사만 끊음(그림 4) |
| 게이팅 | Flamingo의 tanh 게이트 [1] | 식 3, α를 0.001로 초기화 |
| 레지스터 토큰 | ViT registers [14, 30] | 시간 축으로 VL 정보를 운반(16개) |
| 백본 | GR00T N1/N1.7 [51], Eagle [59], DiT [52] | 모든 DiT 층에 TTT 삽입 |
| 사전학습 데이터 | EgoScale [81] | 에고센트릭 사람 영상 혼합(§3.4, 부록 A.2) |
| 일회 모방 | One-shot imitation learning [18], ICRT [22] | 사람 영상을 문맥으로 두고 손실을 가림(§3.3) |
| DAgger Distillation | DAgger [57], Algorithm Distillation [39] | 실패 구간은 문맥, 교정 구간은 손실 대상 |
| 이력의 허위 상관 | Causal confusion [15], copycat [70], 과거 토큰 예측 [68] | 본문은 TTT가 "중복을 지운다"고 설명하나 직접 측정은 없음(§5) |

## 2. 공개 자료 표

2026-11-24 기준. 별 수와 마지막 push는 `gh api`로 읽었습니다.

| 자료 | 주소 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| 논문 | https://arxiv.org/abs/2607.15275 | CC BY 4.0 (abs 페이지), PDF에 "© 2026 NVIDIA. All rights reserved." 표기 | | 2026-07-16 제출, v1 | |
| 프로젝트 페이지 | https://research.nvidia.com/labs/gear/robottt/ | 미표기 | | | 영상, 방법 요약, BibTeX. 코드 링크 없음 |
| **공식 코드** | **찾지 못함** | | | | 논문 감사의 글에 "help with the release"가 있으나 저장소 링크 없음. NVIDIA/Isaac-GR00T 안에서도 TTT 관련 코드는 검색되지 않음 |
| GR00T N1.7 (기반 모델) | https://github.com/NVIDIA/Isaac-GR00T | 코드 Apache-2.0, 가중치 NVIDIA Open Model License | 8,414 | 2026-11-17 (HEAD 708617f) | 설정 파일에서 DiT 16층, 헤드 32 x 48, action_horizon 40, 디노이징 4단계, VLM Cosmos-Reason2-2B를 읽음 |
| 비공식 구현 1: robo_ttt | https://github.com/lucidrains/robo_ttt | MIT | 129 | 2026-09-17 (98eb883, 생성 2026-07-27) | TTT 래퍼, 학습 가능 학습률, 망각 게이트, 선택적 Muon, 다음 잠재 예측기 포함(논문에 없는 부분 있음). 정적으로만 읽음 |
| 비공식 구현 2: lerobot-pi0-ttt | https://github.com/Bala-la-la/lerobot-pi0-ttt | Apache-2.0 | 14 | 2026-07-19 (생성 2026-07-18) | LeRobot π0에 고속 가중치와 TBPTT를 붙임. README 수준만 확인 |
| FLA (GDN 구현) | https://github.com/fla-org/flash-linear-attention | MIT | 6,080 | 2026-11-24 | 기준선의 GDN 층 출처(부록 B.2) |
| TTT 원 구현(언어) | https://github.com/test-time-training/ttt-lm-pytorch | MIT | 1,411 | 2024-07-14 | |
| TTT 비디오 DiT | https://github.com/test-time-training/ttt-video-dit | MIT | 2,453 | 2026-02-25 | |
| LaCT | https://github.com/a1600012888/LaCT | MIT | 542 | 2026-01-05 | 큰 청크 TTT(논문 [78]의 구현) |

공식 코드가 없어서 이 저장소에서는 코드 대조(D 항목) 대신 주장 검증(`04`의 C 항목)을 했습니다. 비공식 구현은 논문과의 일치 여부를 판정하는 근거로 쓰지 않았습니다(작성자가 논문 저자가 아니고, 논문에 없는 요소를 포함합니다).

## 4. 가까운 대안과 차이

| 대안 | 방식 | RoboTTT와 차이 |
|---|---|---|
| 이력 프레임 몇 개를 붙이는 정책 | 2–8 프레임 [7, 8, 41, 47, 67, 75] | 고정 창. 논문 기준선 Hist.는 1프레임 |
| 키프레임·언어 요약 [50, 61, 69] | 이력을 의미 수준으로 압축 | 시각 운동 문맥 대신 상위 표현 |
| 전체 이력을 자기회귀로 처리 [6, 22, 32, 40, 56] | KV 캐시가 이력에 비례해 커짐(§5) | RoboTTT는 상태 크기가 고정 |
| 게이트 메모리 정책 [23], 과거 토큰 예측 [68], ContextVLA [31] | 이력 토큰을 거르거나 압축 | 학습된 고속 가중치로 대체 |
| 선형 순환(GDN, Mamba류) | 선형 연상 갱신 | 논문의 직접 기준선. 선형 갱신 대 경사 하강 갱신이 핵심 변인 |
| 로봇 "test-time training" [2, 45, 82] | 시험 과제에서 추가 데이터를 모아 전체 미세조정 | 고속 가중치 없이 파라미터를 갱신(§5가 구분) |
| VITA [83] | VLM에 고속 가중치를 달아 가치 함수를 적응 | 가장 가까운 설정이나 정책이 아니라 가치 함수 |

## 5. 읽으면서 확인할 것 (선행연구 관점)

- GDN 기준선은 "층 위치, 게이팅, 파라미터 수를 맞춘" 것이지만(부록 B.2) GDN 하이퍼파라미터(상태 크기, 헤드 수)와 사전학습 학습률 탐색 여부는 쓰여 있지 않습니다. 논문이 주장하는 차이의 원인(선형 대 비선형 갱신)이 이 기준선의 튜닝 정도와 분리되는지는 본문만으로 판단할 수 없습니다.
- 설명 가설 둘(GDN은 메타 학습이 안 된다, 선형 연상 갱신은 조밀한 스트림에서 구조를 못 뽑는다)은 §4에서 가설로 표시되어 있고 독립 실험은 없습니다.

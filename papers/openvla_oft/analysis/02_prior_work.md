# 02. 선행연구, 구성 요소의 출처, 공식 저장소

분석일 2026-11-22. 각 구성 요소의 출처는 논문 참고문헌 번호(v2)로 적습니다.

## 1. 계보

1. **VLM에서 VLA로.** RT-2 [4]는 VLM을 로봇 행동 토큰으로 미세조정했습니다. OpenVLA [23]는 Prismatic-7B를 Open X-Embodiment로 학습해 공개했습니다. 이 논문의 기반 모델입니다.
2. **OpenVLA의 미세조정 연구.** Kim 등 [23]은 LoRA로 단일 팔 로봇에 적응시키는 방법을 연구했습니다. 자기회귀 생성이 3–5 Hz라서 고주파 제어에는 맞지 않습니다(논문 I절).
3. **토큰화 개선.** MiniVLA [2]는 벡터 양자화로, FAST [38]는 DCT 압축으로 행동 토큰 수를 줄였습니다. 논문은 이들이 자기회귀 방식이라 2–13배 속도 향상에 그친다고 봅니다. FAST의 750 ms 청크 지연도 인용합니다.
4. **양손 조작용 VLA.** RDT-1B [27]는 확산 기반, π0 [3]는 흐름 정합 기반입니다. 둘 다 양손 데이터로 사전학습되었고, 논문은 이들을 미세조정 비교 기준으로 씁니다.
5. **모방 학습 기준선.** Diffusion Policy [5], ACT [56] (L1 회귀 + 청킹), MDT [40], Octo [49], DiT Policy [13], Seer [50]. 논문 표 I의 비교 기준입니다.
6. **이 논문의 위치.** VLA 모델을 새로 만들지 않고, 미세조정 레시피의 설계 공간을 실험으로 정리했습니다. 기존 방법들의 요소(청킹 [56, 5], L1 회귀 [56], 확산 [5], 병렬 디코딩 [56])를 하나의 레시피로 묶은 것입니다.

## 2. 구성 요소별 출처

| 구성 요소 | 논문에서 쓴 출처 | 메모 |
|---|---|---|
| 청킹 (행동 여러 단계를 한 번에) | ACT [56], Diffusion Policy [5], [28] | 논문은 청킹이 "성공률을 올린다"고 주장하고 표 I에서 검증합니다 |
| 병렬 디코딩 (빈 임베딩, 양방향 어텐션) | ACT [56] (빈 임베딩과 위치 인코딩) | 양방향 어텐션은 이 논문에서 OpenVLA 쪽에 적용. 구현은 외부 fork |
| L1 회귀 헤드 | ACT [56] | 확산 대신 L1을 고르는 근거는 학습 수렴 속도와 추론 속도 |
| 조건부 확산 (DDIM, squared cosine) | Chi 등 [5], Zhao 등 [57] (ALOHA Unleashed) | 50단계가 기본. 단계를 줄이면 표 II에서 성능이 떨어짐 |
| FiLM | Perez 등 [37] (AAAI 2018), 언어 조건 EfficientNet에 적용 [43] | 채널 단위 적용이 패치 단위보다 언어 따르기가 좋다고 보고 |
| LoRA | Hu 등 [14] | 학습 설정은 OpenVLA [23]의 설정을 따름 |
| 다중 시점과 자세 입력 | 이 논문 | 양방향 어텐션 덕에 입력 토큰이 늘어도 지연이 크게 늘지 않음 |

## 4. 공식 저장소와 관련 자료

| 저장소 또는 자료 | 관계 | 라이선스 | 기준 (조회 2026-11-22) | 상태 |
|---|---|---|---|---|
| [moojink/openvla-oft](https://github.com/moojink/openvla-oft) | 공식 학습, 미세조정, LIBERO와 ALOHA 평가, 추론 | MIT | 스타 1,450, 포크 214, 마지막 push 2025-09-09, 커밋 `e4287e9` (2025-09-08) | 얕은 복제 `repo/`, 정적 읽기만 (`04`) |
| [openvla-oft.github.io](https://openvla-oft.github.io/) | 프로젝트 페이지 (영상, 추가 실험 설명) | 명시 없음 | HTTP 200 | 읽음 (보조) |
| [moojink/transformers-openvla-oft](https://github.com/moojink/transformers-openvla-oft) | 양방향 어텐션을 위한 transformers fork (v4.40.1 계열) | Apache-2.0 | 스타 12, 마지막 push 2025-02-25 | 받지 않음. 재현에 필수 (`04` D2) |
| [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | LIBERO 시뮬레이터와 과제 정의 | MIT | 스타 2,563, 마지막 push 2025-03-15 | 받지 않음 |
| [moojink/dlimp_openvla](https://github.com/moojink/dlimp_openvla) | RLDS 데이터 로더 fork | 표시 없음 | 스타 7, 마지막 push 2024-07-12 | 받지 않음. 라이선스 확인 필요 |
| [moojink/openvla-7b-oft-finetuned-libero-*](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial) | 공식 LIBERO 체크포인트 (스위트별 네 개와 합동 한 개) | MIT (카드) | 저장 용량 약 15.9 GB (spatial 기준) | 받지 않음 |
| [openvla/openvla](https://github.com/openvla/openvla) | 기반 OpenVLA 코드 | MIT | 스타 7,329, 마지막 push 2025-03-23 | 공식 코드 아님 (기반) |
| [openvla/openvla-7b](https://huggingface.co/openvla/openvla-7b) | 기반 OpenVLA 가중치 | MIT (카드). 기반 모델의 Llama 2 조건 확인 필요 | 저장 용량 약 15.1 GB | 받지 않음 |
| 데이터셋 `openvla/modified_libero_rlds` | 필터링된 LIBERO 데이터 (LIBERO.md에 링크) | 확인 안 함 | — | 받지 않음 (용량 확인 안 함) |

공식 코드가 있으므로 `04_code_verification.md`(논문 ↔ 코드 대조, D1–D21)를 만들었습니다.

## 5. 주장 간의 관계

- **속도와 성능을 함께 주장하는 근거:** 표 I (성능)과 표 II (속도)가 같은 설정(A100, 224 px, K=8)을 씁니다. 두 표를 한 번에 읽을 수 있습니다.
- **연속 행동 대 이산 행동:** 표 I의 PD&AC 이산(90.2)과 연속(95.3)의 차이 5.1%p가 근거입니다. 같은 기반 모델의 같은 설정에서 나온 비교입니다.
- **ALOHA의 기준선:** ACT, Diffusion Policy, RDT-1B, π0는 원 논문 설정이 아니라 이 논문이 과제별로 다시 학습한 것입니다. 비교는 같은 과제 세트 안에서만 유효합니다.

# 02. 선행연구와 공식 저장소

## 1. 계보

| 계보 | 대표 연구 | TBD-VLA에서의 위치 |
|---|---|---|
| 연속 행동 전문가 VLA | π0.5(flow matching 헤드), GR00T N1.5, SmolVLA | 실로봇·LIBERO의 주된 기준선. VLM과 행동 생성이 분리됨(서론) |
| 한 토큰씩 자기회귀 | OpenVLA, MolmoAct, VLA-0 | 시간 자기회귀는 있으나 토큰 단위라 느림(표 1: 0.34–5.6 s) |
| 압축·학습된 행동 토큰 | FAST(π0-FAST), VQ-VLA, OAT | 토큰 수를 줄이지만 토큰과 시간 스텝의 대응이 약해짐(§2) |
| 병렬 디코딩 | OpenVLA-OFT(완전 병렬 회귀) | 가장 빠름(0.031 s). 시간 자기회귀 없음 |
| 이산 확산 VLA | Discrete Diffusion VLA, dVLA(Fast-dVLA), UD-VLA, LLaDA-VLA | 블록 없이 청크 전체를 확산. TBD-VLA는 m = H_p인 극단에 해당(§5.5) |
| 블록 확산(언어) | Block Diffusion(BD3-LM, Arriola et al., ICLR 2025), Fast-dLLM v2 | 블록 확산과 이중 배치 마스크의 출처. 행동 토큰으로 옮김 |
| 비동기 실행 | Real-Time Chunking(Black et al.) | flow 정책의 in-painting을 마스크 이산 확산의 "블록 완성"으로 대체 |

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 이 논문의 변경 |
|---|---|---|
| 블록 이산 확산, 이중 배치(깨끗한 열 + 오염 열, 위치 공유) | BD3-LM, Fast-dLLM v2 | 토큰 열을 시간 블록으로 해석, 앵커 블록과 시간 단위 이동 추가 |
| 마스크 확산 손실 | MDLM·LLaDA 계열 | 코드에서 블록당 하나의 마스크 비율(`04` D2) |
| Qwen3-VL 2B 백본 | Qwen3-VL | 어휘에 512 bin + 특수 토큰 추가 |
| 기대값 디코딩 | 이산 행동 분포의 평균 | 코드는 기대 bin을 반올림한 뒤 그리퍼는 argmax(`04` D4) |
| RTC | Black et al.(flow 정책) | hard in-painting을 학습된 블록 완성에 맞춤 |
| 사전학습 혼합 | DROID, BC-Z, RoboMIND, RH20T, RoboSet, MolmoAct, OXE 일부 | 표 7: 160,268 에피소드 |
| 구현 틀 | LeRobot | 공식 코드가 LeRobot 포크 |

## 4. GitHub와 공개 자료

| 자료 | 관계 | 라이선스 | 별 / 마지막 push | 대조 기준 | 상태 |
|---|---|---|---|---|---|
| [TBD-VLA/lerobot](https://github.com/TBD-VLA/lerobot) | 공식 구현(LeRobot 포크, `src/lerobot/policies/tbdvla/`) | Apache-2.0 | 7 / 2026-10-17 | `41c746ed487520cf6cc49eb767d090837691e2b2` (main, 2026-10-16) | 얕은 clone, 정적 읽기만(`repo/`) |
| [TBD-VLA/tbd-vla.github.io](https://github.com/TBD-VLA/tbd-vla.github.io) | 프로젝트 페이지 저장소 | MIT | 0 / 2026-11-04 | — | HTML만 읽음 |
| [tbd-vla.github.io](https://tbd-vla.github.io/) | 프로젝트 페이지 | — | — | — | 읽음(갱신된 지연 수치와 LIBERO-Plus 수치표가 있음) |
| [sean1295/tbdvla_libero](https://huggingface.co/sean1295/tbdvla_libero) | LIBERO 체크포인트(BF16, 2,128,039,936 파라미터) | apache-2.0 | 다운로드 987 / 모델 파일 2026-06-07 이후 README만 갱신 | — | 받지 않음, 메타데이터와 config.json만 읽음 |
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | 상위 저장소 | Apache-2.0 | 29,102 / 2026-11-26 | — | 사용 안 함 |
| [kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms) | BD3-LM 원 구현 | Apache-2.0 | 1,043 / 2025-07-10 | — | 참고만 |
| [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) | Fast-dLLM v2 | Apache-2.0 | 1,117 / 2026-05-30 | — | 참고만 |
| [sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) | LIBERO-Plus 벤치마크 | 없음(API) | 504 / 2026-01-21 | — | 참고만 |

- 실로봇 데이터, 사전학습 체크포인트, 실로봇 배포 코드, 지연 시뮬레이션 환경은 저장소에서 확인되지 않았습니다(`04` D9, D10).
- 저장소 이력(`tbdvla/` 경로): 2026-06-06 최초 추가(eec16473) → 06-09 인용 갱신(cf589001) → 08-10 추론 마스크 수정(4f92f86d) → 10-15 추론 구조 재작성과 학습 마스크 변경 → 10-17 최적화. arXiv 게시일은 06-05입니다.

# 02. 선행연구와 관련 저장소

## 1. 선행연구 계보 (구성 요소별 출처)

| 구성 요소 | WoVR에서 쓴 것 | 출처 (논문 참고문헌 번호) |
|---|---|---|
| 기본 VLA | OpenVLA-OFT (LIBERO 기준선, 한 궤적 SFT와 전체 궤적 SFT), π0.5 (실로봇 일반화) | [1] OFT (RSS 2025), [3] π0.5 (CoRL 2025) |
| 온라인 RL | GRPO (기준선, 실제 환경 2,500 궤적) | [16] DeepSeek-R1 GRPO |
| 흐름 기반 VLA RL | πRL의 Flow-SDE로 π0.5 학습 | [8] πRL (arXiv:2510.25889) |
| 세계모델로 정책 최적화 | WMPO (직접 비교 기준선, OpenSora 사용) | [23] WMPO (ICLR 2026) |
| 세계모델 + RL (선행 저자) | World4RL, 같은 제1저자·연구 그룹 | [31] World4RL (RA-L 2026) |
| 세계모델 + RL (기타) | Prophet (Prophesying), World-Env, VLA-RFT | [40], [41], [24] |
| 세계모델 (비교 대상) | EVAC, Cosmos-Predict2, OpenSora | [30], [49], [21] |
| 세계모델 백본 | Wan2.2-TI2V-5B | [22] Wan (arXiv:2503.20314) |
| 정책 평가용 세계모델 (관련) | WorldGym, 보상 모델로 쓰는 세계모델 | [29], [32–36] |
| 벤치마크 | LIBERO | [51] |
| 세계모델 평가 지표 | LPIPS, FID, FVD, FloLPIPS | [45–48] |
| 회전 표현 | 6D 회전 | [56] Zhou 외 (회전 표현의 연속성), [55] X-VLA (같은 행동 표현 사용) |
| 손실 | Rectified Flow | [52] Esser 외 (rectified flow 기반 확산 모델) |
| 실로봇 보상 분류기 | HIL-SERL (희소 보상) | [18] |
| 밀집 보상 백본 | Qwen3-VL (LoRA) | [54] |

**계보에서 볼 점.** WoVR의 차별점은 "세계모델을 시뮬레이터로 쓴다"가 아니라, WMPO와 같은 설정에서 **가짜 성공을 줄이는 장치 세 개**(세계모델 안정화, KIR, PACE)를 더한 것입니다. 비교 대상 WMPO는 OpenSora(약 1.3B)를 쓰고 WoVR는 Wan 5B를 씁니다. 백본 차이가 얼마나 기여하는지는 논문이 분리하지 않았습니다(세계모델 비교 표 1은 백본을 바꾼 것이지 WoVR 장치를 뺀 것이 아님).

## 2. 코드와 저장소 (GitHub 표)

GitHub API로 2026-11-20 확인했습니다. 저장소 메타는 `gh api repos/<repo>`로 조회했습니다.

| 저장소 | 관계 | 라이선스 (API) | 별 | 마지막 push | 복제 커밋 | 상태 |
|---|---|---|---|---|---|---|
| [RLinf/RLinf](https://github.com/RLinf/RLinf) | **공식 코드** (프로젝트 페이지의 "Policy Training Code", "Code" 버튼). `rlinf/envs/sim/world_model/`, `examples/embodiment/config/wan_libero_*` | Apache-2.0 | 5,679 | 2026-11-19 | `a8a02f3` (2026-11-19) | 얕은 복제 (`repo/RLinf/`), 정적 읽기만 |
| [youmo445/diffsynth-studio-rlinf](https://github.com/youmo445/diffsynth-studio-rlinf) | **공식 세계모델 학습 코드** (프로젝트 페이지의 "WM Training Code"). DiffSynth-Studio 포크, Wan2.2-TI2V-5B 학습 | Apache-2.0 | 16 | 2026-04-07 | `6d9ce59` (2026-04-07) | 얕은 복제 (`repo/diffsynth-studio-rlinf/`), 정적 읽기만. 저장소 소유자가 개인 계정 |
| [RLinf/rlinf-wovr (HF 컬렉션)](https://huggingface.co/collections/RLinf/rlinf-wovr) | 가중치와 초기화 데이터 (`RLinf-Wan-LIBERO-{Spatial,Object,Goal}`, 궤적 1,500개씩) | 확인 안 함 | - | - | - | 다운로드하지 않음 (승인 범위 밖) |
| [wovr-rlinf.github.io](https://wovr-rlinf.github.io/) | 프로젝트 페이지 (영상, 요약) | - | - | - | - | HTML만 읽음 |
| [World4RL/World4RL.github.io](https://github.com/World4RL/World4RL.github.io) | 프로젝트 페이지 템플릿 출처(World4RL 논문 페이지). WoVR 코드 아님 | 없음 | 0 | 2026-06-18 | - | 복제하지 않음 |
| RLinf 문서: `docs/.../publications/wovr.rst`, `examples/embodied/wan.rst` | 공식 문서 (LIBERO 결과 표 포함, v1 수치) | Apache-2.0 (저장소) | - | - | (위와 같은 저장소) | 정적 읽기 |
| RLinf 블로그: `docs/.../blog/musa_wovr.rst` | 외부 재현 보고 (Moore Threads S5000 포팅, A100과 비교 상관 0.976, 190스텝 LIBERO 74.6% 대 73.4%) | (위와 같음) | - | - | (위와 같음) | 정적 읽기 |

**공식 코드 여부.** 프로젝트 페이지 버튼에 "Policy Training Code", "WM Training Code", "Code"가 있고, 논문 본문에는 링크가 없습니다. 두 저장소가 공식 코드입니다. 버튼 연결은 페이지 HTML의 버튼 문구 순서로 확인했습니다(정확한 버튼-링크 쌍은 브라우저에서 한 번 더 보면 확실합니다).

**재현 자원 주의.** 두 저장소 모두 학습 스크립트에 `/mnt/project_rlinf/...` 같은 저자 환경 절대 경로가 들어 있습니다(`04` D15). 실로봇 데이터·체크포인트 경로도 같습니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- WMPO (`[23]`): 같은 설정(WMPO가 OpenSora로 상상 RL)의 기준선. WoVR가 이기는 폭의 대부분이 이 비교에서 나옴(LIBERO 평균 +18.6, `03` §1.2).
- OFT (`[1]`): 기본 정책의 수치가 버전마다 바뀜(`01` §0). 기본 정책 재현 여부가 WoVR 개선 폭의 해석을 바꿈.
- Wan (`[22]`): 세계모델 백본. 5 denoising 스텝 추론 속도 주장은 이 백본과 RLinf 구현에 달려 있음.

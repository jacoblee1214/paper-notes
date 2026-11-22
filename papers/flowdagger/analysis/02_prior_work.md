# 02. 선행연구, 공식 저장소, 관련 분석

## 1. 선행연구 계보

FlowDAgger는 네 갈래를 붙입니다. (a) 생성 정책의 잡음 입력을 적응 표면으로 쓰는 잠재 공간 조종, (b) 사람 개입 기반 상호작용 모방학습, (c) 이미지 생성의 역변환(inversion), (d) 흐름 정합과 확산 기반 기본 정책. 새로움은 (b)를 (a)의 지도학습 신호로 바꾸는 연결부인 **행동 역변환**입니다(Key idea, §1).

### 1.1 구성 요소별 출처

| 구성 요소 | 출처 | 논문 쪽 위치 | 비고 |
|---|---|---|---|
| 확산 행동 머리, 시각운동 정책 | Diffusion Policy [1] (RSS 2023) | §1, §2 | 같은 계열의 흐름 정합은 [16] Lipman 외 |
| 흐름 정합 VLA | π0.5 [2], GR00T N1 [3] | §5.1 | 기본 정책 |
| 세계 행동 모델 | Cosmos Policy [4], World Action Models are Zero-Shot Policies [5], mimic-video [17] | §4.2 | 행동이 결합 잠재의 조각 |
| 상호작용 모방 | DAgger [9], HG-DAgger [10], IWR [11] | §2, §3 | 사람이 가로채는 방식 |
| 배포 중 사람 학습 | Sirius [18], Sirius-Fleet [19] | §2 | 신뢰도 가중 BC |
| 사람 개입 + RL | HIL-SERL [20], RLIF [21] | §2 | 보상 신호를 함께 씀 |
| 잔차 행동 | Residual policy learning [22, 23], Compliant Residual DAgger [12], TRANSIC [13] | §1, §5 기준선 | "Residual-DAgger" 기준선의 뿌리 |
| 잡음 공간 조종 | **DSRL [14] (CoRL 2025)** | §1, §2, §5 기준선 | 가장 가까운 방법적 이웃. 보상과 자율 탐색으로 학습 |
| 잠재 행동 | Latent plans from play [24] | §2 | 학습된 행동 공간 압축. FlowDAgger는 기본 정책이 이미 노출하는 잡음 공간을 씀 |
| 확산 역변환 | DDIM inversion [25], null-text inversion [26], 정류 흐름 역변환 [27] | §2, 부록 B.1 | 궤적 단위 고정점 기준선의 출처가 [27] |
| 평가 환경 | MetaWorld [28], BusyBox [29] (저자 일부 공동), robomimic [30], LIBERO | §5, 부록 B | 부록 B에서 LIFT와 LIBERO-90 |

### 1.2 논문이 인용하지 않은 가까운 연구

프로젝트 페이지는 "Concurrent Work"로 두 편을 소개합니다. 논문 PDF의 참고문헌에는 없습니다. 아래 요약은 두 논문의 arXiv abs 페이지 초록만 읽은 것입니다(원문 미열람).

| 논문 | arXiv(제출일) | 라이선스 표기 | 핵심 |
|---|---|---|---|
| UniSteer: Unified Noise Steering for Efficient Human-Guided VLA Adaptation | 2605.10821 (2026-05-11) | 비독점 배포 | 사람 교정 행동을 **단계별 고정점 역변환**으로 잡음 목표로 바꿔 지도학습하고, 잡음 공간 RL과 결합 |
| Improving Robotic Generalist Policies via Flow Reversal Steering (FRS) | 2606.13675 (2026-06-11) | CC BY 4.0 | 흐름을 거꾸로 통과시켜 거친 행동의 잡음을 찾고 가까운 행동 모드로 보냄. 잡음 공간 BC와 RL 부트스트랩 (DSRL 저자 Wagenmaker가 공저) |

둘 다 FlowDAgger v1(2026-07-09)보다 먼저 올라왔습니다. "행동을 잡음으로 되돌려 지도학습 신호로 쓴다"는 아이디어의 새로움 주장은 이 두 편과 비교해 읽어야 합니다. 특히 프로젝트 페이지가 UniSteer를 "per-step fixed-point inversion"으로 소개하는 대목은 이 논문의 §4.1 구성과 이름까지 겹칩니다(소개 문장 수준의 인상이며 UniSteer 원문의 세부는 확인하지 않았습니다).

### 1.3 가까운 대안과 차이

| 방법 | 감독 신호 | 갱신 대상 | FlowDAgger와의 차이 |
|---|---|---|---|
| SFT / LoRA-DAgger | 행동 | 기본 가중치 또는 저랭크 | 기본 정책을 건드림. Table 3에서 보류 과제가 붕괴 |
| Residual-DAgger | 행동 | 행동 공간 잔차 | 기본 정책의 지지 집합 밖으로 나갈 수 있음. 다만 Door Lock처럼 작은 일정 오프셋이면 더 낫다고 인정 |
| DSRL | 보상 | 잡음 정책 (SAC) | 같은 인터페이스. 신호가 희소 보상이라 50 롤아웃에서 거의 못 움직임 |
| 잡음 공간 BC (이 논문) | 사람 교정을 잡음으로 변환 | 잡음 정책 (회귀) | 역변환 오차가 곧 감독 잡음 |

## 2. 공식 저장소와 관련 자료 (2026-11-22 확인)

| 저장소·자료 | 관계 | 라이선스 | 별 | 최근 push | 상태 |
|---|---|---|---|---|---|
| [microsoft/FlowDAgger](https://github.com/microsoft/FlowDAgger) | **공식 코드** (논문 abs와 프로젝트 페이지의 Code 링크) | MIT (Copyright Microsoft Corporation) | 93 (포크 7, 열린 이슈 1) | 2026-10-23 (생성 2026-06-06) | 얕은 복제 `repo/` (HEAD `3426f24`, 2026-09-21 병합 커밋), 정적 읽기만 |
| [프로젝트 페이지](https://microsoft.github.io/FlowDAgger) | 공식 웹사이트 | 사이트 CC BY-SA 4.0 (페이지 하단 표기) | — | — | 열람. 표 1–4가 같은 수치로 다시 실려 있고, 기본 정책의 데모 수와 실로봇 영상 설명이 더 있음 |
| [mmurray/openpi](https://github.com/mmurray/openpi) | 공식 코드의 서브모듈 (`flowdagger_pi05/openpi`, 고정 커밋 `4c5df22`) | Apache-2.0 | 0 | 2026-04-29 | 포크. 복제하지 않음 (서브모듈 미초기화) |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 위 포크의 원본, π0.5 JAX 구현 | Apache-2.0 | 14,729 | 2026-08-24 | 메타데이터만 |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | GR00T N1.7 백엔드의 의존성 (README가 커밋 `9c7e746` 지정) | Apache-2.0 | 8,403 | 2026-11-17 | 메타데이터만 |
| [NVlabs/cosmos-policy](https://github.com/NVlabs/cosmos-policy) | 논문의 세계 행동 모델 기본 정책 | Apache-2.0 | 900 | 2026-01-23 | 메타데이터만. **FlowDAgger의 Cosmos 역변환 코드는 공식 저장소에 없음** |
| [HF mmurray-ms/pi05-metaworld](https://huggingface.co/mmurray-ms/pi05-metaworld) | 공식 코드가 자동으로 받는 π0.5 MetaWorld 체크포인트 | 카드에 라이선스 표기 없음 (카드 데이터 없음) | — | 2026-06-06 | 파일 목록 27개, 합계 약 12.4 GB. **받지 않음** |
| 제3자 복제·응용 | `nicknick416/FlowDAgger` (ARX 로봇 배포, MIT), `CPHughes23/EEE598_FlowDAgger_Replication`, `VatsalyaBhadaurya/FlowDAgger-VLA` 외 | 제각각 | 0 | 2026-07–11 | 열어보지 않음. 있다는 사실만 기록 |
| DSRL 구현 | 기준선 [14]의 공식 저장소 | — | — | — | GitHub 검색에서 찾지 못함 (확인 못 함) |

공식 저장소는 논문이 보고한 실험의 일부만 담습니다(`04`): π0.5 + MetaWorld assembly 한 과제, GR00T N1.7 + LIBERO-90 task 57. 기준선(SFT, LoRA-DAgger, Residual-DAgger, DSRL), Cosmos-Policy, 열두 과제, 실로봇은 없습니다.

## 4. 참고 문헌 중 핵심 항목

- [14] Wagenmaker 외, *Steering Your Diffusion Policy with Latent Space Reinforcement Learning*, CoRL 2025, arXiv:2506.15799. 이 논문의 전제와 기준선.
- [4] Kim 외, *Cosmos Policy*, arXiv:2601.16163. 세계 행동 모델 기본 정책.
- [12] Xu 외, *Compliant Residual DAgger*, NeurIPS 2025, arXiv:2506.16685. Residual-DAgger의 출처.
- [27] Rout 외, *Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations*, arXiv:2410.10792. 궤적 단위 고정점 역변환의 출처.
- [9] Ross, Gordon, Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, AISTATS 2011. DAgger.
- [29] Fortier 외, *Benchmarking Affordance Generalization with BusyBox*, arXiv:2602.05441. 저자 일부가 겹치는 평가 환경.

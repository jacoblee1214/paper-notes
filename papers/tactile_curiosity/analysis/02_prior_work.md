# 02. 선행연구와 관련 저장소 (TacEx)

번호는 논문(v3) 참고문헌 번호입니다.

## 1. 계보

**탐색 전략 (논문 §2.1, 부록 A)**
- 무작위 탐색: ε-greedy([9, 10, 11]), 결정적 정책 + 가우시안 잡음(DDPG [12], TD3 [13]), Boltzmann/엔트로피 정규화(SAC [16]).
- 호기심과 내재 동기: 예측 오차 기반 ICM [4], 세계 모델 계획으로 탐색하는 Plan2Explore [5], 앙상블 **불일치** 기반 [46](Pathak et al. 2019), 무작위 네트워크 증류 RND [39], 정보 이론 관점의 서베이 [3].
- 낙관적·정보 이득 탐색: OAX [6](Sukhija et al. NeurIPS 2023), SOMBRL [7], **MaxInfoRL [1]**(ICLR 2025, 이 논문의 직접 기반), 연속 시간 확장 [41], 시변 동역학 [42].
- 능동 정보 수집으로서의 촉각: 부분 모듈 목적의 촉각 위치추정 [24], 결정에 필요한 불확실성만 줄이는 [25](v3에서 추가).

**촉각을 탐색에 쓴 연구 (§2.3)**
- 누적 접촉력 임계 보상과 접촉 중심 재생 [26](Vulin et al., RA-L 2021), 충격 패널티 + 충격 예측기의 놀람 [27](Huang et al. 2019), 시각-촉각 예측 오차 + 시각 순방향 모델 오차 [28](Rajeswar et al., CoRL 2022, 합산 힘/토크를 보상과 상태에 사용).
- 이 논문의 위치: 모달리티별 **앙상블 불일치(인식론적 불확실성)** 를 가중하고, 합산 대신 공간 분해된 법선·전단 힘 지도를 씁니다(§2.3, 부록 A). 위 세 연구가 힘 크기, 스칼라 충격 예측기, 교차 모달 예측 오차(인식론과 우연적 불확실성을 섞을 수 있음)를 쓴다는 대비입니다.

**촉각 기반 조작 학습 (부록 A)**
- 촉각 예측 모델 기반 제어 [34](Tian et al. 2019), 삽입 [35], DexTouch [36], 입자 매질에서 물체 꺼내기 [37], 시각-촉각 파지 [38], 마스크 다중 모달 학습 [22](Sferrazza et al. IROS 2024, **이 논문의 힘 지도 시뮬레이션 환경 `tactile_envs`의 출처**), 촉각 시뮬 라이브러리 [23](TacSL).

**VLA와 촉각, 사후 학습 (부록 A)**
- 이질 센서 미세조정 FuSe [44](언어를 공통 공간으로 사용), Tactile-VLA [45](VLA 몸통에 촉각 입력을 넣고 모션과 접촉력 목표를 예측). 둘은 VLA를 **바꾸는** 방법이고, 이 논문은 VLA와 입력 모달리티를 그대로 두는 RL 사후 학습입니다.
- 확산 정책 사후 학습: DSRL [31](잠재 잡음 공간 RL, CoRL 2025), DPPO [33](확산 정책 자체에 정책 경사). π0 [43].

## 2. 가까운 대안과 구성 요소별 출처

| 구성 요소 | 출처 | TacEx에서의 역할 |
|---|---|---|
| 정보 이득 보너스 + 자동 온도 | MaxInfoRL [1] | 식 3의 골격. 코드는 별도 탐색 비평가와 λ 자동 조정까지 같은 구조(`04`) |
| 앙상블 동역학 모델과 불일치 | [46], [1] | 5개 머리 MLP |
| 힘 지도 촉각, 시뮬레이션 | `tactile_envs` [22], TacSL [23] | 촉각 관측. 무보상 탐색 환경 |
| 이미지 RL 백본 | DrQ [30], DrQ 계열 `jaxrl`/`jaxrl2` | 무보상 탐색의 SAC 변형, 영상 증강 |
| 확산 조종 | DSRL [31] (`nakamotoo/dsrl_pi0`) | 확산 잡음 공간의 MDP, 고정 π0 |
| VLA | π0 [43], openpi | 고정된 정책 |
| 벤치마크 | LIBERO [32] | 8개 접촉 과제 |
| 보상 형태 | DeepMind Control Suite [29] 의 tolerance | 하류 pick-and-place 보상 |
| 확산 정책 PPO 기준선 | DPPO [33] | 비교용 (저장소 코드는 "가장 작은 명시적 적응"이라고 설명) |

**가까운 대안(이 논문이 직접 겨냥하지 않은 것):**
- **보상 가산 접촉 보너스 + 일반 MaxInfoRL:** 촉각을 상태로만 넣는 DSRL-Tac-SAC와 촉각 목표 없는 MaxInfoRL 변형이 그림 5, 6의 비교 대상입니다.
- **접촉 이벤트 기반 보상 형성:** [26]의 힘 임계 보상은 합산 촉각 보상 기준선으로 다시 구현되었습니다(그림 6 "Reward"). 저장소 코드에서 이 기준선은 과제 보상 없이 합산 촉각만으로 학습합니다(`04` D12).
- **시각 불일치만 쓰는 탐색과 잠재 불일치만 쓰는 탐색:** 그림 2, 3의 V, L, VL.

## 4. GitHub 저장소 표

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 대조 기준 | 상태 |
|---|---|---|---|---|---|---|
| [klemensiten/tacex](https://github.com/klemensiten/tacex) | **공식 구현** (JAX, 탐색 + DSRL 조종) | 없음 (LICENSE 파일 없음, API license null) | 1 | 2026-10-15 | `d268cdb0d769fc9868fe37af462f81ef59c4412c` | 정적 읽기만(`repo/`) |
| [kiten.me/projects/tacex](https://kiten.me/projects/tacex/) | 프로젝트 페이지 (영상, 코드 링크, BibTeX) | — | — | — | — | HTML만 읽음 |
| [carlosferrazza/tactile_envs](https://github.com/carlosferrazza/tactile_envs) | 서브모듈. 힘 지도 시뮬 환경 | NOASSERTION (기타) | 106 | 2024-07-08 | 서브모듈 고정 `f131d499c34b782b3f5d3a9c5a293d53402733b9` | 트리 목록만 GitHub API로 확인 (`04` D2) |
| [nakamotoo/dsrl_pi0](https://github.com/nakamotoo/dsrl_pi0) | DSRL 공식 구현 (TacEx 조종 코드의 기반 계열, README 감사 문구) | MIT | 315 | 2026-04-27 | 받지 않음 | 메타데이터만 |
| [sukhijab/maxinforl_jax](https://github.com/sukhijab/maxinforl_jax) | MaxInfoRL JAX 구현 | NOASSERTION | 29 | 2026-01-08 | 받지 않음 | 메타데이터만 |
| [sukhijab/maxinforl_torch](https://github.com/sukhijab/maxinforl_torch) | MaxInfoRL PyTorch 구현 | NOASSERTION | 51 | 2025-09-18 | 받지 않음 | 메타데이터만 |
| [sukhijab/jaxrl](https://github.com/sukhijab/jaxrl) | `jaxrl` 포크 (`pip` 의존성, 특정 커밋 `268bbdc`) | MIT | 1 | 2024-12-14 | 받지 않음 | 메타데이터만 |
| [ikostrikov/jaxrl2](https://github.com/ikostrikov/jaxrl2) | 저장소 `jaxrl2` 패키지의 원본 | MIT | 59 | 2023-01-20 | 받지 않음 | 메타데이터만 |
| [nakamotoo/LIBERO](https://github.com/nakamotoo/LIBERO) | LIBERO 포크 서브모듈 (`87edbd1`) | MIT | 0 | 2025-04-29 | 받지 않음 | 메타데이터만 |
| [nakamotoo/openpi](https://github.com/nakamotoo/openpi) | openpi 포크 서브모듈 (`a6d2400`) | Apache-2.0 | 4 | 2025-08-05 | 받지 않음 | 메타데이터만 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0 구현과 체크포인트의 원본 | Apache-2.0 | 14,766 | 2026-08-24 | 받지 않음 | 메타데이터만 |

- 공식 저장소에는 라이선스가 없습니다. 저작권 기본 규칙이 적용되므로 코드를 공개본에 넣지 않았고, 정적으로 읽은 내용만 요약했습니다.
- 서브모듈 세 개(LIBERO, openpi, tactile_envs)는 `repo/`에서 비어 있습니다(`git clone`만 했고 서브모듈은 받지 않음). 고정 커밋의 파일 목록은 GitHub API의 트리 목록으로만 확인했습니다.
- 가중치와 데이터는 받지 않았습니다. 저장소 크기는 GitHub API 기준 약 1.1 MB입니다.

# 02. 선행연구와 관련 저장소

## 1. 계보

ArtManip는 세 갈래를 한 파이프라인에 묶습니다. 인용 번호는 논문의 참고문헌 번호입니다.

| 갈래 | 대표 선행 | ArtManip에서의 자리 |
|---|---|---|
| 강체 in-hand 조작 sim-to-real | Hora [3](RMA 잠재 적응), AnyRotate [5], 펜 스피닝 [7], PTLD [6], 손 안 재배향 [1, 2] | 교사-학생 잠재 증류(§3.4)는 [3, 28]을 따름. 학생 잠재가 정책을 구동하는 닫힌 루프 증류도 [3, 28]을 인용 |
| 병렬 RL 알고리즘 | SAPG [27], Isaac Gym [30] | 교사 학습(§3.3). PPO와 비교(Fig. 5) |
| 관절형 물체 조작 | GAPartNet [11], PartManip [12], FlowBot3D [13], Tac-Man [14], DexArt [18] 등 | 대부분 밑동이 고정된 환경 고정 물체. ArtManip는 자유롭게 떠 있는 물체를 손 안에서 다룸(§2.2) |
| 관절형 도구 in-hand | Atar 외 [9](촉각 힘 적응), Xu 외 [22](핀셋형 도구, 계층 RL), Yang 외 [21](지식 그래프로 손가락별 기능 배분) | "특정 인스턴스나 수동 지정 grasp 하나에 한정"으로 구분(§2.3) |
| 기능적 grasp 합성 | Lightning Grasp [26], 기능적 grasp [23–25] | [26]을 확장해 손가락별 허용·금지 접촉 영역을 넣음(§3.2). 코드 확인: 확장은 `func_lygra`에 있음(`04`) |

논문 자체의 선행 대비는 서술 위주입니다. 같은 지표로 다른 방법을 돌린 비교표는 없습니다(`03` K5).

## 2. 가까운 대안과 구성 요소별 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| 교사 PPO-LSTM + 특권 인코더 | Hora [3] | 코드도 Hora의 구조를 따른 것으로 보이며 README가 Hora에 감사를 표함(`04` §1) |
| SAPG | [27] | `rl_games` 포크에 들어 있음(MIT, 원 저장소 Denys88) |
| 에폭 기반 보상 가중치 커리큘럼 | AnyRotate [5] | 본문은 "[5]와 비슷하게 성공 사이클 수에 연동"이라 쓰지만 부록은 에폭 기반 선형 일정(§3.3, App. C.3) |
| 기능적 grasp 합성 | Lightning Grasp [26] | CUDA 전용. 확장판 저장소의 라이선스는 CC BY-NC(`04` §1) |
| primitive 물체 생성 | 이 논문(`make_data`) | 범위가 Tab. 6과 일치(`03` V12) |
| 학생 TCN | [3]의 시간 인코더 계열 | 코드는 `--custom_tcn` 대체 구현을 두고 RTX 50 계열에서 필요하다고 적음(`04` §3) |

## 4. GitHub와 프로젝트 페이지

| 저장소 | 관계 | 라이선스 | ★ | 마지막 push | 확인 커밋 | 상태 |
|---|---|---|---|---|---|---|
| [youngcv/artgym](https://github.com/youngcv/artgym) | **공식 코드**(README: "official implementation of ArtManip"). IsaacGym 환경, 학습, 증류, 평가, 실배포 | **없음**(루트에 LICENSE 없음, GitHub license: null). 포함된 `rl_games`는 MIT | 7 (포크 1) | 2026-09-14 | `63b94fb3364596db51b7e4651b3c3c98ff994710` (커밋 4개, 정적 읽기만) | 얕은 복제 `repo/` |
| [youngcv/make_data](https://github.com/youngcv/make_data) | 서브모듈. primitive 물체 생성과 접촉 영역 라벨 | 없음 | 0 | 2026-09-10 | `eafa7b6e84811de211757119081f86b3f236495f` | `repo/make_data/` |
| [youngcv/func_lygra](https://github.com/youngcv/func_lygra) | 서브모듈. 기능적 grasp 합성(Lightning Grasp 확장) | LICENSE 파일은 **CC BY-NC**(GitHub는 NOASSERTION/Other) | 1 | 2026-09-10 | `55f60c7e8f8db2ab55b50f6af5a07efcad191f57` | `repo/func_lygra/` |
| [artmanip.github.io](https://artmanip.github.io/) | 프로젝트 페이지(정적 HTML). 영상, 초록, BibTeX | 없음(저장소 [artmanip/artmanip.github.io](https://github.com/artmanip/artmanip.github.io) license: null) | 0 | 2026-09-14 | 해당 없음 | HTML만 읽음 |
| [zhaohengyin/lightning-grasp](https://github.com/zhaohengyin/lightning-grasp) | 상류. grasp 합성 | NOASSERTION(`func_lygra`의 LICENSE가 CC BY-NC라 상류에서 이어진 것으로 보임) | 231 | 2026-06-12 | 해당 없음 | 읽지 않음 |
| [HaozhiQi/hora](https://github.com/HaozhiQi/hora) | 감사 대상. 교사-학생 구조의 원형 | NOASSERTION | 262 | 2026-07-08 | 해당 없음 | 읽지 않음 |
| [tylerlum/simtoolreal](https://github.com/tylerlum/simtoolreal) | 감사 대상 SimToolReal | MIT | 439 | 해당 없음 | 해당 없음 | 읽지 않음 |

- 저장소에는 모델 체크포인트, 물체 에셋, 생성된 grasp 캐시가 없습니다. README는 `make_data`와 `func_lygra`로 직접 만들라고 안내합니다(`04` §1).
- 실로봇 배포 코드(`isaacgymenvs/deploy/`)는 있으나 Sharpa SDK(`/opt/sharpa-wave-sdk`)는 외부 의존입니다.
